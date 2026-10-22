"""Load a Phonon model on the CPU engine and turn audio into text.

This is the CPU twin of `fermion/_speech/engine.py`: it presents the same
surface (`load()`, then `.transcribe()` / `.transcribe_array()` /
`.describe()`), so `transcribe`, `serve` and the live-stream plumbing need
no branching beyond the one engine dispatch in `_speech/backends.py`.

THE DECODE CONFIGURATION IS FIXED. Greedy argmax at temperature 0.0, at most
512 new tokens per segment, no repetition penalty, EOS {151643, 151645},
special tokens skipped, result stripped — transcription through this engine
is deterministic, and there are no knobs because there is no configuration
that uses one.

LONG AUDIO: a file up to ``segment.SINGLE_SHOT_MAX_S`` (35 s) is decoded in
one call. Longer audio goes through the segmenter both engines share
(`fermion/_speech/segment.py`): 25-35 s windows cut at pauses, one decode
per window, transcripts joined with single spaces — the same shape one file
gets everywhere else. (0.1.21-0.1.22 split on every ~0.7 s pause above 40 s;
0.1.23 uses the shared windowing so both engines segment identically.)
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

from fermion._brand import BRAND
from fermion._speech.engine import _long_audio_note

SAMPLE_RATE = 16_000

# ---- decode configuration (fixed; do not edit without a transcript gate) --
MAX_NEW_TOKENS = 512
VARIANT = "lut"
HEAD = "int8t"
ATTN = "c"
ATTN_MAX_SEQ = 2048
CONV_CHUNK = 14
SPIN_ITERS = 60_000


def _thread_policy() -> dict:
    """Per-platform thread + prefill configuration (the verified settings)."""
    n = os.cpu_count() or 4
    if platform.system() == "Darwin":
        return {"kernel": 6, "torch": 6, "decode_torch": 6,
                "prefill_attn": "torch"}
    k = max(1, min(n, 16))
    return {"kernel": k, "torch": k, "decode_torch": 1, "prefill_attn": "c"}


#: Catalog profile key -> how that published artifact stores its audio
#: tower / embedding (the decoder format is shared by all three). The
#: manifest format string is what `resolve_profile` checks a directory
#: against.
PROFILES = {
    "audio6": {"manifest_format": "sttg1a-armc2-head8audio6-v1",
               "tower": "6-bit affine", "embedding": "8-bit affine"},
    "parity": {"manifest_format": "sttg1a-armc2-two-plane-ternary-v1"
                                  "+slim-metadata-v1+ten-base5-per-24bit-v1",
               "tower": "dense", "embedding": "dense"},
    "micro": {"manifest_format": "sttg1a-armc2-hybrid4-v1"
                                 "+slim-metadata-v1+ten-base5-per-24bit-v1",
              "tower": "4-bit affine", "embedding": "4-bit affine"},
}
DEFAULT_PROFILE = "audio6"   # the catalog default (Phonon-1)


def resolve_profile(model_dir: Path, profile: str | None) -> str:
    """The profile key a model directory holds, cross-checked with the
    caller's claim (the catalog's `--model` resolution) — refuse, never
    guess, on a mismatch or an unrecognised layout."""
    manifest = json.loads((Path(model_dir) / "packed_manifest.json").read_text())
    fmt = str(manifest.get("format", ""))
    found = next((k for k, v in PROFILES.items() if v["manifest_format"] == fmt),
                 None)
    if found is None:
        raise SystemExit(
            "this model directory is not one of the published Phonon-1 "
            "models (Phonon-1, Phonon-1 Big, Phonon-1 Micro), so the CPU "
            "engine cannot load it")
    if profile is not None and profile != found:
        raise SystemExit(
            f"--model names the {profile!r} model but the directory holds "
            f"{found!r} — point --model at the matching model")
    return found


class CpuSpeech:
    """One loaded model; owns the kernel pool for the process lifetime."""

    kind = "speech"
    #: Decodes through this engine run greedy with no repetition penalty —
    #: reported here so `describe()` and callers see the truth, not the MLX
    #: engine's setting.
    default_repetition_penalty = None

    def __init__(self, model_dir: Path, *, profile: str, quiet: bool = True):
        from fermion import _compat
        _compat.prepare_torch_stack()   # shields first; torch loads next
        import torch

        from fermion._speech import gate_cpu
        from fermion._speech._engine_cpu import driver as _driver

        self.path = Path(model_dir)
        self.profile = profile
        self.backend = "cpu"
        policy = self._policy = _thread_policy()
        torch.set_num_threads(policy["torch"])

        started = time.perf_counter()
        library_path = gate_cpu.binary_path(verify=True)
        loader, fold4_cache = _driver.enable_fold4_cache()
        from fermion._speech._engine_cpu.model import STATS, load_phonon_cpu
        self._stats = STATS
        bundle = load_phonon_cpu(self.path, variant=VARIANT,
                                 nthreads=policy["kernel"],
                                 packed_decode=False)
        lib = _driver.CpuDriverLibrary(library_path,
                                       nthreads=policy["kernel"],
                                       spin_iters=SPIN_ITERS)
        fused = _driver.build_fused_handles(lib, self.path, loader=loader)
        decoder = _driver.DriverDecoder(bundle, lib, fused, VARIANT, STATS,
                                        attn=ATTN, attn_max_seq=ATTN_MAX_SEQ)
        fold4_cache.clear()
        _driver.install_quant_head(bundle, lib, HEAD)
        _driver.install_conv_fix(bundle, CONV_CHUNK)
        _driver.install_prefill_fusion(bundle)
        self._pattn_counters = None
        if policy["prefill_attn"] == "c":
            self._pattn_counters = _driver.install_c_prefill_attention(
                bundle, lib, max_seq=ATTN_MAX_SEQ)

        self.bundle = bundle
        self.model = bundle.model
        self.lib = lib
        self.decoder = decoder
        self._tokenizer = bundle.processor.tokenizer
        self._feature_extractor = bundle.processor.feature_extractor
        from fermion._speech._engine_cpu.graph import _qwen_backend_modules
        _, _, processing = _qwen_backend_modules()
        self._feat_out_lengths = processing._get_feat_extract_output_lengths
        release_load_memory()
        self.load_seconds = time.perf_counter() - started
        if not quiet:
            print(f"[{BRAND}] loaded {profile} on the CPU engine in "
                  f"{self.load_seconds:.1f}s — {policy['kernel']} kernel "
                  f"threads, deterministic decode", file=sys.stderr)

    # ------------------------------------------------------------ decoding
    def _decode_segment(self, wav):
        """One gated decode: audio array (one window) -> (text, n_tokens,
        MAX_NEW_TOKENS). The token count is what `truncated` is judged
        against."""
        import torch

        policy = self._policy
        feats = self._feature_extractor(
            wav, sampling_rate=SAMPLE_RATE, return_attention_mask=True,
            truncation=False, padding=True, return_tensors="pt")
        input_features = feats["input_features"].to(torch.float32)
        feature_attention_mask = feats["attention_mask"]
        num_audio_tokens = int(self._feat_out_lengths(
            feature_attention_mask.sum(-1)).item())
        prompt = (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|>"
            + "<|audio_pad|>" * num_audio_tokens
            + "<|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\nlanguage English<asr_text>")
        input_ids = torch.tensor([self._tokenizer.encode(prompt)],
                                 dtype=torch.long)
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            out = self.model.thinker(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                use_cache=True)
            first = int(torch.argmax(out.logits[0, -1]).item())
            self.decoder.start_utterance(out.past_key_values, prompt_len)
            self._stats.phase = "decode"
            switch = policy["decode_torch"] != policy["torch"]
            if switch:
                torch.set_num_threads(policy["decode_torch"])
            try:
                new_tokens = self.decoder.generate_greedy(
                    first, prompt_len, max_new_tokens=MAX_NEW_TOKENS)
            finally:
                if switch:
                    torch.set_num_threads(policy["torch"])
        text = self._tokenizer.decode(torch.tensor(new_tokens),
                                      skip_special_tokens=True).strip()
        return text, len(new_tokens), MAX_NEW_TOKENS

    def transcribe_array_detailed(self, audio, **_ignored):
        """Decode one array, long or short, into a `segment.Transcript`.

        Extra keyword arguments (the MLX engine's repetition-penalty knob)
        are accepted and ignored: this engine's decode is fixed greedy.
        Up to 35 s is one `_decode_segment` call; longer audio is windowed
        by the segmenter both engines share.
        """
        from fermion._speech import segment

        return segment.transcribe_windows(audio, self._decode_segment)

    def transcribe_array(self, audio, **_ignored):
        """(text, decode_seconds, audio_seconds) for one float32 array."""
        return self.transcribe_array_detailed(audio).triple()

    def transcribe_detailed(self, audio_path, **_ignored):
        """Transcribe an audio FILE into a `segment.Transcript`."""
        from fermion._speech.engine import read_audio
        return self.transcribe_array_detailed(read_audio(audio_path))

    def transcribe(self, audio_path, **_ignored):
        """Transcribe an audio FILE. Returns (text, decode_seconds, duration)."""
        return self.transcribe_detailed(audio_path).triple()

    # ---------------------------------------------------------------- info
    def describe(self) -> dict:
        """The configuration this process is actually running."""
        policy = self._policy
        return {
            "kind": "speech",
            "engine": "cpu",
            "profile": self.profile,
            "backend": "cpu",
            "model_dir": str(self.path),
            "deterministic": True,
            "temperature": 0.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": None,
            "kernel_threads": policy["kernel"],
            "torch_threads": policy["torch"],
            "decode_torch_threads": policy["decode_torch"],
            "audio_tower_bits": None,   # dequantized to fp32 at load
            "load_seconds": round(self.load_seconds, 3),
            "sample_rate": SAMPLE_RATE,
            "long_audio": _long_audio_note(),
        }


def release_load_memory() -> None:
    """Hand the load-time transients back to the operating system (0.1.25).

    Loading builds and frees a few gigabytes of temporaries (the dense state
    before it is copied into the graph, the packed matrices before the kernel
    library copies them). On glibc those freed blocks mostly stay in the
    process heap — the allocator only trims from the top — so the process
    kept a peak-sized RSS for its whole life. `malloc_trim(0)` returns every
    free page in the heap to the kernel; it changes no value anywhere.
    Nothing to do on macOS / Windows (their allocators return pages by
    themselves; measured RSS there is already the steady state).
    """
    import gc
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except (OSError, AttributeError):      # musl or no libc.so.6: fine
            pass


def load(model_dir, *, profile: str | None = None, backend: str = "cpu",
         quiet: bool = True) -> CpuSpeech:
    """Gate first (`gate_cpu.require()` upstream), then load.

    All three published models resolve here: `--model phonon` (audio6, the
    default), `--model phonon-1-big` (parity) and `--model phonon-1-micro`
    (micro) — the catalog maps the name to the profile key, and the
    directory's own manifest must agree.
    """
    claim = profile if profile in PROFILES else None
    key = resolve_profile(Path(model_dir), claim)
    return CpuSpeech(Path(model_dir), profile=key, quiet=quiet)

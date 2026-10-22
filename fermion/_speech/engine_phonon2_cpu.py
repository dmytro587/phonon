"""Phonon-2 on plain CPU (tier 1): the five-value container expanded EXACTLY to dense fp32 inside the stock
`transformers` ParakeetForTDT graph, decoded by a greedy TDT loop, on Torch's CPU BLAS.

Same `SpeechModel` contract as `engine.py` / `engine_phonon2.py` (transcribe / transcribe_detailed /
transcribe_array_detailed / describe), so `transcribe`, `serve`, `listen`, `speech bench` work unchanged.

What this is and is not (measured, runtime lane 2026-09-28):
  * a COVERAGE path: Linux x86-64 / aarch64, Windows x64, Intel Macs (0.2.8), and a Mac without MLX get a correct Phonon-2 from the
    same 164 MB download; resident memory is ~2.5 GB (fp32 weights), not the packed 0.4 GB of the MLX engine;
  * speed: 13x realtime on 12 vCPU (Intel Cascade Lake, 76 ms of compute per second of audio), 10x on 4 threads;
    the packed five-value GEMM engine that would change this is priced, not built (report §4.2);
  * numerics: fp32 dense arithmetic; transcripts are the same-accuracy class as the MLX packed path (bf16 two-plane),
    NOT word-identical to it (the identity count on 40 clips is in the report). WER on the official meter matches
    the fp32 reference read boxes.
The log-mel front end is the HF ParakeetFeatureExtractor's arithmetic written out here (pre-emphasis 0.97, 512-pt
STFT / hop 160 / Hann 400, slaney mel 128, log(x + 2^-24), per-feature normalisation over the true length) so the
package does not need librosa; the slaney filterbank is a numpy port checked equal to librosa's matrix.
Tokens are decoded with the vocabulary in the model directory's `config.json` (the mlx-community NeMo-shaped
config), exactly as the MLX engine does ("▁" -> space), so no HF tokenizer files are needed.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from fermion._brand import BRAND
from fermion._speech.engine import SAMPLE_RATE, SpeechModel
from fermion._speech.hotwords import Phonon2HotwordsMixin

CONTAINER = "model.fermion"
CONFIG = "config.json"

# ---- HF ParakeetTDT config of the base architecture (nvidia/parakeet-tdt-0.6b-v3), values only -----------------
HF_CONFIG = {
    "architectures": ["ParakeetForTDT"], "model_type": "parakeet_tdt", "blank_token_id": 8192, "vocab_size": 8193,
    "decoder_hidden_size": 640, "num_decoder_layers": 2, "hidden_act": "relu", "max_symbols_per_step": 10,
    "durations": [0, 1, 2, 3, 4], "pad_token_id": 2, "is_encoder_decoder": True, "initializer_range": 0.02,
    "encoder_config": {"model_type": "parakeet_encoder", "hidden_size": 1024, "intermediate_size": 4096,
                       "num_hidden_layers": 24, "num_attention_heads": 8, "num_key_value_heads": 8,
                       "num_mel_bins": 128, "conv_kernel_size": 9, "subsampling_conv_channels": 256,
                       "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2, "subsampling_factor": 8,
                       "max_position_embeddings": 5000, "scale_input": False, "attention_bias": False,
                       "convolution_bias": False, "hidden_act": "silu", "dropout": 0.0, "attention_dropout": 0.0,
                       "activation_dropout": 0.0, "dropout_positions": 0.0, "layerdrop": 0.0,
                       "initializer_range": 0.02},
}
N_FFT, HOP, WIN, N_MELS, PREEMPH = 512, 160, 400, 128, 0.97
LOG_GUARD = 2.0 ** -24


# ---- slaney mel filterbank (librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney")) --------
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64); f_sp = 200.0 / 3; min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-30) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64); f_sp = 200.0 / 3; min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filters(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f); ramps = np.subtract.outer(mel_f, fftfreqs)
    w = np.zeros((n_mels, 1 + n_fft // 2))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]; upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


def _special(piece: str) -> bool:
    """mlx-audio's rule (tokenizer._is_special_piece), so both engines drop the same pieces."""
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in ("<unk>", "<pad>")


class Phonon2CpuSpeechModel(Phonon2HotwordsMixin, SpeechModel):
    kind = "speech"

    def _runtime_string(self) -> str:
        pk = self.decode.get("packed") or {}
        enc = ("packed one-dot int8 encoder" if pk.get("onedot") else "packed two-dot int8 encoder") if pk.get("modules") else "dense fp32 encoder (torch)"
        if getattr(self, "_cenc", None) is not None:
            enc = "C encoder (packed one-dot int8)"
        loop = "C TDT loop" if getattr(self, "_ctdt", None) is not None else "torch TDT loop"
        tier = f", {pk['tier']} tier" if pk.get("tier") else ""
        return f"cpu: {enc} + {loop}{tier}"

    def describe(self) -> dict:
        return {
            "kind": "speech", "engine": "cpu", "profile": self.profile, "backend": self.backend, "model_dir": str(self.path),
            "runtime": self._runtime_string(),
            "activations": "bfloat16 autocast (AMX)" if getattr(self, "_bf16", False) else "float32", "threads": self.decode.get("threads"),
            "five_value_modules": self.decode.get("five_value_modules"),
            "packed": self.decode.get("packed"), "c_tdt_loop": getattr(self, "_ctdt", None) is not None, "ctdt_error": self.decode.get("ctdt_error"),
            "c_encoder": getattr(self, "_cenc", None) is not None, "cenc_error": self.decode.get("cenc_error"),
            "selftest": self.decode.get("selftest"),
            "load_seconds": round(self.load_seconds, 2),
            **self._hotword_describe(),
            "note": "packed one-dot int8 encoder + C TDT loop where the platform binary loads (46x on 8 Zen-5 cores, WER = fp32 "
                    "on 2,939 test-other clips), else fp32 tier-1 (21x); same-accuracy class as the MLX engine",
        }

    # ---- front end -------------------------------------------------------------------------------------------
    def _log_mel(self, wave):
        import torch
        x = torch.as_tensor(np.asarray(wave, dtype=np.float32))[None]
        x = torch.cat([x[:, :1], x[:, 1:] - PREEMPH * x[:, :-1]], dim=1)
        st = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=self._window, return_complex=True, pad_mode="constant")
        mag = torch.view_as_real(st); mag = torch.sqrt(mag.pow(2).sum(-1)).pow(2)
        mel = torch.log(self._melf @ mag + LOG_GUARD).permute(0, 2, 1)             # [1, T, 128]
        n = mel.shape[1]
        mean = mel.mean(1, keepdim=True); var = ((mel - mean) ** 2).sum(1) / (n - 1)
        mel = (mel - mean) / (torch.sqrt(var).unsqueeze(1) + 1e-5)
        return mel, torch.ones((1, n), dtype=torch.long)

    # ---- decode ----------------------------------------------------------------------------------------------
    def _decode_single(self, audio, repetition_penalty: float):
        import torch
        m = self.model; cfg = m.config
        blank, V = cfg.blank_token_id, cfg.vocab_size
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=self._bf16):
            feats, am = self._log_mel(audio)
            cenc = getattr(self, "_cenc", None)
            if cenc is not None:                      # whole encoder in C (0007b); one utterance, no padding
                enc = torch.from_numpy(cenc.forward(feats[0].float().numpy()))[None]; T = int(enc.shape[1])
            else:
                enc = m.encoder(input_features=feats, attention_mask=am).last_hidden_state
                T = int(m._get_subsampling_output_length(am.sum(-1))[0])
            encp = m.encoder_projector(enc)[0]
            ctdt = getattr(self, "_ctdt", None)
            auto = self._hotword_automaton(self._vocab) if self.hotwords else None
            if ctdt is not None and (auto is None or getattr(ctdt, "can_bias", False)):
                if auto is None:
                    ids, frames, durs = ctdt.decode_timed(encp[:T].float().numpy())
                else:
                    ids, frames, durs = ctdt.decode_timed(encp[:T].float().numpy(), bias=auto)
                return self._finish(ids, frames, durs, T)
            # The torch loop: the default where the C loop is not loaded, and the hotword path for a C loop binary
            # without the bias entry point (an override library older than 0.2.9).
            bias_np = None; bstate = 0
            t = 0; last = blank; ids = []; frames = []; durs = []; nsym = 0; it = 0
            h = torch.zeros(m.decoder.lstm.num_layers, 1, m.decoder.lstm.hidden_size); c = torch.zeros_like(h)
            while t < T and it < self._max_sym * T + 16:
                emb = m.decoder.embedding(torch.tensor([[last]]))
                out, (h2, c2) = m.decoder.lstm(emb, (h, c)); dec = m.decoder.decoder_projector(out[0])
                logits = m.joint.head(m.joint.activation(encp[t][None] + dec))[0]
                if auto is None:
                    tok = int(logits[:V].argmax())
                else:                                     # hotwords: bonus on the pieces that advance a listed word
                    bias_np = auto.bonus(bstate)[:V]
                    tok = int((logits[:V].float() + torch.from_numpy(bias_np)).argmax())
                dur = int(self._durations[int(logits[V:].argmax())]); it += 1
                if tok == blank and dur == 0:
                    dur = 1
                if tok != blank:
                    ids.append(tok); frames.append(t); durs.append(dur); last = tok; h, c = h2, c2
                    if auto is not None:
                        bstate = auto.step(bstate, tok)
                if dur == 0:
                    nsym += 1
                    if nsym >= self._max_sym:
                        dur = 1; nsym = 0
                else:
                    nsym = 0
                t += dur
        return self._finish(ids, frames, durs, T)

    #: seconds per encoder frame = subsampling 8 x hop 160 samples at 16 kHz; the same expression order as the MLX graph
    #: (`frame * sub / sr * hop`) so both engines print the same millisecond for the same frame.
    _SUB, _HOP = 8, 160

    def _finish(self, ids, frames, durs, T):
        """Text (exactly as every earlier release built it) + the per-token timings for `segment.transcribe_windows`.
        `frames`/`durs` are None when the loaded C binary predates the timed decode (an override library): then the text
        is unchanged and the transcript simply carries no word timings."""
        text = "".join(self._vocab[i] for i in ids if i < len(self._vocab) and not _special(self._vocab[i])).replace("\u2581", " ").strip()
        tokens = None
        if frames is not None and durs is not None:
            tokens = []
            for i, fr, du in zip(ids, frames, durs):
                if i >= len(self._vocab) or _special(self._vocab[i]):
                    continue
                tokens.append((self._vocab[i].replace("\u2581", " "), fr * self._SUB / SAMPLE_RATE * self._HOP,
                               du * self._SUB / SAMPLE_RATE * self._HOP))
        return text, len(ids), T + 1_000_000, tokens   # no token budget on a transducer: `truncated` can never trigger


def _use_bf16() -> bool:
    env = os.environ.get("FERMION_CPU_BF16")
    if env is not None:
        return env == "1"
    try:
        flags = [l for l in open("/proc/cpuinfo") if l.startswith("flags")][0]
        return "amx_bf16" in flags
    except Exception:
        return False


def _physical_cores_linux() -> int | None:
    """Distinct physical cores in this process's affinity set, restricted to the BIG cores on heterogeneous parts.

    SMT siblings share a (package, core_id) pair and only slow an int8 GEMM down (two threads fight for one VNNI/NEON
    unit); on big.LITTLE / DynamIQ ARM (Snapdragon X, RK3588, Apple silicon under Linux) the little cores set the pace
    of a barrier-synchronised thread pool, so only cpus whose `cpu_capacity` equals the maximum are counted. None when
    the sysfs topology is unreadable (containers with a masked /sys, non-Linux)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None
    base = "/sys/devices/system/cpu"
    def _read(path):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            return None
    caps = {c: _read(f"{base}/cpu{c}/cpu_capacity") for c in cpus}
    if all(v is not None for v in caps.values()) and len({int(v) for v in caps.values()}) > 1:
        top = max(int(v) for v in caps.values())
        cpus = [c for c in cpus if int(caps[c]) == top]
    cores = set()
    for c in cpus:
        core = _read(f"{base}/cpu{c}/topology/core_id"); pkg = _read(f"{base}/cpu{c}/topology/physical_package_id")
        if core is None:
            return None
        cores.add((pkg, core))
    return len(cores) or None


def _physical_cores_windows() -> int | None:
    import subprocess
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                             capture_output=True, text=True, timeout=8).stdout.strip()
        return int(out) or None
    except Exception:                                    # noqa: BLE001
        return None


def _default_threads() -> int:
    """Thread count the engine starts with (FERMION_CPU_THREADS overrides). Mac: the performance cores (M5 4P+6E: 4 threads
    17x realtime, 8 = 7x, the efficiency cores set the pace). Linux aarch64: one thread per physical big core (mixed-core
    parts: cpu_capacity picks the big cluster). x86 / Windows: one thread per physical core from 6 cores up, every logical
    cpu on smaller parts (measured: SMT threads win on 4-core parts, lose 25 % on 8-core Zen 5); at most 16."""
    import platform, subprocess
    system = platform.system()
    if system == "Darwin" and platform.machine().lower() == "x86_64":   # 0.2.8 Intel Macs: the x86 rule below, read through sysctl
        try:
            def sc(n): return int(subprocess.run(["sysctl", "-n", n], capture_output=True, text=True, timeout=3).stdout.strip() or 0)
            phys, logical = sc("hw.physicalcpu"), sc("hw.logicalcpu")
            if phys > 0 and logical > 0:
                return max(1, min(phys if phys >= 6 else logical, 16))
        except Exception:
            pass
    elif system == "Darwin":
        try:
            p = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"], capture_output=True, text=True, timeout=3).stdout.strip() or 0)
            if p > 0:
                return p
        except Exception:
            pass
    logical = os.cpu_count() or 4
    phys = _physical_cores_linux() if system == "Linux" else (_physical_cores_windows() if system == "Windows" else None)
    if not phys or phys <= 0 or phys > logical:
        return max(1, min(logical, 16))
    # Measured 2026-09-30 (single stream, 5 s utterance): 8-core/16-thread Zen 5 -- 8 threads 157x / 33 ms vs 16 threads
    # 118x / 45 ms (SMT siblings fight for the one int8 unit per core and the barrier waits for the slowest); 4-core/8-thread
    # Cascade Lake -- 8 threads 38.9x / 124 ms vs 4 threads 35.8x / 133 ms (on a small part the extra threads hide the glue
    # latency); 4c/8t Sapphire Rapids a tie. ARM servers have no SMT; mixed-core ARM wants the big cores only.
    arm = platform.machine() in ("aarch64", "arm64")
    if arm or phys >= 6 or _has_amx() or _is_amd():
        return max(1, min(phys, 16))
    return max(1, min(logical, 16))


def _is_amd() -> bool:
    """Zen: the SMT sibling costs, even on small parts (c4d-standard-8, 4c/8t: 4 threads 102.6x vs 8 threads 94.1x, 2026-09-30);
    on small Intel parts the sibling helps (Cascade Lake 4c/8t: +8 % with 8), so Intel keeps logical cpus below 6 cores."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("vendor_id"):
                    return "AMD" in line
    except OSError:
        pass
    return False


def _has_amx() -> bool:
    """Sapphire Rapids+: the two SMT threads of a core share one AMX unit, so on the AMX tier 4 threads beat 8 by 15 % on a
    4c/8t part (measured 2026-09-30) -> physical cores there regardless of size."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("flags"):
                    return "amx_int8" in line.split()
    except OSError:
        pass
    return False


PLANE_CACHE = "cpu_planes_v1.bin"


def _plane_cache_key(model_dir: Path) -> str | None:
    try:
        man = json.loads((model_dir / "packed_manifest.json").read_text())
        return f"planes-v1:{man['container_sha256']}:{man['five_value_modules']}"
    except Exception:                                    # noqa: BLE001
        return None


def _plane_cache_read(model_dir: Path):
    """name -> (rows, cols, pa, pb, lo, hi) as read-only memmaps, or None when absent/stale/disabled."""
    if os.environ.get("FERMION_P2_PLANE_CACHE", "1") == "0":
        return None
    f = model_dir / PLANE_CACHE
    key = _plane_cache_key(model_dir)
    if key is None or not f.is_file():
        return None
    try:
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little"); head = json.loads(fh.read(n))
        if head.get("key") != key or 8 + n + int(head.get("payload", -1)) != f.stat().st_size:
            return None
        base = 8 + n; mm = np.memmap(f, dtype=np.uint8, mode="r"); out = {}
        for e in head["modules"]:
            o, i = e["rows"], e["cols"]; off = base + e["off"]; q = i // 4
            pa = mm[off: off + o * q].reshape(o, q); off += o * q
            pb = mm[off: off + o * q].reshape(o, q); off += o * q
            lo = mm[off: off + 2 * o].view(np.float16); off += 2 * o
            hi = mm[off: off + 2 * o].view(np.float16)
            out[e["name"]] = (o, i, pa, pb, lo, hi)
        return out
    except Exception:                                    # noqa: BLE001  (a damaged cache is rebuilt, never trusted)
        return None


def _plane_cache_write(model_dir: Path, packed: dict) -> bool:
    """Write the packed planes of every five-value module beside the model (≈300 MB for Phonon-2). Atomic rename; any
    failure (read-only cache dir, disk full) is silent — the next load simply pays the decode again."""
    if os.environ.get("FERMION_P2_PLANE_CACHE", "1") == "0":
        return False
    key = _plane_cache_key(model_dir)
    if key is None:
        return False
    f = model_dir / PLANE_CACHE; tmp = f.with_suffix(".tmp")
    try:
        mods, off = [], 0
        for name, pm in packed.items():
            size = 2 * pm.pa.nbytes + 4 * pm.rows
            mods.append({"name": name, "rows": pm.rows, "cols": pm.cols, "off": off}); off += size
        head = json.dumps({"key": key, "modules": mods, "payload": off}).encode()   # size check = header + payload
        with open(tmp, "wb") as fh:
            fh.write(len(head).to_bytes(8, "little")); fh.write(head)
            for pm in packed.values():
                fh.write(pm.pa.tobytes()); fh.write(pm.pb.tobytes()); fh.write(pm.lo.tobytes()); fh.write(pm.hi.tobytes())
        os.replace(tmp, f)
        return True
    except Exception:                                    # noqa: BLE001
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _read_container(container: Path, *, packed_lib=None, planes=None):
    """ONE pass over the five-value container.

    Returns (sd, raw, packed, n_fv): `sd` = HF-named fp32 torch tensors for everything the torch graph keeps
    (decoder, joint, convs, norms, biases, and the 264 encoder linears only when `packed_lib` is None);
    `raw` = the intN decoder records (for the C TDT loop); `packed` = name -> PackedMatrix (planes + int8 tiles
    built straight from the record, the fp32 expansion of each linear is dropped module by module).
    The previous loader expanded the whole encoder to fp32 three times (state dict, packed swap, C loop) and glibc
    kept the freed 16 MB chunks in its heap: 7.9 GB resident on Linux for the packed path (measured 2026-09-28)."""
    import re
    import torch
    from fermion._speech._engine_phonon2 import load as _engine
    fc = _engine("fermion_container")
    from fermion._speech import _engine_phonon2_cpu as pk
    sd, raw, packed, n_fv = {}, {}, {}, 0

    def keep(name, arr):
        if name.endswith("num_batches_tracked"):
            v32 = float(np.asarray(arr, dtype=np.float32).reshape(-1)[0])
            sd[name] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64); return
        arr = np.ascontiguousarray(np.asarray(arr).astype(np.float32))
        if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", name):
            arr = arr[:, :, None]
        sd[name] = torch.from_numpy(arr)

    with open(container, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
        assert header["format"] == fc.FORMAT, header["format"]
        for e in header["index"]:
            k, shape = e["k"], tuple(e["shape"])
            if k == "five_value" and packed_lib is not None and planes is not None and e["n"] in planes:
                fh.seek(e["b"], 1); blob = None                    # planes come from the cache; skip the bytes
            else:
                blob = fh.read(e["b"]); assert len(blob) == e["b"]
            if k == "five_value":
                n_fv += 1
                if packed_lib is not None and planes is not None and e["n"] in planes:
                    packed[e["n"]] = pk.PackedMatrix(packed_lib, None, planes=planes[e["n"]])   # cache hit: no trit decode
                elif packed_lib is not None:
                    r = {}
                    fc._five_value(blob, shape, r)                 # fp32 view dropped here; only the record survives
                    packed[e["n"]] = pk.PackedMatrix(packed_lib, r)
                else:
                    keep(e["n"] + ".weight", fc._five_value(blob, shape, None))
            elif k.startswith("int"):
                r = {}
                keep(e["n"], fc._intn(blob, shape, int(k[3:]), r)); raw[e["n"]] = r
            elif k == "fp16":
                keep(e["n"], np.frombuffer(blob, dtype=np.float16).reshape(shape))
            else:
                raise ValueError(k)
        assert fh.read(1) == b"", "trailing bytes"
    return sd, raw, packed, n_fv


def _torch_build_note(torch) -> None:
    """Linux x86 strangers get the multi-gigabyte CUDA torch from a plain `pip install`; on a box with no NVIDIA GPU that is
    pure download and disk (the CPU engine never touches CUDA). Say so once, with the one-line fix, and move on."""
    if os.environ.get("FERMION_TORCH_NOTED") or platform.system() != "Linux":
        return
    try:
        cuda_build = getattr(torch.version, "cuda", None) is not None
        has_gpu = torch.cuda.is_available()
    except Exception:                                    # noqa: BLE001
        return
    if cuda_build and not has_gpu:
        print(f"[{BRAND}] note: this torch is the CUDA build but no NVIDIA GPU is present; the CPU engine does not need it. "
              f"A smaller install: pip install --no-deps torch --index-url https://download.pytorch.org/whl/cpu  (then pip install fermion-research again)",
              file=sys.stderr)
    os.environ["FERMION_TORCH_NOTED"] = "1"


def _trim_heap():
    """Give freed heap pages back to the OS (glibc keeps them otherwise); no-op where malloc_trim does not exist."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _parakeet_classes():
    """(ParakeetForTDT, ParakeetTDTConfig, source). transformers' own classes wherever it has them with PyTorch enabled
    (every platform through 0.2.7); the vendored torch-only copy (`fermion/_vendor/parakeet_tdt`, same module tree and
    parameter names) where it lacks them or has switched torch off: Intel Macs, whose last torch build is 2.2.2, which
    transformers 5.1+ refuse. The weights, the kernels and the decode are the same either way."""
    if os.environ.get("FERMION_P2_PARAKEET", "").lower() != "vendored":
        try:
            import transformers
            from transformers.utils import is_torch_available
            if is_torch_available() and "ParakeetForTDT" in dir(transformers):
                from transformers import ParakeetForTDT, ParakeetTDTConfig
                return ParakeetForTDT, ParakeetTDTConfig, "transformers"
        except Exception:                                # noqa: BLE001
            pass
    from fermion._vendor.parakeet_tdt import ParakeetForTDT, ParakeetTDTConfig
    return ParakeetForTDT, ParakeetTDTConfig, "vendored"


def load(model_dir, *, profile: str, backend: str, quiet: bool = False) -> Phonon2CpuSpeechModel:
    import torch
    ParakeetForTDT, ParakeetTDTConfig, model_impl = _parakeet_classes()
    model_dir = Path(model_dir)
    container = model_dir / CONTAINER; config = model_dir / CONFIG
    if not container.is_file() or not config.is_file():
        raise SystemExit(f"{model_dir} does not hold a Phonon-2 model ({CONTAINER} and {CONFIG} side by side)")
    started = time.perf_counter()
    _torch_build_note(torch)
    threads = int(os.environ.get("FERMION_CPU_THREADS", "0") or 0) or _default_threads()
    torch.set_num_threads(threads)
    cfg = ParakeetTDTConfig(**HF_CONFIG)
    # Default (2026-09-28, gate PASS on 2,939 LibriSpeech test-other clips, official meter, 8 Zen-5 cores: fp32 3.906 / packed
    # one-dot 3.898 / two-dot 3.900; one-dot 46x vs fp32 21x): the 264 encoder linears on the packed five-value int8 GEMM
    # library (one-dot = per-row int8 weights, 1 byte/weight + the 2-bit planes) + the C TDT loop wherever the platform
    # binaries load; any failure falls back to fp32 tier-1 silently.  FERMION_P2_CPU=fp32 forces tier-1; =packed keeps the
    # two-dot exact weights (PHONON2_CPU_ONEDOT=0).
    mode = os.environ.get("FERMION_P2_CPU", "onedot").lower()
    from fermion._speech import gate_cpu as _gate_cpu
    glibc_note = _gate_cpu.glibc_below_floor()
    if glibc_note and mode != "fp32":
        if not quiet and not os.environ.get("FERMION_P2_GLIBC_NOTED"):
            print(f"[{BRAND}] {glibc_note}", file=sys.stderr)
        mode = "fp32"
    lib = None; packed_info = None
    if mode in ("packed", "onedot"):
        if mode == "onedot":
            os.environ.setdefault("PHONON2_CPU_ONEDOT", "1")
        try:
            from fermion._speech import _engine_phonon2_cpu as pk
            lib = pk.load_library(); lib.phonon2_cpu_set_threads(int(threads))
        except Exception as exc:                      # never lose the decode over a speed lever
            packed_info = {"error": f"{type(exc).__name__}: {exc}", "fallback": "fp32 tier-1"}; lib = None
    planes = _plane_cache_read(model_dir) if lib is not None else None
    cache_state = "hit" if planes else ("off" if os.environ.get("FERMION_P2_PLANE_CACHE", "1") == "0" else "miss")
    try:
        sd, raw, packed, n_fv = _read_container(container, packed_lib=lib, planes=planes)
    except Exception as exc:                          # a packed read that fails falls back to the fp32 read
        if lib is None:
            raise
        packed_info = {"error": f"{type(exc).__name__}: {exc}", "fallback": "fp32 tier-1"}; lib = None
        sd, raw, packed, n_fv = _read_container(container, packed_lib=None)
    if packed and cache_state == "miss":
        cache_state = "written" if _plane_cache_write(model_dir, packed) else "unwritable"
    # Parameters are built on the meta device and ASSIGNED from the container (no 2.4 GB random init, no copy); the one
    # non-persistent buffer (relative-position inv_freq) is rebuilt on the CPU afterwards.
    with torch.device("meta"):
        model = ParakeetForTDT(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith("num_batches_tracked") and k[:-len(".weight")] not in packed]
    if missing or unexpected:
        raise SystemExit(f"Phonon-2 CPU load: missing {missing[:5]} unexpected {list(unexpected)[:5]}")
    if packed:
        try:
            n_sw = pk.swap_encoder(model, packed, list(packed), lib)
            from fermion._speech import _cpu_features as _cf
            kid = int(lib.phonon2_cpu_kernel_in_use())
            packed_info = {"modules": n_sw, "kernel_id": kid, "kernel": _cf.KERNEL_NAMES.get(kid, "?"), "tier": pk._plan()["tier"],
                           "library": str(pk.default_library().name), "onedot": os.environ.get("PHONON2_CPU_ONEDOT", "0") == "1", "plane_cache": cache_state}
        except Exception as exc:
            raise SystemExit(f"Phonon-2 CPU load: packed swap failed after a packed read: {type(exc).__name__}: {exc}")
    pe = model.encoder.encode_positions
    model.encoder.encode_positions = type(pe)(pe.config)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if left:
        raise SystemExit(f"Phonon-2 CPU load: tensors left on the meta device: {left[:5]}")
    model.eval()
    del sd
    vocab = json.loads(config.read_text()).get("vocabulary") or json.loads(config.read_text())["joint"]["vocabulary"]
    speech = Phonon2CpuSpeechModel(model, path=model_dir, profile=profile, backend=backend,
                                   decode={"five_value_modules": n_fv, "threads": threads, "packed": packed_info, "model_impl": model_impl},
                                   load_seconds=time.perf_counter() - started)
    speech._vocab = list(vocab)
    # bf16 autocast only where the CPU has AMX-bf16 (Sapphire Rapids+): measured 30x -> 46.6x at the same WER (3.187, 78/80 texts)
    # on a Xeon 8481C; on Cascade Lake bf16 autocast is 2x SLOWER, so it is gated on the flag. FERMION_CPU_BF16=0/1 overrides.
    speech._bf16 = _use_bf16()
    # FERMION_P2_CPU_TDT=c: the greedy TDT loop in C (int6 decoder tables exact, int8 activations; 39/40 texts = torch loop on
    # the M5 gate, loop 1.9 ms per audio-second vs ≈35 in torch). Falls back to the torch loop on any error.
    speech._ctdt = None
    if os.environ.get("FERMION_P2_CPU_TDT", "c").lower() == "c" and not glibc_note:
        try:
            from fermion._speech import _engine_phonon2_cpu as pk
            biases = {n: t.detach().numpy() for n, t in model.state_dict().items() if n.startswith(("decoder.", "joint.")) and "bias" in n}
            speech._ctdt = pk.CTdtDecoder(model, raw, biases, cfg.vocab_size, threads)
        except Exception as exc:
            speech._ctdt = None; speech.decode["ctdt_error"] = f"{type(exc).__name__}: {exc}"
    # FERMION_P2_CPU_ENC=c (default): the whole encoder in C where the superset library (libphonon2_enc-*) loaded and the
    # packed one-dot modules are in place; any failure keeps the torch encoder over the same packed GEMMs.
    speech._cenc = None
    if os.environ.get("FERMION_P2_CPU_ENC", "c").lower() == "c" and packed and lib is not None and hasattr(lib, "phonon2_enc_abi_version"):
        try:
            speech._cenc = pk.CEncoder(lib, model)
        except Exception as exc:
            speech._cenc = None; speech.decode["cenc_error"] = f"{type(exc).__name__}: {exc}"
    # Self-test (0007b.1): a binary that is shipped and pinned for this platform must have loaded. A silent fallback here
    # would hide a packaging defect behind a slower path, so it fails the load unless the user opted out of that binary.
    try:
        from fermion._speech import _engine_phonon2_cpu as pk
        shipped = pk.shipped_binaries()
    except Exception:
        shipped = {}
    if glibc_note:
        shipped = {k: False for k in shipped}        # below the floor the binaries are not for this machine: no self-test refusal
    speech.decode["selftest"] = {"shipped": shipped, "gemm_loaded": bool(packed), "tdt_loaded": speech._ctdt is not None, "enc_loaded": speech._cenc is not None,
                                 "glibc_note": glibc_note}
    if shipped.get("tdt") and speech._ctdt is None and os.environ.get("FERMION_P2_CPU_TDT", "c").lower() == "c":
        raise SystemExit(f"Phonon-2 CPU load: the C TDT-loop binary is shipped for this platform but did not load: {speech.decode.get('ctdt_error')}")
    if shipped.get("gemm") and not packed and mode in ("packed", "onedot"):
        raise SystemExit(f"Phonon-2 CPU load: the packed GEMM binary is shipped for this platform but did not load: {(packed_info or {}).get('error')}")
    del raw
    _trim_heap()
    speech._durations = list(cfg.durations); speech._max_sym = getattr(cfg, "max_symbols_per_step", 10) or 10
    speech._window = torch.hann_window(WIN, periodic=False); speech._melf = torch.from_numpy(mel_filters())
    if not quiet:
        print(f"[{BRAND}] loaded {profile} on the CPU engine in {speech.load_seconds:.1f}s — {speech._runtime_string()[5:]}, "
              f"{n_fv} five-value linears, {threads} threads", file=sys.stderr)
    return speech

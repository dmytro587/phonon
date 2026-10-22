"""Versioned inference path for the immutable V18 checkpoint.

This module never rewrites weights.  Its default behaviour is one execution
change, verified byte-identical to the reference path on a 400-utterance gate:

* prompt prefill without projecting every audio position through the 151,936-row
  tied vocabulary head; only the final prompt position needs logits.

Two further changes stay opt-in and off by default:

* ``audio_blocks`` independently encodes/caches closed 8-second audio-attention
  blocks — small gain, transcript drift;
* ``compiled_decode`` runs a shapeless compiled fixed-capacity one-token decoder
  step — exact but measurably slower.

The optional fused QMV flag remains experimental and is gated separately.

Batch and streaming decoding share a single ``_iter_tokens`` generator, so a
streamed transcript is byte-identical to the batch transcript by construction.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
    StreamingResult,
    split_audio_into_chunks,
)
from mlx_audio.stt.utils import load_audio
from mlx_lm.sample_utils import make_logits_processors

from v18_runtime import load_v18


@dataclass
class AudioCacheStats:
    hits: int = 0
    misses: int = 0


class ClosedBlockAudioCache:
    """Cache independent Qwen audio-attention blocks for a growing utterance.

    Qwen's audio tower chunks 10-ms mel frames in groups of 100 and applies a
    block-diagonal attention mask across eight such chunks.  A completed
    800-frame block can therefore be reused when later microphone partials add
    audio to the tail.  Short utterances keep exactly one block.
    """

    feature_block = 800

    def __init__(self):
        self._entries: dict[int, tuple[bytes, mx.array]] = {}
        self._last_frames = 0
        self.stats = AudioCacheStats()

    def reset(self) -> None:
        self._entries.clear()
        self._last_frames = 0
        self.stats = AudioCacheStats()

    @staticmethod
    def _fingerprint(features: np.ndarray) -> bytes:
        return hashlib.sha256(features.tobytes(order="C")).digest()

    def encode(self, model, input_features: mx.array, attention_mask: mx.array) -> mx.array:
        frames = int(np.asarray(attention_mask.sum(axis=-1))[0])
        if frames < self._last_frames:
            self.reset()
        self._last_frames = frames

        outputs: list[mx.array] = []
        live_indices: set[int] = set()
        for index, start in enumerate(range(0, frames, self.feature_block)):
            end = min(frames, start + self.feature_block)
            block_np = np.asarray(input_features[:, :, start:end])
            fingerprint = self._fingerprint(block_np)
            cached = self._entries.get(index)
            if cached is not None and cached[0] == fingerprint:
                encoded = cached[1]
                self.stats.hits += 1
            else:
                block = mx.array(block_np)
                block_mask = mx.ones((1, end - start), dtype=attention_mask.dtype)
                encoded = model.get_audio_features(block, block_mask)
                mx.eval(encoded)
                self._entries[index] = (fingerprint, encoded)
                self.stats.misses += 1
            outputs.append(encoded)
            live_indices.add(index)

        for index in tuple(self._entries):
            if index not in live_indices:
                del self._entries[index]
        return mx.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]


class CompiledFixedCapacityDecoder:
    """Compiled one-token decoder backed by fixed-capacity KV arrays.

    The earlier experiment concatenated one K/V row per generated token.  That
    was exact, but it discarded the native cache's 256-token preallocation and
    made the compiled graph return and carry an ever-growing tensor. A later
    full-capacity masked-attention experiment changed the SDPA reduction path
    and failed the quality gate. This version keeps fixed-capacity backing
    buffers outside the graph, accepts only their active prefix, returns only
    each layer's new one-token K/V rows, and uses the same unmasked SDPA path as
    native MLX. Capacity follows the native 256-token slabs.
    """

    def __init__(self, model):
        self.model = model
        self._compiled_step = mx.compile(self._step, shapeless=True)

    def _step(
        self,
        token: mx.array,
        keys: list[mx.array],
        values: list[mx.array],
        position: mx.array,
    ) -> tuple[mx.array, list[mx.array], list[mx.array]]:
        hidden = self.model.model.embed_tokens(token.reshape((1, 1)))
        next_keys: list[mx.array] = []
        next_values: list[mx.array] = []

        # Call the underlying native RoPE primitive directly.  Unlike the
        # ``nn.RoPE`` wrapper, it accepts a scalar array offset, so the compiled
        # graph keeps the position dynamic without reimplementing RoPE math (or
        # changing BF16 rounding).
        head_dim = self.model.config.text_config.head_dim
        base = self.model.config.text_config.rope_theta

        def apply_rope(x: mx.array) -> mx.array:
            return mx.fast.rope(
                x,
                head_dim,
                traditional=False,
                base=base,
                scale=1.0,
                offset=position,
            )

        for index, layer in enumerate(self.model.model.layers):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            attn = layer.self_attn
            batch, length, _ = normed.shape

            queries = attn.q_proj(normed)
            new_keys = attn.k_proj(normed)
            new_values = attn.v_proj(normed)
            queries = queries.reshape(
                batch, length, attn.num_heads, attn.head_dim
            )
            new_keys = new_keys.reshape(
                batch, length, attn.num_kv_heads, attn.head_dim
            )
            new_values = new_values.reshape(
                batch, length, attn.num_kv_heads, attn.head_dim
            )
            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
            new_keys = attn.k_norm(new_keys).transpose(0, 2, 1, 3)
            new_values = new_values.transpose(0, 2, 1, 3)

            queries = apply_rope(queries)
            new_keys = apply_rope(new_keys)
            # Prefix concatenation is temporary attention input only. We
            # return just ``new_keys``/``new_values``; the caller writes those
            # rows into its preallocated backing buffers.
            all_keys = mx.concatenate((keys[index], new_keys), axis=2)
            all_values = mx.concatenate((values[index], new_values), axis=2)
            attended = mx.fast.scaled_dot_product_attention(
                queries,
                all_keys,
                all_values,
                scale=attn.scale,
                mask=None,
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(
                batch, length, -1
            )
            hidden = residual + attn.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            next_keys.append(new_keys)
            next_values.append(new_values)

        hidden = self.model.model.norm(hidden)
        if self.model.lm_head is not None:
            logits = self.model.lm_head(hidden)[:, -1, :]
        else:
            logits = self.model.model.embed_tokens.as_linear(hidden)[:, -1, :]
        return logits, next_keys, next_values

    def __call__(self, token, keys, values, position):
        return self._compiled_step(token, keys, values, position)


class TieredVocabularyHead:
    """Two-tier greedy argmax over the tied 151,936-row vocabulary head.

    The BF16 tied head is the single most expensive object in a V18 decode step
    (2.57 ms, 40.6% of a parity step) purely because greedy argmax has to read
    all 311 MB of it.  ``mx.argmax`` itself is free -- fused into the projection
    graph it costs nothing measurable (round 3 §4), so there is no fused-kernel
    win available.  What *is* available is reading fewer bytes for the scan:

    1. score every row against a low-bit copy of the head (87 MB at 4 bits,
       derived at load time -- the artifact on disk is unchanged);
    2. take the top ``k`` candidates;
    3. re-score only those ``k`` rows against the exact BF16 head and take the
       argmax there.

    In exact arithmetic this returns the true argmax whenever it is inside the
    low-bit top ``k``.  On real decode traces the true argmax never fell below
    rank 1 of the 4-bit scan, so k=32 is a wide margin -- but the scheme is
    still an approximation, is WER-gated, and is never the parity default.

    The re-score tier is whatever the profile's own head already is: BF16 for
    ``parity``, and the profile's 8-bit affine head for the compact family (its
    candidate rows are dequantized on the fly, which is 32 rows, not 151,936).
    So the scheme never *adds* head error to a profile -- it only changes which
    rows get read at full precision.
    """

    def __init__(self, model, *, bits: int = 4, group_size: int = 64, k: int = 32):
        embed = model.model.embed_tokens
        if model.lm_head is not None:
            raise TypeError("tiered head requires a tied output head")
        self.group_size = group_size
        self.bits = bits
        self.k = k
        self.quantized_reference = isinstance(embed, nn.QuantizedEmbedding)
        if self.quantized_reference:
            # The profile's own 8-bit head is the exact reference.  Dequantize
            # it once to build the cheap scan tier, then release the BF16 copy.
            self.ref_q = embed.weight
            self.ref_scales = embed.scales
            self.ref_biases = embed.biases
            self.ref_group_size = embed.group_size
            self.ref_bits = embed.bits
            dense = mx.dequantize(
                self.ref_q, self.ref_scales, self.ref_biases,
                group_size=self.ref_group_size, bits=self.ref_bits, mode="affine",
            )
        elif isinstance(embed, nn.Embedding):
            self.weight = embed.weight
            dense = self.weight
        else:
            raise TypeError(f"unsupported tied embedding type {type(embed)!r}")
        self.q, self.scales, self.biases = mx.quantize(
            dense, group_size=group_size, bits=bits, mode="affine"
        )
        mx.eval(self.q, self.scales, self.biases)
        if self.quantized_reference:
            del dense
            mx.clear_cache()

    @property
    def extra_bytes(self) -> int:
        return int(self.q.nbytes + self.scales.nbytes + self.biases.nbytes)

    def _exact_rows(self, candidates: mx.array) -> mx.array:
        if not self.quantized_reference:
            return self.weight[candidates]
        return mx.dequantize(
            self.ref_q[candidates],
            self.ref_scales[candidates],
            self.ref_biases[candidates],
            group_size=self.ref_group_size,
            bits=self.ref_bits,
            mode="affine",
        )

    @staticmethod
    def _penalise(values: mx.array, penalty: float) -> mx.array:
        # mlx_lm.sample_utils.make_repetition_penalty, verbatim rule.
        return mx.where(values < 0, values * penalty, values / penalty)

    def argmax(self, hidden: mx.array, token_history: mx.array | None = None,
               penalty: float | None = None, context_size: int = 0) -> mx.array:
        """Two-tier greedy argmax; with ``penalty`` the repetition penalty is
        carried inside both tiers (reports/phonon_tiered_rp_gate_20260909.md):
        the 4-bit scan is penalised at the last ``context_size`` history tokens
        before the top-k, and the exact re-score penalises the same candidates
        the same way.  Without a penalty this is the original two-tier argmax,
        byte for byte."""
        approx = mx.quantized_matmul(
            hidden,
            self.q,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )[0, 0]
        if penalty and token_history is not None:
            ctx = token_history[-context_size:]
            approx[ctx] = self._penalise(approx[ctx], penalty)
        candidates = mx.argpartition(-approx, self.k)[: self.k]
        exact = hidden[0, 0] @ self._exact_rows(candidates).T
        if penalty and token_history is not None:
            in_ctx = mx.any(candidates[:, None] == ctx[None, :], axis=1)
            exact = mx.where(in_ctx, self._penalise(exact, penalty), exact)
        return candidates[mx.argmax(exact, axis=-1)].astype(mx.int32)


#: Default additive boost, in logit units, for a hotword's next token.  The
#: founder picks the shipping value from the measured table in
#: reports/phonon_hotword_bias_20260906.md; this constant is the fallback when a
#: caller passes ``hotwords`` without a strength.
HOTWORD_STRENGTH_DEFAULT = 4.0

#: Root-arc policy.  ``"v1"`` is the design registered in
#: reports/phonon_hotword_bias_20260906.md §1.2 (every variant's first token is
#: always eligible).  ``"v2"`` is the §1.6 amendment measured after v1's control
#: slice showed self-reinforcing single-letter roots: (a) no forced-lowercase
#: variant, (b) a root arc is not eligible when its token equals the token just
#: emitted, (c) single-character first tokens carry no root boost (their
#: continuations still do).  ``"v3"`` = v2 plus the general form of (b): a
#: root arc is not eligible while the last tokens of the history spell out a
#: variant that starts with that arc's token (a hotword never re-fires straight
#: after its own completion — "Mr. Mr. Mr." cannot happen).  ``"v4"`` = v3 with
#: the re-fire block keyed on the WORD instead of the variant's first token:
#: when the history tail spells any surface form of a word, every root arc of
#: every form of that word is blocked — v3 left the leading-space form
#: eligible after the no-space form completed ("Mr." then " Mr." -> "Mr. Mr.",
#: the §2.5 residue).
HOTWORD_POLICY_DEFAULT = "v3"  # the measured shipping shape (§2.5); v1/v2 stay selectable

#: Tiered head under the repetition penalty (reports/phonon_tiered_rp_gate_20260909.md).
#: The shipped rule engaged the two-tier argmax head only with no penalty, so under
#: the app/CLI call (rp 1.05 / 96) Phonon-1 Big's loaded ``tiered4`` head never ran.
#: With this ON the penalty is carried inside both tiers (``TieredVocabularyHead.argmax``)
#: and the head engages whenever no extra logits processor (hotword bias) is present.
#: Only a profile that loads a tiered head (``tiered_head_bits`` set — Big) is affected.
#: ``DETTA_TIERED_RP=0`` restores the pre-gate rule.
TIERED_UNDER_RP = os.environ.get("DETTA_TIERED_RP", "1").strip().lower() not in ("0", "", "off", "false")


class HotwordBias:
    """Decode-time vocabulary bias: a prefix-trie over the tokenised hotwords.

    One more processor for ``OptimizedV18._apply_processors`` — the same
    ``(token_history, logits) -> logits`` shape as mlx_lm's repetition penalty.
    At every step the next token of every ACTIVE trie path gets ``+strength``
    added to its logit; a path is active when its prefix equals the tail of the
    generated history (root arcs — the first token of every variant — are
    always eligible).  Once a variant's last token has been emitted the path
    proposes nothing more.  When several paths propose the same token the boost
    is the maximum, never the sum, so no token ever moves by more than
    ``strength`` in one step.

    Word boundary: a variant with no leading space may only START where a word
    can start — after the prompt's final token (the transcript start) or after
    a token whose text ends in a non-alphanumeric character.  Leading-space
    variants are their own boundary.  The check is applied at the position
    BEFORE the matched prefix, so a glued start is not rewarded at depth either.

    Everything runs on the GPU from the ``token_history`` array — no host sync,
    so the decode loop's launch-before-read overlap is preserved.  Shapes are
    fixed at construction (``rows x max_prefix_len``), so the per-step cost is a
    handful of tiny kernels regardless of how many names are listed.

    Registered design: reports/phonon_hotword_bias_20260906.md §1.2.
    """

    #: Fill value for unused prefix slots; never equals a token id.
    _PAD = -2

    def __init__(self, tokenizer, hotwords, *, strength: float = HOTWORD_STRENGTH_DEFAULT,
                 vocab_size: int | None = None, policy: str = HOTWORD_POLICY_DEFAULT):
        if policy not in ("v1", "v2", "v3", "v4"):
            raise ValueError(f"unknown hotword policy {policy!r}")
        self.policy = policy
        self.strength = float(strength)
        self.words = self._clean(hotwords)
        # v4: which word each variant belongs to, and which words own each
        # root row (rows are shared across words when their forms start with
        # the same token, so a row may have several owners).
        variant_word: list[int] = []
        root_owner: dict[tuple, set[int]] = {}
        # The boundary table is indexed by token ids; ``len(tokenizer)`` covers
        # every id the decoder can emit and the slot after it is the sentinel
        # ("before the transcript starts").  The bonus vector must match the
        # model's logit width (151,936 for Qwen3, wider than the tokenizer), so
        # it is sized from the first logits seen unless given.
        self.n_ids = self._count_ids(tokenizer)
        self._sentinel = self.n_ids
        self.vocab_size = int(vocab_size) if vocab_size else None
        boundary = self._boundary_mask(tokenizer, self.n_ids)
        rows: dict[tuple, tuple[int, bool]] = {}
        self.variant_tokens: dict[str, list[int]] = {}
        for wi, word in enumerate(self.words):
            for variant in self.variants(word, lowercase=(policy == "v1")):
                ids = self._encode(tokenizer, variant)
                if not ids:
                    continue
                if variant not in self.variant_tokens:
                    variant_word.append(wi)
                self.variant_tokens[variant] = ids
                needs_boundary = not variant[0].isspace()
                for depth in range(len(ids)):
                    if depth == 0 and policy != "v1" and len(tokenizer.decode([ids[0]]).strip()) <= 1:
                        continue  # (c) single-character first tokens: no root boost
                    key = (tuple(ids[:depth]), ids[depth])
                    rows.setdefault(key, (ids[depth], needs_boundary))
                    if depth == 0:
                        root_owner.setdefault(key, set()).add(wi)
        self.n_rows = len(rows)
        self.max_len = max((len(k[0]) for k in rows), default=0)
        L = self.max_len
        prefix = np.full((max(self.n_rows, 1), max(L, 1)), self._PAD, dtype=np.int32)
        ignore = np.zeros_like(prefix, dtype=bool)
        plen = np.zeros(max(self.n_rows, 1), dtype=np.int32)
        nxt = np.zeros(max(self.n_rows, 1), dtype=np.int32)
        needs = np.zeros(max(self.n_rows, 1), dtype=bool)
        for r, ((pre, tok), (_, nb)) in enumerate(rows.items()):
            p = len(pre)
            if p:
                prefix[r, L - p:] = pre
            if L:
                ignore[r, : L - p] = True
            plen[r] = p
            nxt[r] = tok
            needs[r] = nb
        self._prefix = mx.array(prefix)
        self._ignore = mx.array(ignore)
        self._before_index = mx.array((L - plen).astype(np.int32))  # index into the window
        self._is_root = mx.array(plen == 0)
        # v3: every variant's full token sequence, right-aligned in a window of
        # the same width as the history window, for the "just completed" test.
        seqs = list(self.variant_tokens.values())
        W = L + 1
        vm = np.full((max(len(seqs), 1), W), self._PAD, dtype=np.int32)
        vig = np.zeros_like(vm, dtype=bool)
        vfirst = np.zeros(max(len(seqs), 1), dtype=np.int32)
        for r, ids in enumerate(seqs):
            k = min(len(ids), W)
            vm[r, W - k:] = ids[-k:]
            vig[r, : W - k] = True
            vfirst[r] = ids[0]
        self._var_mat = mx.array(vm)
        self._var_ignore = mx.array(vig)
        self._var_first = mx.array(vfirst)
        self._n_var = len(seqs)
        # v4: variant -> word, and (root row, owning word) pairs.
        own_rows: list[int] = []
        own_words: list[int] = []
        for r, (key, _) in enumerate(rows.items()):
            for wi in sorted(root_owner.get(key, ())):
                own_rows.append(r)
                own_words.append(wi)
        self._n_words = len(self.words)
        self._n_own = len(own_rows)
        self._var_word = mx.array(np.array(variant_word or [0], dtype=np.int32))
        self._own_row = mx.array(np.array(own_rows or [0], dtype=np.int32))
        self._own_word = mx.array(np.array(own_words or [0], dtype=np.int32))
        self._next = mx.array(nxt)
        self._needs_boundary = mx.array(needs)
        self._boundary = mx.array(np.concatenate([boundary, np.ones(1, dtype=bool)]))
        self._pad_window = mx.full((L + 1,), self._sentinel, dtype=mx.int32)
        self._sentinel_arr = mx.array(self._sentinel, dtype=mx.int32)
        self._zeros = (mx.zeros((self.vocab_size,), dtype=mx.float32)
                       if self.vocab_size else None)
        mx.eval(self._prefix, self._ignore, self._before_index, self._next,
                self._needs_boundary, self._boundary, self._pad_window, self._is_root,
                self._var_mat, self._var_ignore, self._var_first,
                self._var_word, self._own_row, self._own_word)

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _clean(hotwords) -> list[str]:
        seen, out = set(), []
        for word in hotwords or ():
            word = " ".join(str(word).split())
            if word and word not in seen:
                seen.add(word)
                out.append(word)
        return out

    @staticmethod
    def variants(word: str, lowercase: bool = True) -> list[str]:
        """≤ 6 surface forms: {as given, lower, first letter capitalised} x
        {no leading space, one leading space}.  The capitalised form keeps the
        rest of the word ("HMO" stays "HMO"; "kensbok" -> "Kensbok").  With
        ``lowercase=False`` (policy v2) the forced-lowercase form is dropped."""
        forms = [word, word[:1].upper() + word[1:]]
        if lowercase:
            forms.insert(1, word.lower())
        out: list[str] = []
        for form in forms:
            for surface in (form, " " + form):
                if surface not in out:
                    out.append(surface)
        return out

    # The tokenizer may be a transformers tokenizer (the CLI, the benches) or
    # the app's ``LeanTokenizer`` (``local_stt/lean_boot.py``: ``encode(text)``
    # and ``decode(ids)`` only — no ``__len__``, no ``batch_decode``, no
    # ``add_special_tokens`` keyword).  Everything below speaks the smaller
    # surface and uses the richer one when it is there.

    @staticmethod
    def _encode(tokenizer, text: str) -> list[int]:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)   # Qwen adds no specials on a bare encode
        ids = getattr(ids, "ids", ids)
        return [int(t) for t in (ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids)]

    @staticmethod
    def _count_ids(tokenizer) -> int:
        try:
            return int(len(tokenizer))
        except TypeError:
            pass
        inner = getattr(tokenizer, "_tok", None)
        if inner is not None and hasattr(inner, "get_vocab_size"):
            return int(inner.get_vocab_size(with_added_tokens=True))
        vocab = getattr(tokenizer, "vocab_size", None)
        return int(vocab) if vocab else 151_936

    @staticmethod
    def _decode_each(tokenizer, n_ids: int) -> list[str]:
        if hasattr(tokenizer, "batch_decode"):
            return tokenizer.batch_decode([[i] for i in range(n_ids)], skip_special_tokens=False)
        return [tokenizer.decode([i]) for i in range(n_ids)]

    @staticmethod
    def _boundary_mask(tokenizer, n_ids: int) -> np.ndarray:
        """``mask[id]`` is True when a new word may start right after token ``id``."""
        cached = getattr(tokenizer, "_hotword_boundary_mask", None)
        if cached is not None and len(cached) == n_ids:
            return cached
        texts = HotwordBias._decode_each(tokenizer, n_ids)
        mask = np.ones(n_ids, dtype=bool)
        for i, text in enumerate(texts):
            mask[i] = (not text) or (not text[-1].isalnum())
        try:
            tokenizer._hotword_boundary_mask = mask
        except Exception:  # pragma: no cover - a tokenizer that refuses attributes
            pass
        return mask

    # -- the processor -------------------------------------------------------

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.n_rows == 0 or self.strength == 0.0:
            return logits
        L = self.max_len
        n = int(tokens.shape[0])
        window = tokens[-(L + 1):].astype(mx.int32)
        if n < L + 1:
            window = mx.concatenate([self._pad_window[: L + 1 - n], window])
        if L:
            eq = (window[1:][None, :] == self._prefix) | self._ignore
            match = mx.all(eq, axis=1)
        else:
            match = mx.ones((self.n_rows,), dtype=mx.bool_)
        before = mx.minimum(mx.take(window, self._before_index), self._sentinel_arr)
        ok = mx.take(self._boundary, before) | ~self._needs_boundary
        match = match & ok
        if self.policy in ("v2", "v3", "v4"):
            # (b) a root arc never re-fires for the token that was just emitted
            match = match & ~(self._is_root & (self._next == window[-1]))
        if self.policy == "v3" and self._n_var:
            # (b') ...nor while the history tail spells a variant that starts
            # with that arc's token: a hotword does not restart itself.
            done = mx.all((window[None, :] == self._var_mat) | self._var_ignore, axis=1)
            blocked = mx.zeros((self._sentinel + 1,), dtype=mx.bool_).at[self._var_first].maximum(done)
            match = match & ~(self._is_root & mx.take(blocked, mx.minimum(self._next, self._sentinel_arr)))
        if self.policy == "v4" and self._n_var and self._n_own:
            # (b'') keyed on the WORD: while the history tail spells any form
            # of a word, no form of that word may start again ("Mr." done ->
            # " Mr." blocked too). Root rows shared by several words are
            # blocked when any owner just completed (a superset of v3).
            done = mx.all((window[None, :] == self._var_mat) | self._var_ignore, axis=1)
            word_done = mx.zeros((self._n_words,), dtype=mx.bool_).at[self._var_word].maximum(done)
            row_blocked = mx.zeros((self.n_rows,), dtype=mx.bool_).at[self._own_row].maximum(
                mx.take(word_done, self._own_word))
            match = match & ~row_blocked
        if self._zeros is None or self._zeros.shape[0] != logits.shape[-1]:
            self.vocab_size = int(logits.shape[-1])
            self._zeros = mx.zeros((self.vocab_size,), dtype=mx.float32)
        bonus = self._zeros.at[self._next].maximum(match.astype(mx.float32) * self.strength)
        return logits + bonus.astype(logits.dtype)[None, :]

    def describe(self) -> dict:
        return {"words": len(self.words), "variants": len(self.variant_tokens),
                "rows": self.n_rows, "max_prefix_len": self.max_len, "strength": self.strength,
                "policy": self.policy}


class OptimizedV18:
    """Drop-in V18 adapter used only by explicit experimental backends.

    ``audio_blocks`` and ``compiled_decode`` are independent switches so each
    execution change can be quality/performance gated separately.  The default
    deliberately enables neither: it is the exact prompt-head-elision control.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        model=None,
        fused_qmv: bool = False,
        audio_blocks: bool = False,
        compiled_decode: bool = False,
        fold_bits: int | None = None,
        tiered_head_bits: int | None = None,
        tiered_head_k: int = 32,
        audio_tower_bits: int | None = None,
        audio_tower_group_size: int = 128,
    ):
        if model is None:
            if path is None:
                raise ValueError("either a model directory or a loaded model is required")
            model = load_v18(
                path,
                use_fused_qmv=fused_qmv,
                fold_bits=fold_bits,
                audio_tower_bits=audio_tower_bits,
                audio_tower_group_size=audio_tower_group_size,
            )
        self.model = model
        self.tiered_head = (
            TieredVocabularyHead(model, bits=tiered_head_bits, k=tiered_head_k)
            if tiered_head_bits
            else None
        )
        self.audio_cache = ClosedBlockAudioCache()
        self.audio_blocks = audio_blocks
        self.compiled_decode = compiled_decode
        self.decoder = (
            CompiledFixedCapacityDecoder(self.model) if compiled_decode else None
        )
        self._hotword_cache: dict[tuple, HotwordBias] = {}

    def reset_stream(self) -> None:
        self.audio_cache.reset()

    def parameters(self):
        return self.model.parameters()

    @staticmethod
    def _apply_processors(
        token_history: mx.array,
        logits: mx.array,
        processors: Iterable | None,
    ) -> mx.array:
        if processors:
            for processor in processors:
                logits = processor(token_history, logits)
        return logits

    def _iter_tokens(
        self,
        audio: np.ndarray,
        *,
        max_tokens: int,
        language: str | None,
        repetition_penalty: float | None,
        repetition_context_size: int,
        system_prompt: str | None = None,
        prefill_step_size: int = 2048,
        prompt_length: list[int] | None = None,
        logits_processors: list | None = None,
    ):
        """Yield generated token ids one at a time.

        Both the batch and streaming entry points consume this single
        generator, so a streamed transcript cannot diverge from the batch
        transcript: there is only one decode loop.
        """
        m = self.model
        features, feature_mask, num_audio_tokens = m._preprocess_audio(audio)
        audio_features = (
            self.audio_cache.encode(m, features, feature_mask)
            if self.audio_blocks
            else m.get_audio_features(features, feature_mask)
        )
        input_ids = m._build_prompt(num_audio_tokens, language, system_prompt)
        inputs = m._build_inputs_embeds(input_ids, audio_features)[0]
        prompt = input_ids[0]
        if prompt_length is not None:
            prompt_length.append(int(prompt.shape[0]))
        if max_tokens <= 0:
            return

        cache = m.make_cache()
        # Do not project the audio-prefix positions through the enormous
        # vocabulary head: only their hidden states/KV entries are needed.
        # Chunk exactly like ``mlx_lm.generate_step`` so a very long single
        # chunk sees the same Metal shapes as the native path.
        remaining = inputs.shape[0] - 1
        consumed = 0
        while remaining > 0:
            n = min(prefill_step_size, remaining)
            m.model(inputs_embeds=inputs[consumed : consumed + n][None], cache=cache)
            mx.eval([entry.state for entry in cache])
            consumed += n
            remaining -= n
            mx.clear_cache()
        hidden = m.model(inputs_embeds=inputs[consumed:][None], cache=cache)
        hidden = hidden[:, -1:, :]
        # Extra processors (hotword bias) need the logits to exist: the tiered
        # argmax head returns a token without ever materialising them.  The
        # repetition penalty no longer switches the head off (TIERED_UNDER_RP):
        # it is carried inside both tiers by ``TieredVocabularyHead.argmax``.
        tiered = (
            self.tiered_head
            if not logits_processors
            and (TIERED_UNDER_RP or repetition_penalty in (None, 0))
            else None
        )
        logits = None
        if tiered is None:
            logits = (
                m.lm_head(hidden)[:, -1, :]
                if m.lm_head is not None
                else m.model.embed_tokens.as_linear(hidden)[:, -1, :]
            )
        if self.compiled_decode:
            # Reuse the native prefill buffers at their existing 256-token
            # capacity. Unlike the failed max-budget experiment, this does not
            # force a short active prefix through hundreds of masked slots.
            keys = [entry.keys for entry in cache]
            values = [entry.values for entry in cache]
            mx.eval(keys, values)
            position_int = int(cache[0].offset)
            position = mx.array(position_int, dtype=mx.int32)

        processors = list(
            make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if repetition_penalty
            else []
        )
        # Order is load-bearing: the hotword boost is added AFTER the repetition
        # penalty and is never itself penalised (registered design).
        processors.extend(logits_processors or [])
        processors = processors or None
        token_history = prompt[-1:]
        if tiered is None:
            logits = self._apply_processors(token_history, logits, processors)
            token = mx.argmax(logits, axis=-1).astype(mx.int32)
        else:
            token = tiered.argmax(hidden, token_history, repetition_penalty,
                                  repetition_context_size)
        mx.async_eval(token)

        eos = m._eos_token_ids()
        for step in range(max_tokens):
            # Launch the next GPU step before synchronously reading the current
            # token.  This is the same overlap pattern used by mlx_lm's
            # generate_step; reading first serializes Python and Metal and is a
            # severe latency regression even when the math is identical.
            token_history = mx.concatenate((token_history, token.reshape((-1,))))
            if self.compiled_decode:
                if position_int >= keys[0].shape[2]:
                    # Mirror mlx_lm.KVCache exactly: grow all layers by one
                    # 256-token slab, then keep the next compiled steps fixed.
                    grown_keys = []
                    grown_values = []
                    for key, value in zip(keys, values):
                        grown_keys.append(
                            mx.concatenate(
                                (
                                    key,
                                    mx.zeros(
                                        (*key.shape[:2], 256, key.shape[3]),
                                        dtype=key.dtype,
                                    ),
                                ),
                                axis=2,
                            )
                        )
                        grown_values.append(
                            mx.concatenate(
                                (
                                    value,
                                    mx.zeros(
                                        (*value.shape[:2], 256, value.shape[3]),
                                        dtype=value.dtype,
                                    ),
                                ),
                                axis=2,
                            )
                        )
                    keys, values = grown_keys, grown_values
                    mx.eval(keys, values)
                active_keys = [key[..., :position_int, :] for key in keys]
                active_values = [value[..., :position_int, :] for value in values]
                logits, new_keys, new_values = self.decoder(
                    token, active_keys, active_values, position
                )
                for index, (new_key, new_value) in enumerate(
                    zip(new_keys, new_values)
                ):
                    keys[index][..., position_int : position_int + 1, :] = new_key
                    values[index][..., position_int : position_int + 1, :] = new_value
                position_int += 1
                position = position + 1
            elif tiered is not None:
                hidden = m.model(
                    inputs_embeds=m.model.embed_tokens(token.reshape((1, 1))),
                    cache=cache,
                )
                next_token = tiered.argmax(hidden[:, -1:, :], token_history,
                                           repetition_penalty, repetition_context_size)
                mx.async_eval(next_token)
                token_id = int(token.item())
                if token_id in eos:
                    break
                yield token_id
                token = next_token
                if step and step % 256 == 0:
                    mx.clear_cache()
                continue
            else:
                logits = m._forward_with_embeds(
                    m.model.embed_tokens(token.reshape((1, 1))), cache
                )[:, -1, :]
            logits = self._apply_processors(token_history, logits, processors)
            next_token = mx.argmax(logits, axis=-1).astype(mx.int32)
            mx.async_eval(next_token)

            token_id = int(token.item())
            if token_id in eos:
                break
            yield token_id
            token = next_token
            if step and step % 256 == 0:
                mx.clear_cache()

    def hotword_bias(self, hotwords, strength: float = HOTWORD_STRENGTH_DEFAULT,
                     policy: str = HOTWORD_POLICY_DEFAULT):
        """The processor for this hotword list, or None for an empty list.

        Built once per distinct (list, strength) and kept in a small cache:
        Detta's partials re-decode the same list every few hundred ms.
        """
        words = HotwordBias._clean(hotwords)
        if not words or not strength:
            return None
        key = (tuple(words), float(strength), policy)
        bias = self._hotword_cache.get(key)
        if bias is None:
            if len(self._hotword_cache) >= 4:
                self._hotword_cache.pop(next(iter(self._hotword_cache)))
            bias = HotwordBias(self.model._tokenizer, words, strength=strength, policy=policy)
            self._hotword_cache[key] = bias
        return bias

    def _prepare(self, audio, chunk_duration: float, min_chunk_duration: float):
        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio))
        elif isinstance(audio, mx.array):
            audio = np.asarray(audio)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return split_audio_into_chunks(
            audio,
            sr=16_000,
            chunk_duration=chunk_duration,
            min_chunk_duration=min_chunk_duration,
        )

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 8192,
        language: str | None = None,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 100,
        chunk_duration: float = 1200.0,
        min_chunk_duration: float = 1.0,
        prefill_step_size: int = 2048,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        hotwords=None,
        hotword_strength: float = HOTWORD_STRENGTH_DEFAULT,
        hotword_policy: str = HOTWORD_POLICY_DEFAULT,
        logits_processors: list | None = None,
        **kwargs,
    ):
        if temperature != 0.0 or kwargs.get("sampler") is not None:
            # Prompt-head elision is a greedy-decoding optimisation.  Anything
            # that needs a real sampler falls back to the untouched native
            # implementation rather than losing a capability.
            return self.model.generate(
                audio,
                max_tokens=max_tokens,
                language=language,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                chunk_duration=chunk_duration,
                min_chunk_duration=min_chunk_duration,
                prefill_step_size=prefill_step_size,
                system_prompt=system_prompt,
                temperature=temperature,
                stream=stream,
                **kwargs,
            )

        # Hotwords -> one extra logits processor.  An empty or absent list builds
        # nothing, so the default decode is the unchanged path, byte for byte.
        processors = list(logits_processors or [])
        bias = self.hotword_bias(hotwords, hotword_strength, hotword_policy)
        if bias is not None:
            processors.append(bias)
        processors = processors or None

        chunks = self._prepare(audio, chunk_duration, min_chunk_duration)
        if stream:
            return self._stream(
                chunks,
                max_tokens=max_tokens,
                language=language,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                system_prompt=system_prompt,
                prefill_step_size=prefill_step_size,
                logits_processors=processors,
            )

        started = time.perf_counter()
        texts: list[str] = []
        segments: list[dict] = []
        prompt_tokens = 0
        generation_tokens = 0
        for chunk, offset in chunks:
            if max_tokens - generation_tokens <= 0:
                break
            prompt_length: list[int] = []
            tokens = list(
                self._iter_tokens(
                    chunk,
                    max_tokens=max_tokens - generation_tokens,
                    language=language,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                    system_prompt=system_prompt,
                    prefill_step_size=prefill_step_size,
                    prompt_length=prompt_length,
                    logits_processors=processors,
                )
            )
            text = self.model._tokenizer.decode(tokens, skip_special_tokens=True)
            chunk_language = language
            if chunk_language is None:
                chunk_language, text = self.model.extract_language(text)
            texts.append(text)
            prompt_tokens += prompt_length[0] if prompt_length else 0
            generation_tokens += len(tokens)
            segments.append(
                {
                    "text": text,
                    "language": chunk_language,
                    "start": offset,
                    "end": offset + len(chunk) / 16_000,
                }
            )
            mx.clear_cache()
        elapsed = time.perf_counter() - started
        return STTOutput(
            text=" ".join(texts),
            segments=segments,
            language=[row["language"] for row in segments],
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=prompt_tokens + generation_tokens,
            total_time=elapsed,
            prompt_tps=prompt_tokens / elapsed if elapsed else 0,
            generation_tps=generation_tokens / elapsed if elapsed else 0,
        )

    def _stream(
        self,
        chunks,
        *,
        max_tokens: int,
        language: str | None,
        repetition_penalty: float | None,
        repetition_context_size: int,
        system_prompt: str | None,
        prefill_step_size: int,
        logits_processors: list | None = None,
    ):
        """Emit one ``StreamingResult`` per token, matching the native
        ``stream_transcribe`` contract.  The token sequence comes from the same
        ``_iter_tokens`` generator the batch path uses, so a streamed decode
        cannot diverge from the batch decode."""
        tokenizer = self.model._tokenizer
        total_prompt_tokens = 0
        total_generation_tokens = 0
        remaining_tokens = max_tokens
        language_accumulator = ""

        for index, (chunk, offset) in enumerate(chunks):
            is_last_chunk = index == len(chunks) - 1
            duration = len(chunk) / 16_000
            token_count = 0
            prompt_length: list[int] = []
            for i, token in enumerate(
                self._iter_tokens(
                    chunk,
                    max_tokens=remaining_tokens,
                    language=language,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                    system_prompt=system_prompt,
                    prefill_step_size=prefill_step_size,
                    prompt_length=prompt_length,
                    logits_processors=logits_processors,
                )
            ):
                if not total_prompt_tokens and prompt_length:
                    total_prompt_tokens += prompt_length[0]
                text = tokenizer.decode([token])
                if language is None and i <= 2:
                    language_accumulator += text
                    if "<asr_text>" in language_accumulator:
                        language, _ = self.model.extract_language(language_accumulator)
                    continue
                previous = token_count / max(remaining_tokens, 1)
                token_count += 1
                current = min(token_count / max(remaining_tokens, 1), 1.0)
                yield StreamingResult(
                    text=text,
                    is_final=False,
                    start_time=offset + duration * previous,
                    end_time=offset + duration * current,
                    language=language or "English",
                )
            if prompt_length and not total_prompt_tokens:
                total_prompt_tokens += prompt_length[0]
            total_generation_tokens += token_count
            remaining_tokens -= token_count
            yield StreamingResult(
                text="",
                is_final=is_last_chunk or remaining_tokens <= 0,
                start_time=offset,
                end_time=offset + duration,
                language=language or "English",
                prompt_tokens=total_prompt_tokens,
                generation_tokens=total_generation_tokens,
            )
            mx.clear_cache()
            if remaining_tokens <= 0:
                break


def load_optimized_v18(
    path: Path | None = None,
    *,
    model=None,
    fused_qmv: bool = False,
    audio_blocks: bool = False,
    compiled_decode: bool = False,
    fold_bits: int | None = None,
    tiered_head_bits: int | None = None,
    tiered_head_k: int = 32,
    audio_tower_bits: int | None = None,
    audio_tower_group_size: int = 128,
) -> OptimizedV18:
    return OptimizedV18(
        path,
        model=model,
        fused_qmv=fused_qmv,
        audio_blocks=audio_blocks,
        compiled_decode=compiled_decode,
        fold_bits=fold_bits,
        tiered_head_bits=tiered_head_bits,
        tiered_head_k=tiered_head_k,
        audio_tower_bits=audio_tower_bits,
        audio_tower_group_size=audio_tower_group_size,
    )

"""Phonon-2 runtime speed levers (2026-09-26), applied AFTER load_parakeet() so the plane cache is untouched.

  install_fast_tdt(model, block=16)
      ParakeetTDT.decode with the greedy TDT loop kept on the GPU: the SAME compiled `_tdt_step` graph
      (same shapes, same ops -> same logits, same argmax) is chained `block` steps at a time with the
      control flow (blank / duration / max_symbols) expressed as mx.where, and ONE host sync per block
      instead of one per step.  Steps past the end of the utterance are computed but masked out.
      Tokens, start frames and durations come back per block and are turned into AlignedTokens with
      the exact formula mlx-audio uses.
      set_hotwords(model, automaton) arms the loop with a hotword automaton (fermion/_speech/hotwords.py): a
      biased twin of the step adds lam per piece that advances a listed word to the token logits (bonus only,
      blank never biased, same per-frame symbol cap); with no automaton the original compiled step runs, so a
      decode without hotwords is the same computation as before.

  densify_encoder(model, dtype=mx.float32)
      Replace every PackedFiveValueLinear (two 2-bit QMM planes, fp32 arithmetic) with a dense
      nn.Linear / nn.Conv1d whose weight is the module's own exact dequantisation
      ({0, +-lo, +-hi}; fp32 and fp16 hold these values exactly, bf16 does NOT).  fp32 keeps the shipped
      arithmetic class (fp32 inputs, fp32 accumulate, one rounding to the activation dtype); fp16 adds a
      rounding of the GEMM output to fp16 before the cast back to bf16.  Costs 4 (fp32) / 2 (fp16) bytes
      per weight of memory for the 604M encoder weights.

  install_fast_rel_shift(model)
      RelPositionMultiHeadAttention.rel_shift via one gather instead of pad+reshape+slice+reshape copies
      (a pure re-indexing of the same matmul outputs -> identical values).

Every lever is gated on transcripts: 20 holds identical, G-DECODE 40/40 vs dense fp32, 400-utterance
ruler unchanged, 660-clip gate unchanged (lane_phonon2_runtime_speed_20260926/).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.parakeet import parakeet as _pk
from mlx_audio.stt.models.parakeet import tokenizer as _tok
from mlx_audio.stt.models.parakeet.attention import RelPositionMultiHeadAttention


# ----------------------------------------------------------------------------- TDT loop
def _make_fast_step(self):
    """One greedy TDT step + its control flow as ONE compiled graph.  Inputs are fixed-shape ((1,1,D) feature,
    scalars, LSTM state) so mx.compile traces once per model.  The joint/LSTM math is `_tdt_step` verbatim."""
    blank = self.blank_id
    max_symbols = self.max_symbols
    durations = mx.array(self.durations, dtype=mx.int32)
    tdt_step = self._tdt_step

    def fast_step(feature, last, h, c, t, nsym, max_len):
        valid = t < max_len
        pred, dec, h2, c2 = tdt_step(feature, last.reshape(1, 1), h, c)
        dur = mx.take(durations, dec)
        emit = mx.logical_and(valid, pred != blank)
        tok_out = mx.where(emit, pred, -1)
        last = mx.where(emit, pred, last)
        h = mx.where(emit, h2, h)
        c = mx.where(emit, c2, c)
        nsym = nsym + 1
        t_adv = t + dur
        if max_symbols is not None:
            hit = mx.logical_and(dur == 0, nsym >= max_symbols)
            t_adv = mx.where(hit, t_adv + 1, t_adv)
            nsym = mx.where(mx.logical_or(dur != 0, hit), 0, nsym)
        else:
            nsym = mx.where(dur != 0, 0, nsym)
        t_new = mx.where(valid, t_adv, t)
        idx_next = mx.minimum(t_new, max_len - 1)
        return tok_out, t, dur, t_new, last, h, c, nsym, idx_next

    return mx.compile(fast_step)


def _make_fast_step_biased(self):
    """_make_fast_step + the hotword automaton.  The decoder / joint ops are `_tdt_step`'s own, written out so the
    token logits are visible: the automaton row for the current state becomes a (V,) bonus (lam per piece that
    advances a listed word, 0 otherwise, blank 0) added before the token argmax; the duration argmax reads the raw
    logits; the automaton state is one more scalar carried through the block.  Used ONLY while hotwords are armed."""
    blank = self.blank_id
    max_symbols = self.max_symbols
    durations = mx.array(self.durations, dtype=mx.int32)
    embed = self.decoder.prediction["embed"]
    rnn = self.decoder.prediction["dec_rnn"]
    joint = self.joint

    def fast_step(feature, last, h, c, t, nsym, max_len, state, table, toks, col, depth, lam):
        valid = t < max_len
        cur = last.reshape(1, 1)
        x = embed(cur)
        x = mx.where(mx.expand_dims(cur == blank, -1), mx.zeros_like(x), x)
        dec, (h2, c2) = rnn(x, (h, c))
        dec = dec.astype(feature.dtype); h2 = h2.astype(feature.dtype); c2 = c2.astype(feature.dtype)
        logits = joint(feature, dec)[0, 0, 0]
        row = mx.take(table, state, axis=0)                                  # (K,) next state per automaton piece
        gain = (mx.take(depth, row) - mx.take(depth, state)).astype(mx.float32)
        bias = mx.zeros((blank + 1,), dtype=mx.float32).at[toks].add(lam * mx.maximum(gain, 0.0))   # blank is never a piece
        pred = mx.argmax(logits[: blank + 1].astype(mx.float32) + bias).astype(mx.int32)
        decn = mx.argmax(logits[blank + 1:]).astype(mx.int32)
        dur = mx.take(durations, decn)
        emit = mx.logical_and(valid, pred != blank)
        tok_out = mx.where(emit, pred, -1)
        last = mx.where(emit, pred, last)
        h = mx.where(emit, h2, h)
        c = mx.where(emit, c2, c)
        k = mx.take(col, pred)                                               # column of the emitted piece, -1 = not a piece
        nxt = mx.where(k >= 0, mx.take(row, mx.maximum(k, 0)), mx.array(0, dtype=mx.int32))
        state = mx.where(emit, nxt, state)                                   # a blank never moves the automaton
        nsym = nsym + 1
        t_adv = t + dur
        if max_symbols is not None:                                          # the per-frame symbol cap
            hit = mx.logical_and(dur == 0, nsym >= max_symbols)
            t_adv = mx.where(hit, t_adv + 1, t_adv)
            nsym = mx.where(mx.logical_or(dur != 0, hit), 0, nsym)
        else:
            nsym = mx.where(dur != 0, 0, nsym)
        t_new = mx.where(valid, t_adv, t)
        idx_next = mx.minimum(t_new, max_len - 1)
        return tok_out, t, dur, t_new, last, h, c, nsym, idx_next, state

    return mx.compile(fast_step)


def _make_fast_step_biased_dense(self):
    """The step the engine runs while hotwords are armed: the same math as _make_fast_step_biased with the automaton
    expanded per state over the whole vocabulary -- B[s] is the (V,) bonus row of state s (lam * max(0, depth gain) on
    the automaton's pieces, 0 elsewhere and on blank) -- so the bonus is one row gather and an add instead of a
    gather / scatter chain; the next state is T[state * K + col[piece]] (col -1 = not a piece -> root).  Same bonus
    values, same argmax, same state walk (checked against the sparse step in ci/test_hotwords_029.py)."""
    blank = self.blank_id
    max_symbols = self.max_symbols
    durations = mx.array(self.durations, dtype=mx.int32)
    embed = self.decoder.prediction["embed"]
    rnn = self.decoder.prediction["dec_rnn"]
    joint = self.joint
    width = blank + 1

    def fast_step(feature, last, h, c, t, nsym, max_len, state, B, T, col):
        valid = t < max_len
        cur = last.reshape(1, 1)
        x = embed(cur)
        x = mx.where(mx.expand_dims(cur == blank, -1), mx.zeros_like(x), x)
        dec, (h2, c2) = rnn(x, (h, c))
        dec = dec.astype(feature.dtype); h2 = h2.astype(feature.dtype); c2 = c2.astype(feature.dtype)
        logits = joint(feature, dec)[0, 0, 0]
        bias = mx.take(B, state, axis=0).astype(mx.float32)                 # (V,) this state's bonus row; blank 0
        pred = mx.argmax(logits[:width].astype(mx.float32) + bias).astype(mx.int32)
        decn = mx.argmax(logits[width:]).astype(mx.int32)
        dur = mx.take(durations, decn)
        emit = mx.logical_and(valid, pred != blank)
        tok_out = mx.where(emit, pred, -1)
        last = mx.where(emit, pred, last)
        h = mx.where(emit, h2, h)
        c = mx.where(emit, c2, c)
        k = mx.take(col, pred)                                               # column of the emitted piece, -1 = not a piece
        nxt = mx.where(k >= 0, mx.take(T, state * (T.shape[0] // B.shape[0]) + mx.maximum(k, 0)), mx.array(0, dtype=mx.int32))
        state = mx.where(emit, nxt, state)                                   # a blank never moves the automaton
        nsym = nsym + 1
        t_adv = t + dur
        if max_symbols is not None:                                          # the per-frame symbol cap
            hit = mx.logical_and(dur == 0, nsym >= max_symbols)
            t_adv = mx.where(hit, t_adv + 1, t_adv)
            nsym = mx.where(mx.logical_or(dur != 0, hit), 0, nsym)
        else:
            nsym = mx.where(dur != 0, 0, nsym)
        t_new = mx.where(valid, t_adv, t)
        idx_next = mx.minimum(t_new, max_len - 1)
        return tok_out, t, dur, t_new, last, h, c, nsym, idx_next, state

    return mx.compile(fast_step)


def dense_tables(automaton, width: int):
    """numpy (B [S, width] float16, T [S * K] int32, col [width] int32) for _make_fast_step_biased_dense.  The bonus
    values are small multiples of lam, exact in fp16 for usual lam values (else B stays fp32); the step adds in fp32."""
    import numpy as np
    S, K = automaton.table.shape
    gain = automaton.depth[automaton.table] - automaton.depth[:, None]                 # (S, K)
    B = np.zeros((S, width), dtype=np.float32)
    B[:, automaton.toks] = np.float32(automaton.lam) * np.maximum(gain, 0)
    B16 = B.astype(np.float16)
    if np.array_equal(B16.astype(np.float32), B):
        B = B16
    col = np.full((width,), -1, dtype=np.int32); n = min(width, automaton.col.shape[0]); col[:n] = automaton.col[:n]
    return B, np.ascontiguousarray(automaton.table.reshape(-1), dtype=np.int32), col


#: Device tables (the dense bonus rows are S x V, ~13 MB for a 25-word list) are kept for recent lists up to this
#: many bytes, so a server alternating a handful of lists does not rebuild them per request.
_DEVICE_TABLES_BYTES = 320 << 20
_device_tables: list = []


def set_hotwords(model, automaton) -> dict | None:
    """Arm the next decode(s) with a hotwords.HotwordAutomaton; None disarms (the unbiased step runs).
    Returns the automaton's description."""
    from fermion._speech.mlx_memory import release_automaton_mlx_tables

    if automaton is None:
        model._hotword_bias = None
        while _device_tables:
            release_automaton_mlx_tables(_device_tables.pop(0))
        return None
    width = model.blank_id + 1
    cached = getattr(automaton, "_mlx_tables", None)
    if cached is None or cached[0].shape[1] != width:
        cached = tuple(mx.array(x) for x in dense_tables(automaton, width))
        automaton._mlx_tables = cached
    if automaton in _device_tables:
        _device_tables.remove(automaton)
    _device_tables.append(automaton)                                    # most recent last
    while len(_device_tables) > 1 and sum(a._mlx_tables[0].nbytes for a in _device_tables) > _DEVICE_TABLES_BYTES:
        release_automaton_mlx_tables(_device_tables.pop(0))
    model._hotword_bias = cached
    if getattr(model, "_fast_step_biased", None) is None:
        model._fast_step_biased = _make_fast_step_biased_dense(model)
    if getattr(model, "_fast_step", None) is None:
        model._fast_step = _make_fast_step(model); model._fast_step_kind = "plain"
    return automaton.describe()


def biased_decode(model):
    """The decode to run while hotwords are armed: `_decode_fast` (which takes the biased step), whatever lever
    is installed as `model.decode` (none, `tdt`, or another loop)."""
    return _decode_fast.__get__(model, type(model))


def _make_fast_step_lean(self):
    """Same as _make_fast_step, with (a) the LSTM step written out as nn.LSTM's own op sequence (addmm x, addmm h, split,
    sigmoid/tanh, f*c+i*g, o*tanh(c)) on 2-D state -- no wrapper transposes / stacks / astype no-ops -- and (b) a blank with
    duration 0 advancing one frame at once (mlx-audio replays that identical state up to max_symbols times before advancing:
    same tokens, same frame, fewer steps).  Gated on token identity like every lever."""
    blank = self.blank_id
    max_symbols = self.max_symbols
    durations = mx.array(self.durations, dtype=mx.int32)
    embed = self.decoder.prediction["embed"]
    layers = self.decoder.prediction["dec_rnn"].lstm
    joint = self.joint
    nl = len(layers)

    def lstm_cell(layer, x, h, c):
        ifgo = mx.addmm(layer.bias, x, layer.Wx.T) if layer.bias is not None else x @ layer.Wx.T
        ifgo = mx.addmm(ifgo, h, layer.Wh.T)
        i, f, g, o = mx.split(ifgo, 4, axis=-1)
        i = mx.sigmoid(i); f = mx.sigmoid(f); g = mx.tanh(g); o = mx.sigmoid(o)
        c = f * c + i * g
        h = o * mx.tanh(c)
        return h, c

    def fast_step(feature, last, h, c, t, nsym, max_len):
        valid = t < max_len
        x = embed(last.reshape(1, 1))                                        # (1, 1, E)
        x = mx.where(mx.expand_dims(last.reshape(1, 1) == blank, -1), mx.zeros_like(x), x)[0]   # (1, E)
        hs, cs = [], []
        for li in range(nl):
            x, cl = lstm_cell(layers[li], x, h[li], c[li]); hs.append(x); cs.append(cl)
        h2 = mx.stack(hs, 0).astype(feature.dtype); c2 = mx.stack(cs, 0).astype(feature.dtype)
        dec = x.astype(feature.dtype)[None]                                  # (1, 1, H)
        logits = joint(feature, dec)[0, 0, 0]
        pred = mx.argmax(logits[: blank + 1]).astype(mx.int32)
        decn = mx.argmax(logits[blank + 1:]).astype(mx.int32)
        dur = mx.take(durations, decn)
        is_blank = pred == blank
        emit = mx.logical_and(valid, mx.logical_not(is_blank))
        tok_out = mx.where(emit, pred, -1)
        last = mx.where(emit, pred, last)
        h = mx.where(emit, h2, h)
        c = mx.where(emit, c2, c)
        # blank with duration 0: mlx-audio repeats the identical step until max_symbols then advances 1 -> do it now
        blank0 = mx.logical_and(is_blank, dur == 0)
        nsym = nsym + 1
        t_adv = t + dur
        if max_symbols is not None:
            hit = mx.logical_or(blank0, mx.logical_and(dur == 0, nsym >= max_symbols))
        else:
            hit = blank0
        t_adv = mx.where(hit, t_adv + 1, t_adv)
        nsym = mx.where(mx.logical_or(dur != 0, hit), 0, nsym)
        t_new = mx.where(valid, t_adv, t)
        idx_next = mx.minimum(t_new, max_len - 1)
        return tok_out, t, dur, t_new, last, h, c, nsym, idx_next

    return mx.compile(fast_step)


def _decode_fast(self, mel: mx.array):
    block = getattr(self, "_fast_tdt_block", 16)
    batch_size = mel.shape[0]
    if mel.ndim == 2:
        batch_size = 1
        mel = mx.expand_dims(mel, 0)
    batch_features, lengths = self.encoder(mel)
    mx.eval(batch_features, lengths)

    blank = self.blank_id
    sub = self.encoder_config.subsampling_factor
    sr = self.preprocessor_config.sample_rate
    hop = self.preprocessor_config.hop_length
    step = self._fast_step
    hotwords = getattr(self, "_hotword_bias", None)          # (B, NS) dense automaton tables or None: see set_hotwords
    step_b = getattr(self, "_fast_step_biased", None) if hotwords is not None else None

    results = []
    for b in range(batch_size):
        features = batch_features[b:b + 1]
        max_length = int(lengths[b])
        max_len = mx.array(max_length, dtype=mx.int32)
        h, c = self._make_initial_decoder_state(1, features.dtype)
        t = mx.array(0, dtype=mx.int32)
        last = mx.array(blank, dtype=mx.int32)
        nsym = mx.array(0, dtype=mx.int32)
        idx = mx.array([0], dtype=mx.int32)
        state = mx.array(0, dtype=mx.int32)
        t_host = 0
        hyp = []
        while t_host < max_length:
            toks, ts, durs = [], [], []
            for _ in range(block):
                feature = mx.take(features, idx, axis=1)                    # (1,1,D) == features[:, t:t+1]
                if step_b is None:
                    tok_out, t_at, dur, t, last, h, c, nsym, idx_next = step(feature, last, h, c, t, nsym, max_len)
                else:
                    tok_out, t_at, dur, t, last, h, c, nsym, idx_next, state = step_b(
                        feature, last, h, c, t, nsym, max_len, state, *hotwords)
                idx = idx_next.reshape(1)
                toks.append(tok_out); ts.append(t_at); durs.append(dur)
            vals = mx.stack(toks + ts + durs + [t]).tolist()                # the ONE host sync per block
            t_host = vals[-1]
            for k in range(block):
                tok = vals[k]
                if tok < 0 or _tok.is_special_token(tok, self.vocabulary):
                    continue
                # same float expression order as mlx-audio's decode(): time * sub / sr * hop
                hyp.append(_pk.AlignedToken(tok, start=vals[block + k] * sub / sr * hop,
                                            duration=vals[2 * block + k] * sub / sr * hop,
                                            text=_tok.decode([tok], self.vocabulary)))
        results.append(_pk.sentences_to_result(_pk.tokens_to_sentences(hyp)))
    return results


def install_fast_tdt(model, block: int = 16, lean: bool = False):
    model._fast_tdt_block = int(block)
    want = "lean" if lean else "plain"
    if getattr(model, "_fast_step_kind", None) != want:
        model._fast_step = _make_fast_step_lean(model) if lean else _make_fast_step(model)
        model._fast_step_kind = want
    if not hasattr(model, "_orig_decode"):
        model._orig_decode = model.decode
    model.decode = _decode_fast.__get__(model, type(model))
    return model


def uninstall_fast_tdt(model):
    if hasattr(model, "_orig_decode"):
        model.decode = model._orig_decode


# ----------------------------------------------------------------------------- dense encoder
_PACKED_LEAVES = ("feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2",
                  "self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v", "self_attn.linear_out",
                  "self_attn.linear_pos", "conv.pointwise_conv1", "conv.pointwise_conv2")


class DenseFromPacked(nn.Module):
    """Dense stand-in for PackedFiveValueLinear: x -> (x @ W.T + b) in W.dtype, cast back to x.dtype."""

    def __init__(self, w: mx.array, bias):
        super().__init__()
        self.weight = w
        if bias is not None:
            self.bias = bias.astype(w.dtype)

    def __call__(self, x):
        in_dtype = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.astype(self.weight.dtype)
        y = x @ self.weight.T
        if "bias" in self:
            y = y + self.bias
        return y.astype(in_dtype)


def densify_encoder(model, dtype: mx.Dtype = mx.float32) -> dict:
    """PackedFiveValueLinear -> DenseFromPacked in `dtype` for the 264 encoder modules.  Receipt: count, bytes,
    max |dense - fp32 dequant| (0.0 for fp32 and fp16 by construction; bf16 would not be exact)."""
    from packed_runtime import PackedFiveValueLinear
    n = 0; total_bytes = 0; worst = 0.0
    for layer in model.encoder.layers:
        for dotted in _PACKED_LEAVES:
            parent_name, leaf = dotted.split(".")
            parent = getattr(layer, parent_name)
            mod = getattr(parent, leaf)
            if not isinstance(mod, PackedFiveValueLinear):
                continue
            w32 = mod.dequantize()
            w = w32.astype(dtype)
            worst = max(worst, float(mx.abs(w.astype(mx.float32) - w32).max()))
            bias = mod.bias if "bias" in mod else None
            new = DenseFromPacked(w, bias)
            mx.eval(new.parameters())
            setattr(parent, leaf, new)
            total_bytes += w.nbytes; n += 1
    mx.eval(model.parameters())
    return {"densified_modules": n, "dense_bytes": total_bytes, "dtype": str(dtype),
            "max_abs_delta_vs_fp32_dequant": worst}


# ----------------------------------------------------------------------------- rel_shift
def _rel_shift_gather(self, x: mx.array) -> mx.array:
    """x: (B, H, Tq, P) with P = 2*Tq - 1 relative positions ordered (Tq-1 ... -(Tq-1)).
    mlx-audio's pad+reshape+slice produces out[b,h,i,j] = x[b,h,i, (P - Tq) + j - i] ... reproduced here
    by one gather along the last axis (identical elements, no arithmetic)."""
    B, H, Tq, P = x.shape
    if not hasattr(self, "_fast_shift_idx") or self._fast_shift_idx.shape[0] != Tq or self._fast_shift_P != P:
        i = mx.arange(Tq, dtype=mx.int32)[:, None]
        j = mx.arange(P, dtype=mx.int32)[None, :]
        # original: pad left by 1 -> (B,H,Tq,P+1) -> view as (B,H,P+1,Tq) -> drop first row -> view (B,H,Tq,P)
        # flat index arithmetic of that sequence, evaluated once per (Tq, P):
        flat = (i * P + j)                                   # position in the final (Tq, P) view
        src = flat + Tq                                      # + Tq: the dropped first row of the (P+1, Tq) view
        # src indexes the padded (Tq, P+1) matrix flattened; column 0 of each row is the pad (never selected
        # where the original slice keeps data within [:, :, :, :k_seq] -- kept here for the full P width)
        r = src // (P + 1); cc = src % (P + 1)
        idx = mx.where(cc == 0, -1, r * P + (cc - 1))       # -1 -> pad element (value 0 in the original)
        self._fast_shift_idx = idx; self._fast_shift_P = P
        mx.eval(idx)
    idx = self._fast_shift_idx
    flat = x.reshape(B, H, Tq * P)
    pad_col = mx.zeros((B, H, 1), dtype=x.dtype)
    flat = mx.concatenate([flat, pad_col], axis=-1)          # index -1 -> the zero pad
    out = mx.take(flat, idx.reshape(-1), axis=-1)
    return out.reshape(B, H, Tq, P)


def install_fast_rel_shift(model):
    for layer in model.encoder.layers:
        at = layer.self_attn
        if isinstance(at, RelPositionMultiHeadAttention):
            at.rel_shift = _rel_shift_gather.__get__(at, type(at))
    return model


# ============================================================================= 2026-09-28 levers (runtime lane #2)
class StackedFiveValueLinear(nn.Module):
    """Bit-exact CANDIDATE for PackedFiveValueLinear: the two 2-bit planes stacked along the output axis
    ([base_q; residual_q] -> 2*O rows) so ONE quantized_matmul dispatch computes both, then the halves are added
    (same per-row group accumulation, same add, one dispatch instead of two).  Identity is measured, never assumed."""

    def __init__(self, mod):
        super().__init__()
        self.out_features = mod.out_features; self.in_features = mod.in_features; self.scale_dtype = mod.scale_dtype
        self.q = mx.concatenate([mod.base_q, mod.residual_q], axis=0)
        self.scales = mx.concatenate([mod._base_scales, mod._resid_scales], axis=0)
        self.biases = mx.concatenate([mod._base_biases, mod._resid_biases], axis=0)
        if "bias" in mod:
            self.bias = mod.bias
        mx.eval(self.q, self.scales, self.biases)

    def __call__(self, x):
        in_dtype = x.dtype
        if self.scale_dtype != in_dtype:
            x = x.astype(self.scale_dtype)
        y = mx.quantized_matmul(x, self.q, self.scales, self.biases, transpose=True, group_size=128, bits=2, mode="affine")
        y = y[..., : self.out_features] + y[..., self.out_features:]
        if "bias" in self:
            y = y + self.bias
        return y.astype(in_dtype)


def install_stacked_planes(model) -> dict:
    from packed_runtime import PackedFiveValueLinear
    n = 0
    for layer in model.encoder.layers:
        for dotted in _PACKED_LEAVES:
            parent_name, leaf = dotted.split(".")
            parent = getattr(layer, parent_name); mod = getattr(parent, leaf)
            if isinstance(mod, PackedFiveValueLinear):
                setattr(parent, leaf, StackedFiveValueLinear(mod)); n += 1
    mx.eval(model.parameters())
    return {"stacked_modules": n}


def drop_packed_planes(model) -> dict:
    """After densify_encoder: nothing else references the packed modules, so this is a no-op placeholder that
    reports what densify left behind (the DenseFromPacked modules replaced the packed ones; the numpy/MLX plane
    arrays were released when the module was dropped).  Kept for the receipt."""
    return {"active_gb": mx.get_active_memory() / 1e9}


def install_joint_preproject(model) -> dict:
    """Bit-exact CANDIDATE: joint.enc (1024->640) is applied to EVERY encoder frame once per utterance as one GEMM
    instead of once per TDT step on a single frame (GEMV).  The step then adds the pre-projected frame.  Same weights,
    same fp32-accumulate GEMM class, but a different kernel (GEMM vs GEMV) -> identity measured."""
    from mlx_audio.stt.models.parakeet import parakeet as _pk
    blank = model.blank_id; max_symbols = model.max_symbols
    durations = mx.array(model.durations, dtype=mx.int32)
    joint = model.joint; decoder = model.decoder

    def step_pre(feature_p, last, h, c, t, nsym, max_len):
        valid = t < max_len
        dec, (h2, c2) = decoder(last.reshape(1, 1), (h, c))
        x = mx.expand_dims(feature_p, 2) + mx.expand_dims(joint.pred(dec), 1)
        for layer in joint.joint_net:
            x = layer(x)
        logits = x[0, 0, 0]
        pred = mx.argmax(logits[: blank + 1]).astype(mx.int32)
        decn = mx.argmax(logits[blank + 1:]).astype(mx.int32)
        dur = mx.take(durations, decn)
        emit = mx.logical_and(valid, pred != blank)
        tok_out = mx.where(emit, pred, -1)
        last = mx.where(emit, pred, last)
        h = mx.where(emit, h2, h); c = mx.where(emit, c2, c)
        nsym = nsym + 1
        t_adv = t + dur
        if max_symbols is not None:
            hit = mx.logical_and(dur == 0, nsym >= max_symbols)
            t_adv = mx.where(hit, t_adv + 1, t_adv)
            nsym = mx.where(mx.logical_or(dur != 0, hit), 0, nsym)
        else:
            nsym = mx.where(dur != 0, 0, nsym)
        t_new = mx.where(valid, t_adv, t)
        idx_next = mx.minimum(t_new, max_len - 1)
        return tok_out, t, dur, t_new, last, h, c, nsym, idx_next

    model._fast_step = mx.compile(step_pre); model._fast_step_kind = "jointpre"
    model._fast_preproject = True
    return {"jointpre": "installed"}


def _decode_fast_pre(self, mel):
    """_decode_fast with the encoder frames pre-projected through joint.enc once per utterance."""
    block = getattr(self, "_fast_tdt_block", 16)
    batch_size = mel.shape[0]
    if mel.ndim == 2:
        batch_size = 1; mel = mx.expand_dims(mel, 0)
    batch_features, lengths = self.encoder(mel)
    proj = self.joint.enc(batch_features)                     # ONE GEMM over all frames
    mx.eval(proj, lengths)
    blank = self.blank_id; sub = self.encoder_config.subsampling_factor
    sr = self.preprocessor_config.sample_rate; hop = self.preprocessor_config.hop_length
    step = self._fast_step; results = []
    for b in range(batch_size):
        features = proj[b:b + 1]; max_length = int(lengths[b]); max_len = mx.array(max_length, dtype=mx.int32)
        h, c = self._make_initial_decoder_state(1, batch_features.dtype)
        t = mx.array(0, dtype=mx.int32); last = mx.array(blank, dtype=mx.int32); nsym = mx.array(0, dtype=mx.int32)
        idx = mx.array([0], dtype=mx.int32); t_host = 0; hyp = []
        while t_host < max_length:
            toks, ts, durs = [], [], []
            for _ in range(block):
                feature = mx.take(features, idx, axis=1)
                tok_out, t_at, dur, t, last, h, c, nsym, idx_next = step(feature, last, h, c, t, nsym, max_len)
                idx = idx_next.reshape(1); toks.append(tok_out); ts.append(t_at); durs.append(dur)
            vals = mx.stack(toks + ts + durs + [t]).tolist()
            t_host = vals[-1]
            for k in range(block):
                tok = vals[k]
                if tok < 0 or _tok.is_special_token(tok, self.vocabulary):
                    continue
                hyp.append(_pk.AlignedToken(tok, start=vals[block + k] * sub / sr * hop, duration=vals[2 * block + k] * sub / sr * hop,
                                            text=_tok.decode([tok], self.vocabulary)))
        results.append(_pk.sentences_to_result(_pk.tokens_to_sentences(hyp)))
    return results


def install_fast_tdt_pre(model, block: int = 16):
    model._fast_tdt_block = int(block)
    install_joint_preproject(model)
    if not hasattr(model, "_orig_decode"):
        model._orig_decode = model.decode
    model.decode = _decode_fast_pre.__get__(model, type(model))
    return model


def install_compiled_layers(model) -> dict:
    """~EXACT candidate (bf16 rounding between fused element-wise ops changes): every ConformerBlock.__call__ wrapped in
    mx.compile so LayerNorm / SiLU / GLU / residual adds fuse into fewer Metal kernels.  Shapes vary per utterance
    (T), so mx.compile(shapeless=True) is used; identity vs the shipped path is measured on the ruler + holds."""
    n = 0
    for layer in model.encoder.layers:
        f = layer.__call__
        layer.__call__ = mx.compile(f, shapeless=True)
        n += 1
    return {"compiled_layers": n}

#!/usr/bin/env python3
"""Reference greedy decoder for the two tiny models, computed in float64
DIRECTLY from the artifact bytes (independent of the torch float32 dequant),
so the browser/Node JS engine can be verified BIT-EXACT on token ids.

Vendored from the gated reference decoder, minus its receipt-writing tooling
(torch cross-checks and main()), which depended on internal training modules
and was never used by the CLI.
"""
from __future__ import annotations

import math
import struct

import numpy as np

# 12 fixed prompts (both models), greedy / temp 0 — the gated battery.
PROMPTS = ["hi", "hello", "how are you", "what is your name", "what are you",
           "where do you live", "are you real", "do you like me",
           "tell me a story", "what do you like", "are you happy",
           "say something"]

M64 = (1 << 64) - 1


# --------------------------------------------------------------- primitives
def half_to_double(u16: int) -> float:
    """IEEE-754 binary16 (uint16 bit pattern) -> Python float (exact)."""
    s = (u16 >> 15) & 1
    e = (u16 >> 10) & 0x1F
    m = u16 & 0x3FF
    if e == 0:
        val = (m / 1024.0) * (2.0 ** -14)
    elif e == 0x1F:
        val = math.inf if m == 0 else math.nan
    else:
        val = (1.0 + m / 1024.0) * (2.0 ** (e - 15))
    return -val if s else val


def rd_f16(buf: bytes, off: int) -> float:
    return half_to_double(buf[off] | (buf[off + 1] << 8))


def erf_approx(x):
    """Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7). Used IDENTICALLY in
    the JS engine so gelu matches to ~1e-15, not just to the erf error."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def gelu(z):
    return 0.5 * z * (1.0 + erf_approx(z / math.sqrt(2.0)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def rmsnorm(x, g):
    # x [..., d]; g [d]
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ms + 1e-6)) * g


def argmax_first(v: np.ndarray) -> int:
    return int(np.argmax(v))  # numpy returns first max on ties


# ------------------------------------------------------- QRB8 int4 bank GRU
def splitmix64(state: int):
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z ^ (z >> 31)) & M64, state


def hash_consts(seed: int):
    if seed == 0:
        return (2654435761, 2246822519, 3266489917, 668265263)
    st = seed & M64
    out = []
    for _ in range(4):
        v, st = splitmix64(st)
        out.append((v & 0xFFFFFFFF) | 1)
    return tuple(out)


BITS_NAME = {0: "binary", 1: "tern", 2: "int4", 3: "int8", 4: "five"}
QRB8_HEAD = "<4sBBHHHBIB"


def unpack_qrb8(buf: bytes):
    (magic, ver, bid, vocab_n, dim, ctx, n_layers, seed, nb) = \
        struct.unpack_from(QRB8_HEAD, buf, 0)
    assert magic == b"QRB8" and ver == 8
    bits = BITS_NAME[bid]
    assert bits == "int4"
    off = struct.calcsize(QRB8_HEAD)
    bank_ks = list(struct.unpack_from(f"<{nb}H", buf, off))
    off += 2 * nb
    (blen,) = struct.unpack_from("<H", buf, off)
    off += 2
    words = buf[off:off + blen].decode("utf-8").split("\n") if blen else []
    off += blen
    FIXED = ["<unk>", "<eos>", "\n", "A:", "B:", ".", ",", "?", "!", "'"]
    vocab = FIXED + words
    assert len(vocab) == vocab_n, (len(vocab), vocab_n)

    consts = hash_consts(seed)
    c1, c2, c3, c4 = consts
    GROUP = 64
    # bank linears: (out, in, layer_idx). embed then wi0,wh0,wi1,wh1
    specs = [(vocab_n, dim, 0)]
    for li in range(n_layers):
        specs.append((3 * dim, dim, 1 + 2 * li))
        specs.append((3 * dim, dim, 2 + 2 * li))
    banks = []
    for k, (out_f, in_f, layer_idx) in zip(bank_ks, specs):
        rows = k // GROUP
        scale = np.array([rd_f16(buf, off + 2 * r) for r in range(rows)],
                         dtype=np.float64)
        off += 2 * rows
        nbu = (k + 1) // 2
        raw = np.frombuffer(buf, np.uint8, nbu, off).astype(np.int32)
        off += nbu
        lo = raw & 0xF
        hi = (raw >> 4) & 0xF
        codes = np.empty(2 * nbu, np.int32)
        codes[0::2] = lo - 8
        codes[1::2] = hi - 8
        codes = codes[:k].astype(np.float64)
        baked = (codes.reshape(rows, GROUP) * scale[:, None]).reshape(-1)
        li = layer_idx + 1
        i = np.arange(out_f, dtype=np.int64)[:, None]
        j = np.arange(in_f, dtype=np.int64)[None, :]
        idx = ((i * c1 + j) ^ (c2 * li)) % k
        sgn = (((i * c3 + j) ^ (c4 * li)) % 2) * 2 - 1
        W = baked[idx] * sgn.astype(np.float64)   # [out, in] float64
        banks.append(W)
    # RMSNorm lnf
    gmax = rd_f16(buf, off); off += 2
    gq = np.frombuffer(buf, np.int8, dim, off).astype(np.float64); off += dim
    g = gq * (gmax / 127.0 if gmax > 0 else 1.0)
    cfg = {"vocab": vocab_n, "dim": dim, "n_layers": n_layers}
    return cfg, vocab, banks, g, off


def qrb8_greedy(buf: bytes, prompt: str, maxlen=30):
    cfg, vocab, banks, g, _ = unpack_qrb8(buf)
    stoi = {w: i for i, w in enumerate(vocab)}
    h = cfg["dim"]
    We = banks[0]                 # [vocab, h]
    layers = []
    for li in range(cfg["n_layers"]):
        layers.append((banks[1 + 2 * li], banks[2 + 2 * li]))  # Wi, Wh
    eos, nl = stoi["<eos>"], stoi["\n"]
    ids = [stoi["A:"]] + [stoi.get(w, stoi["<unk>"]) for w in prompt.split()] \
        + [nl, stoi["B:"]]
    states = [np.zeros(h, np.float64) for _ in layers]

    def step(xi, hprev, Wh):
        hh = Wh @ hprev
        xr, xz, xn = xi[:h], xi[h:2 * h], xi[2 * h:]
        hr, hz, hn = hh[:h], hh[h:2 * h], hh[2 * h:]
        r = sigmoid(xr + hr)
        z = sigmoid(xz + hz)
        n = np.tanh(xn + r * hn)
        return (1 - z) * n + z * hprev

    def feed(tok):
        x = We[tok] * math.sqrt(h)
        for i, (Wi, Wh) in enumerate(layers):
            states[i] = step(Wi @ x, states[i], Wh)
            x = states[i]
        return x

    x = None
    for t in ids:
        x = feed(t)
    gen = []
    for _ in range(maxlen):
        logits = We @ (rmsnorm(x, g))
        nxt = argmax_first(logits)
        if nxt == eos:
            break
        gen.append(nxt)
        x = feed(nxt)
    text = " ".join(vocab[t] for t in gen if t not in (eos, nl))
    return gen, text, vocab


# ------------------------------------------------- QRL8 looped transformer
QRL8_HEAD = "<4sBBBHHHBBBBBe"


def unpack_qrl8(buf: bytes):
    (magic, ver, arch, quant, vocab, dim, ctx, n_layers, n_heads, mlp,
     loop, loop_mode, thr) = struct.unpack_from(QRL8_HEAD, buf, 0)
    assert magic == b"QRL8" and ver == 8
    off = struct.calcsize(QRL8_HEAD)
    voc = []
    for _ in range(vocab):
        ln = buf[off]; off += 1
        voc.append(buf[off:off + ln].decode("utf-8")); off += ln
    # rec order: lin (embed, per-block wq wk wv wo fc1 fc2), then norms
    lin_specs = [("embed", vocab, dim)]
    for b in range(n_layers):
        for nm, R, C in [("wq", dim, dim), ("wk", dim, dim), ("wv", dim, dim),
                         ("wo", dim, dim), ("fc1", mlp * dim, dim),
                         ("fc2", dim, mlp * dim)]:
            lin_specs.append((f"b{b}.{nm}", R, C))
    norm_specs = []
    for b in range(n_layers):
        norm_specs += [(f"b{b}.ln1", dim), (f"b{b}.ln2", dim)]
    norm_specs += [("lnf", dim)]

    lin = []
    for name, R, C in lin_specs:
        amax = rd_f16(buf, off); off += 2
        s = rd_f16(buf, off); off += 2                 # five -> residual scale
        alpha_q = np.frombuffer(buf, np.int8, R, off).astype(np.float64)
        off += R
        alpha = alpha_q * (amax / 127.0 if amax > 0 else 1.0)
        lin.append({"name": name, "R": R, "C": C, "alpha": alpha, "s": s})
    norms = []
    for name, n in norm_specs:
        gmax = rd_f16(buf, off); off += 2
        gq = np.frombuffer(buf, np.int8, n, off).astype(np.float64); off += n
        norms.append({"name": name, "g": gq * (gmax / 127.0 if gmax > 0 else 1.0)})
    planes = buf[off:]

    n = sum(r["R"] * r["C"] for r in lin)
    n_mb = (n + 7) // 8
    mask_bytes = np.frombuffer(planes, np.uint8, n_mb, 0)
    nz = np.unpackbits(mask_bytes)[:n].astype(bool)
    k = int(nz.sum())
    n_cb = (k + 3) // 4
    cb = np.frombuffer(planes, np.uint8, n_cb, n_mb).astype(np.int32)
    codes = np.empty(4 * n_cb, np.int32)
    codes[0::4] = cb & 3
    codes[1::4] = (cb >> 2) & 3
    codes[2::4] = (cb >> 4) & 3
    codes[3::4] = (cb >> 6) & 3
    codes = codes[:k]
    trit = np.zeros(n, np.float64)
    dbit = np.zeros(n, np.float64)
    trit[nz] = np.where((codes >> 1) == 1, -1.0, 1.0)
    dbit[nz] = (codes & 1).astype(np.float64)

    weights = {}
    pos = 0
    for r in lin:
        R, C = r["R"], r["C"]
        tr = trit[pos:pos + R * C].reshape(R, C)
        db = dbit[pos:pos + R * C].reshape(R, C)
        pos += R * C
        dpm = (db * 2 - 1) * (tr != 0)
        W = tr * r["alpha"][:, None] + dpm * r["s"]
        weights[r["name"]] = W
    normg = {r["name"]: r["g"] for r in norms}
    cfg = {"vocab": vocab, "dim": dim, "ctx": ctx, "n_layers": n_layers,
           "n_heads": n_heads, "mlp": mlp, "loop": loop, "loop_mode": loop_mode}
    return cfg, voc, weights, normg


def build_schedule(n_layers, loop, loop_mode):
    if loop_mode == 0 or loop <= 1:      # none
        return list(range(n_layers))
    if loop_mode == 2:                   # all
        return list(range(n_layers)) * loop
    if n_layers < 3:                     # middle
        return list(range(n_layers))
    return [0] + list(range(1, n_layers - 1)) * loop + [n_layers - 1]


def sin_pos(ctx, d):
    pos = np.arange(ctx, dtype=np.float64)[:, None]
    i = np.arange(d, dtype=np.float64)[None, :]
    ang = pos / np.power(10000.0, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(ang), np.cos(ang))


def qrl8_forward_last(seq, cfg, W, G, POS):
    d = cfg["dim"]; nh = cfg["n_heads"]; hd = d // nh
    T = len(seq)
    We = W["embed"]                      # [vocab, d]
    x = We[np.asarray(seq)] * math.sqrt(d) + POS[:T]   # [T, d]
    sched = build_schedule(cfg["n_layers"], cfg["loop"], cfg["loop_mode"])
    # causal mask
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for bi in sched:
        p = f"b{bi}."
        h = rmsnorm(x, G[p + "ln1"])
        q = (h @ W[p + "wq"].T).reshape(T, nh, hd)
        k = (h @ W[p + "wk"].T).reshape(T, nh, hd)
        v = (h @ W[p + "wv"].T).reshape(T, nh, hd)
        a = np.empty((T, nh, hd), np.float64)
        for hh_ in range(nh):
            qh, kh, vh = q[:, hh_], k[:, hh_], v[:, hh_]
            sc = (qh @ kh.T) / math.sqrt(hd) + mask
            sc -= sc.max(axis=-1, keepdims=True)
            e = np.exp(sc)
            p_ = e / e.sum(axis=-1, keepdims=True)
            a[:, hh_] = p_ @ vh
        a = a.reshape(T, d)
        x = x + a @ W[p + "wo"].T
        h2 = rmsnorm(x, G[p + "ln2"])
        x = x + gelu(h2 @ W[p + "fc1"].T) @ W[p + "fc2"].T
    xl = rmsnorm(x[-1], G["lnf"])
    return We @ xl


class BPETok:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.maxlen = max(len(t) for t in vocab)
    import re as _re
    CHUNK = _re.compile(r"[AB]:|\n| ?[a-z]+(?:'[a-z]+)?| ?[0-9]+|[^\sa-z0-9]| ")

    def encode(self, text):
        out = []
        for w in self.CHUNK.findall(text):
            if w in self.stoi:
                out.append(self.stoi[w]); continue
            i = 0
            while i < len(w):
                j = min(len(w), i + self.maxlen)
                matched = False
                while j > i:
                    t = w[i:j]
                    if t in self.stoi:
                        out.append(self.stoi[t]); i = j; matched = True
                        break
                    j -= 1
                if not matched:
                    i += 1
        return out


def qrl8_greedy(buf: bytes, prompt: str, maxlen=60):
    cfg, vocab, W, G = unpack_qrl8(buf)
    tok = BPETok(vocab)
    POS = sin_pos(cfg["ctx"], cfg["dim"])
    ctx = cfg["ctx"]
    nl_id = tok.stoi["\n"]
    ids = tok.encode(f"A: {prompt}\nB:")
    gen = []
    for _ in range(maxlen):
        seq = ids[-ctx:]
        logits = qrl8_forward_last(seq, cfg, W, G, POS)
        nxt = argmax_first(logits)
        if nxt == nl_id:
            break
        gen.append(nxt)
        ids.append(nxt)
    text = "".join(vocab[t] for t in gen).strip()
    return gen, text, vocab

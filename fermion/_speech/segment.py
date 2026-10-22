"""Long-audio segmentation shared by both speech engines.

THE PROBLEM THIS SOLVES. The model is graded on utterances of up to ~30 s,
and both engines decode one array in one call with a token budget sized for
that regime (`engine.token_budget`: 7 tokens/s + 20, capped at 384). Handing
a whole 13-minute file to that call cannot work: the budget runs out after
~2 minutes of speech, and past ~3 minutes the decoder degenerates into a
repeated letter until the budget is gone. Measured on the published 0.1.22
package: a 5-minute clip came back as 212 words (about 920 expected), a
13-minute chapter as 196 words of which 65 were "P.". Fed 30 s windows,
the same engine returned 2,369 words for the chapter with no degenerate run,
at 39.5x real time and flat memory.

THE RULE. Audio up to `SINGLE_SHOT_MAX_S` (35 s) is decoded exactly as
before, in one call, so the published short-utterance behaviour is
byte-identical. Longer audio is cut into windows of 25-35 s, each window
ending in the quietest pause the band offers, and each window is decoded
with the single-utterance call and its own token budget. Window
transcripts are joined with single spaces, the same shape `fermion listen`
gives one session. The energy gate is `live.py`'s (room-tone floor 0.004
RMS, 0.18 of the window's own peak, 50 ms blocks), so `transcribe`,
`serve` and `listen` all agree on what a pause is.

WHY 30 s WINDOWS AND NOT LONGER. Measured on the 5-minute clip: 30 s and
60 s windows both return the full 919 words; 120 s windows return 774,
already dropping content. 30 s matches the graded regime and the live
session's segment cap, so it is the one window length with evidence behind
it.

WHY A PAUSE-ALIGNED CUT. A fixed 30 s cut lands mid-word roughly once per
window; a word split across two decodes is a word lost or doubled. Cutting
in the longest pause inside the 25-35 s band avoids that whenever a pause
exists; when the whole band is continuous speech, the cut falls at 30 s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SAMPLE_RATE = 16_000

#: Audio up to this length is decoded in one call, exactly as before.
SINGLE_SHOT_MAX_S = 35.0
#: A cut is searched for inside [BAND_MIN_S, BAND_MAX_S] of the window start.
BAND_MIN_S = 25.0
BAND_MAX_S = 35.0
#: Where the cut falls when the band holds no pause at all.
TARGET_S = 30.0

# The live session's energy gate, reused verbatim (see live.py).
BLOCK_S = 0.05
NOISE_FLOOR_RMS = 0.004
GATE_RATIO = 0.18


@dataclass
class Transcript:
    """Everything one transcription produced, for `--json`, `--verbose`
    and `verbose_json`.

    `text` is the transcript (windows joined with single spaces);
    `decode_seconds` is decode time only (no imports, no model load, no
    file read); `segments` carries one `{id, start, end, text}` per decoded
    window in seconds; `truncated` is True if ANY window used its whole
    token budget, which is the signal that output was cut short.

    `words` (0.2.7) is one `{text, start, end}` per word in seconds from the
    start of the file, built from the decoder's own token timings (see
    `words_from_tokens`) and shifted by each window's offset, so a word in
    the fourth window of a long file carries its position in the whole file.
    `tokens` is the same at sub-word level, `{text, start, end}` per decoded
    piece. Both are empty lists when the engine produced no timings; `timed`
    says whether it did (the Phonon-1 decoder has no timings, so a Phonon-1
    transcript has `timed=False` and no words).
    """

    text: str
    decode_seconds: float
    audio_seconds: float
    segments: list = field(default_factory=list)
    truncated: bool = False
    generation_tokens: int = 0
    words: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    timed: bool = False

    def triple(self):
        """The historical `(text, decode_seconds, audio_seconds)` shape."""
        return self.text, self.decode_seconds, self.audio_seconds


def plan(audio, sample_rate: int = SAMPLE_RATE) -> list:
    """Window boundaries for one mono float32 array: `[(start, end), ...]`
    in samples, contiguous, covering the whole array.

    One window for anything up to `SINGLE_SHOT_MAX_S`. Otherwise each cut
    is placed in the middle of the longest sub-gate run of 50 ms blocks
    inside the 25-35 s band after the window start (ties go to the run
    nearest 30 s); a band with no sub-gate block is cut at 30 s.
    """
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(audio.size)
    if n <= int(SINGLE_SHOT_MAX_S * sample_rate):
        return [(0, n)]

    block = int(BLOCK_S * sample_rate)
    n_blocks = -(-n // block)
    padded = np.zeros(n_blocks * block, dtype=np.float32)
    padded[:n] = audio
    rms = np.sqrt(np.mean(padded.reshape(n_blocks, block) ** 2, axis=1))

    single = int(SINGLE_SHOT_MAX_S * sample_rate)
    windows = []
    start = 0
    while n - start > single:
        lo = start + int(BAND_MIN_S * sample_rate)
        hi = start + int(BAND_MAX_S * sample_rate)
        b0 = -(-lo // block)          # first block fully inside the band
        b1 = hi // block              # one past the last block inside it
        peak = float(rms[start // block:b1].max()) if b1 > start // block else 0.0
        gate = max(NOISE_FLOOR_RMS, GATE_RATIO * peak)
        quiet = rms[b0:b1] <= gate
        cut = start + int(TARGET_S * sample_rate)
        if quiet.any():
            # Longest run of quiet blocks; cut in its middle.
            best = None                       # (length, -distance, mid_sample)
            i = 0
            while i < quiet.size:
                if not quiet[i]:
                    i += 1
                    continue
                j = i
                while j < quiet.size and quiet[j]:
                    j += 1
                mid = int(((b0 + i) + (b0 + j)) * block // 2)
                cand = (j - i, -abs(mid - (start + int(TARGET_S * sample_rate))), mid)
                if best is None or cand > best:
                    best = cand
                i = j
            cut = best[2]
        cut = max(lo, min(hi, cut, n))
        windows.append((start, cut))
        start = cut
    windows.append((start, n))
    return windows


def is_silent(audio) -> bool:
    """The engines' shared 'nothing here to decode' test."""
    import numpy as np

    return audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4


def _is_punctuation(piece: str) -> bool:
    """A token that carries no letter or digit (".", ",", "?", "'s" is NOT one)."""
    return bool(piece) and not any(ch.isalnum() for ch in piece)


def words_from_tokens(tokens, offset: float = 0.0, limit: float | None = None) -> list:
    """Merge decoder tokens into words with timings.

    `tokens` is a sequence of `(text, start, duration)` in seconds relative
    to the decoded window, in decode order; `text` is the token's piece with
    the SentencePiece word marker already turned into a leading space ("▁" ->
    " "), which is what both Phonon-2 decoders hand over. A token that
    starts with a space starts a new word; every other token continues the
    word before it, so "▁trans" + "cri" + "ption" is one word. A token made
    only of punctuation ("." "," "?" ...) is attached to the word before it
    even when it carries a leading space (the space stays inside the word
    text, exactly as it appears in the transcript), so no word is ever just
    a comma. A word's `start` is its first token's start and its `end` is its last
    token's end, both shifted by `offset` (the window's position in the
    file) and rounded to the millisecond. `limit` is the window's length in
    seconds: the decoder predicts a duration for its last token on a frame
    grid that can run up to one 80 ms frame past the end of the audio, so
    an `end` is never later than the audio it describes. Pieces that are
    only whitespace produce no word of their own. The word texts, joined
    with single spaces, equal the window's transcript (a transcript with two
    spaces in a row, which the decoder does not produce in practice, would
    join with one).
    """
    words: list = []
    cur: list = []           # the tokens of the word being built

    def flush():
        if not cur:
            return
        text = "".join(t[0] for t in cur).strip()
        if text:
            end = cur[-1][1] + cur[-1][2]
            if limit is not None:
                end = min(end, limit)
            words.append({"text": text,
                          "start": round(cur[0][1] + offset, 3),
                          "end": round(max(end, cur[0][1]) + offset, 3)})
        cur.clear()

    for tok in tokens:
        piece, start, duration = str(tok[0]), float(tok[1]), float(tok[2])
        # A bare word marker (the piece is only a space) starts the next word and is kept, so a transcript such
        # as "by125 ,000" gives the words "by125" and ",000" and the join still equals the text.
        starts_word = piece.startswith(" ")
        if starts_word and cur and (not piece.strip() or not _is_punctuation(piece.strip())):
            flush()
        cur.append((piece, start, duration))
    flush()
    return words


def _window_tokens(tokens, offset: float, limit: float | None = None) -> list:
    """`{text, start, end}` per sub-word piece, shifted to file time (ends clamped to the window like the words)."""
    out = []
    for tok in tokens:
        piece = str(tok[0])
        if not piece.strip():
            continue
        start = float(tok[1]); end = start + float(tok[2])
        if limit is not None:
            end = max(start, min(end, limit))
        out.append({"text": piece, "start": round(start + offset, 3), "end": round(end + offset, 3)})
    return out


def transcribe_windows(audio, decode_one, sample_rate: int = SAMPLE_RATE
                       ) -> Transcript:
    """Run `decode_one(window) -> (text, generation_tokens, token_budget)`
    over `plan(audio)` and assemble a `Transcript`.

    Both engines call this; the only engine-specific part is `decode_one`,
    which is each engine's unchanged single-utterance decode. A silent
    window (the engines' historical `< 1e-4` peak test) is skipped, not
    decoded, exactly as a silent file always was.

    0.2.7: `decode_one` may return a fourth element, the window's decoder
    tokens as `(text, start, duration)` triples in seconds relative to the
    window (both Phonon-2 decoders do; Phonon-1's cannot). Those are
    shifted by the window's start and merged into `Transcript.words` and
    `Transcript.tokens`. `text` is built exactly as before, from the window
    transcripts alone, so timings never change a transcript.
    """
    import time

    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    seconds = audio.size / sample_rate
    if is_silent(audio):
        return Transcript("", 0.0, seconds)
    started = time.perf_counter()
    texts = []
    segments = []
    words: list = []
    pieces: list = []
    timed = False
    truncated = False
    gen_total = 0
    for idx, (a, b) in enumerate(plan(audio, sample_rate)):
        window = audio[a:b]
        toks = None
        if is_silent(window):
            text, gen, budget = "", 0, 0
        else:
            from fermion._speech.mlx_memory import clear_mlx_cache

            try:
                res = decode_one(window)
                text, gen, budget = res[0], res[1], res[2]
                if len(res) > 3 and res[3] is not None:
                    toks = res[3]
                    timed = True
                gen_total += int(gen)
                if budget and gen >= budget:
                    truncated = True
            finally:
                clear_mlx_cache()
        text = (text or "").strip()
        if text:
            texts.append(text)
        offset = a / sample_rate
        if toks:
            limit = (b - a) / sample_rate
            words.extend(words_from_tokens(toks, offset, limit))
            pieces.extend(_window_tokens(toks, offset, limit))
        segments.append({"id": idx, "start": round(a / sample_rate, 3),
                         "end": round(b / sample_rate, 3), "text": text})
    decode_s = time.perf_counter() - started
    return Transcript(" ".join(texts), decode_s, seconds, segments,
                      truncated, gen_total, words, pieces, timed)

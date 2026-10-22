"""The rolling live-transcription loop behind `fermion listen`.

ONE SESSION, TWO FEEDS — that is the whole design. `LiveSession.feed()` takes
small blocks of 16 kHz mono float32 audio and does everything that makes the
command live: it rolls partial hypotheses while a phrase is in flight, cuts a
segment when the speaker pauses, and decodes the finalized segment. The two
feeders — `run_mic()` (a `sounddevice.InputStream`) and `run_wav()` (a file,
paced to real time) — do nothing but hand blocks to that one method, so the
`--wav` path exercises the IDENTICAL streaming stack and microphone capture is
the only layer a machine cannot test.

THE DECODE IS `transcribe`'s DECODE. Every partial and every final goes
through `SpeechModel.transcribe_array()` — the same `temperature=0.0`,
`repetition_penalty=1.05` call `fermion transcribe` makes — re-decoding the
whole in-flight segment each time, which is exactly the shipping Mac app's
rolling-partial pattern ("every partial re-decodes the WHOLE hold"). Because
the final decode of a segment sees the segment's complete audio in one call,
a `--wav` run over a single-utterance clip produces the batch transcript
byte for byte; the selftest asserts that.

THE CADENCE IS THE APP'S. First partial once ~0.35 s of audio is in the
segment, then one every ~0.5 s of audio — the shipping app's ramp values.
Partials self-clock: the next one is scheduled relative to where the buffer
stands AFTER a decode returns, so a long segment's slower re-decodes simply
space the partials out instead of piling up behind one another.

SEGMENTATION IS A SIMPLE ENERGY GATE, deliberately. A segment is cut after
~0.7 s of sub-threshold audio (or at a 30 s cap), where the threshold is the
louder of an absolute room-tone floor and a fraction of the segment's own
peak — the app's adaptive-gate idea (`CONVERSATION_NOISE_FLOOR` 0.004,
`CONVERSATION_GATE_RATIO` 0.18), reimplemented minimally. Nothing fed to the
session is ever dropped from the segment it belongs to: the cut INCLUDES the
trailing quiet, so the audio a final decode sees is a contiguous, gap-free
copy of what came in.
"""
from __future__ import annotations

import time

from fermion._speech.engine import SAMPLE_RATE

# ------------------------------------------------------------- the cadence
#: Audio in the segment before the FIRST partial decode (the app's ramp head).
FIRST_PARTIAL_S = 0.35
#: Audio added between partial decodes after that.
PARTIAL_EVERY_S = 0.50
#: Sub-threshold trailing audio that finalizes the segment ("they stopped").
SILENCE_CLOSE_S = 0.70
#: A segment is finalized at this length no matter what, so one unbroken
#: monologue cannot grow an unbounded buffer or an unbounded re-decode.
MAX_SEGMENT_S = 30.0
#: RMS at or below this is room tone (the app's CONVERSATION_NOISE_FLOOR)...
NOISE_FLOOR_RMS = 0.004
#: ...as is anything this far under the segment's own peak (its GATE_RATIO).
GATE_RATIO = 0.18
#: Feed granularity: 50 ms blocks, for both the microphone and `--wav`.
BLOCK_S = 0.05


class LiveSession:
    """Rolling-decode state for one `fermion listen` run.

    `on_partial(text)` fires with the CURRENT WHOLE HYPOTHESIS for the
    in-flight segment (not a delta — each call replaces the last), and
    `on_final(text)` fires once per finalized segment. Both receive stripped,
    non-empty text only. `transcript` joins the finals with single spaces,
    which is the same shape `transcribe` gives one file.
    """

    def __init__(self, speech, *, on_partial=None, on_final=None):
        self.speech = speech
        self.on_partial = on_partial or (lambda text: None)
        self.on_final = on_final or (lambda text: None)
        self.finals: list[str] = []
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._blocks: list = []       # every block of the in-flight segment
        self._samples = 0             # segment length, in samples
        self._voiced = False          # has anything above the gate been heard
        self._last_voice = 0          # sample count when voice was last heard
        self._peak_rms = 0.0          # loudest block so far (adaptive gate)
        self._next_partial = int(FIRST_PARTIAL_S * SAMPLE_RATE)

    # ------------------------------------------------------------- feeding
    def feed(self, block) -> None:
        """Accept one block of 16 kHz mono float32 samples."""
        import numpy as np

        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        self._blocks.append(block)
        self._samples += block.size

        rms = float(np.sqrt(np.mean(block * block)))
        self._peak_rms = max(self._peak_rms, rms)
        gate = max(NOISE_FLOOR_RMS, GATE_RATIO * self._peak_rms)
        if rms > gate:
            self._voiced = True
            self._last_voice = self._samples

        trailing = self._samples - self._last_voice
        if self._voiced and trailing >= int(SILENCE_CLOSE_S * SAMPLE_RATE):
            self._finalize()
        elif self._samples >= int(MAX_SEGMENT_S * SAMPLE_RATE):
            if self._voiced:
                self._finalize()
            else:
                # An idle microphone: 30 s of pure room tone holds nothing
                # worth decoding, and buffering it forever is the session's
                # one unbounded cost. Keep the last second so a word that
                # starts at the boundary keeps its onset.
                keep = np.concatenate(self._blocks)[-SAMPLE_RATE:]
                self._reset_segment()
                self._blocks = [keep]
                self._samples = keep.size
        elif self._voiced and self._samples >= self._next_partial:
            text, _, _ = self.speech.transcribe_array(
                np.concatenate(self._blocks))
            if text:
                self.on_partial(text)
            # Self-clocking: schedule off the buffer as it stands NOW, after
            # the decode, so slow re-decodes of a long segment cannot queue.
            self._next_partial = self._samples + int(
                PARTIAL_EVERY_S * SAMPLE_RATE)

    # ---------------------------------------------------------- finalizing
    def _finalize(self) -> None:
        import numpy as np

        audio = np.concatenate(self._blocks)
        self._reset_segment()
        text, _, _ = self.speech.transcribe_array(audio)
        if text:
            self.finals.append(text)
            self.on_final(text)

    def finish(self) -> str:
        """Finalize whatever is in flight; return the full transcript."""
        if self._samples and self._voiced:
            self._finalize()
        else:
            self._reset_segment()
        return self.transcript

    @property
    def transcript(self) -> str:
        return " ".join(self.finals)


def warm_up(speech) -> float:
    """One discarded decode, so the first partial lands on cadence.

    The first decode through a freshly loaded model pays the Metal graph
    compile — ~4-5 s of wall time that would otherwise land on the FIRST
    PARTIAL, turning the 0.35 s cadence into a five-second stall. The app's
    daemon solves this the same way (`stt.py:warm_up()` decodes a bundled
    clip at spin-up); the wheel bundles no audio, so this decodes half a
    second of fixed-seed low-level noise instead — bounded at
    `token_budget(0.5s)` = 24 tokens, output discarded. Returns the wall
    seconds it took, for `--verbose`.
    """
    import numpy as np

    noise = (np.random.default_rng(0)
             .standard_normal(int(0.5 * SAMPLE_RATE)) * 0.02
             ).astype(np.float32)
    t0 = time.monotonic()
    speech.transcribe_array(noise)
    return time.monotonic() - t0


# ------------------------------------------------------------- the feeders
def run_wav(session: LiveSession, path, *, realtime: bool = True) -> str:
    """Stream an audio FILE through the live path, paced to real time.

    The pacing sleeps against an absolute schedule (`t0 + samples/rate`), so
    time spent inside a partial decode is deducted from the next sleep exactly
    as a microphone would have kept capturing through it. `realtime=False` is
    for the selftest, which needs the identical code path without the clock.
    """
    from fermion._speech.engine import read_audio

    audio = read_audio(path)
    step = int(BLOCK_S * SAMPLE_RATE)
    t0 = time.monotonic()
    for i in range(0, len(audio), step):
        session.feed(audio[i:i + step])
        if realtime:
            delay = t0 + (i + step) / SAMPLE_RATE - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    return session.finish()


def run_mic(session: LiveSession) -> str:
    """Stream the default input device until Ctrl-C; return the transcript.

    The callback only enqueues copies — capture must never block behind a
    decode — and the consumer drains the queue completely after every decode
    so no audio is lost to a slow partial. `q.get` uses a short timeout so
    Ctrl-C lands promptly on every platform.
    """
    import queue

    import sounddevice as sd

    blocks: "queue.Queue" = queue.Queue()

    def _on_audio(indata, frames, time_info, status):  # sounddevice thread
        blocks.put(indata[:, 0].copy())

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32",
                                blocksize=int(BLOCK_S * SAMPLE_RATE),
                                callback=_on_audio)
        stream.start()
    except Exception as exc:                                # noqa: BLE001
        # A device/permission failure is an explanation, never a traceback —
        # the same contract as gate.require(). On macOS the usual cause is
        # that this terminal has no microphone permission.
        raise SystemExit(
            f"could not open the microphone ({exc}). On macOS, grant this "
            f"terminal microphone access in System Settings → Privacy & "
            f"Security → Microphone and retry. `fermion listen --wav "
            f"file.wav` runs the same live path from a file.")
    try:
        while True:
            try:
                session.feed(blocks.get(timeout=0.25))
            except queue.Empty:
                continue
            while True:                # drain what queued during the decode
                try:
                    session.feed(blocks.get_nowait())
                except queue.Empty:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
    return session.finish()

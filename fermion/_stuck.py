"""Stuck detector — the serve-side loop breaker for client-driven tool loops.

WHY THIS LIVES IN THE SERVER
----------------------------
`fermion serve` is stateless: an agent framework re-POSTs the WHOLE
conversation on every step, so the loop is visible in the request body. A
model that has emitted `read_file(sales.csv)` three times has told us so in
`messages`, and no per-connection state is needed to see it. Everything below
is therefore a pure function of the inbound `messages` (plus the `tools`
schemas, for the poll whitelist) — which is also why it is unit-testable on
CPU with no model and no server.

THE THREE SIGNALS (thresholds derived from an OpenHands-style production
pattern set: exact-hash >=3, alternating-pair >=6, same-tool-same-error >=2)

  (a) repeat_hash   the SAME tool call — name + canonicalised arguments,
                    sha256'd — appears >= `repeat` times (default 3).
  (b) alternating   the last `alternate` calls (default 6) are a strict
                    two-key A-B-A-B-A-B alternation. NOTE: one reading of the
                    source pattern set is ">=6 cycles" (= 12 calls); the
                    implementation follows the ">= 6 alternating calls" reading
                    (= 3 cycles) and exposes `--stuck-alt` so either reading
                    is one flag away. Disclosed, not silently resolved.
  (c) repeat_error  the same tool returned the same error-shaped payload
                    >= `error` times CONSECUTIVELY (default 2).

WHY A WHITELIST IS MANDATORY
----------------------------
A poll-type tool is SUPPOSED to be called with identical arguments until it
changes its answer, and it is SUPPOSED to return the same "not ready yet"
payload while it waits. Firing on those would turn a working agent into a
broken one, so poll-type calls are removed from all three sequences before any
signal is evaluated. A tool is poll-type when (i) the operator named it in
`--stuck-whitelist`, or (ii) its name contains a polling token
(`poll`/`wait`/`status`/`heartbeat`/...), or (iii) its own JSON-schema
`description` says it polls / waits / must be called repeatedly. (ii) and
(iii) are heuristics and can be turned off with `--stuck-no-auto-whitelist`;
they are deliberately GENEROUS, because a false whitelist only loses us a loop
break while a false trip breaks a working agent.

DISCLOSURE
----------
Every trip writes one structured single-line JSON record to stderr and bumps
counters exposed on `/health`, because a server that silently rewrites a turn
is a server nobody can debug. House doctrine: runtime enforcement never
appears in published benchmark numbers; what it does must be visible and
documented.

Stdlib only, and no `fermion` import at module scope: the CPU unit tests load
this file by path (`importlib.util.spec_from_file_location`) precisely so they
can prove that.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading

# `fermion.server` sets this to BRAND at import time. Do NOT hardcode the
# brand here: _brand.py is the one grep point for user-facing names, and this
# module has to stay importable without the package (see the docstring).
LOG_PREFIX: str | None = None

MODES = ("enforce", "observe", "off")

DEFAULT_REPEAT = 3            # (a) exact-hash repeats
DEFAULT_ALTERNATE = 6         # (b) alternating calls in the tail
DEFAULT_ERROR = 2             # (c) consecutive identical errors
DIAG_MAX_NEW = 192            # ceiling on the forced diagnosis turn

SIGNALS = ("repeat_hash", "alternating", "repeat_error")

# The two injected turns. Both are written to attack the MEASURED disease
# (fabricated values in arguments and in final answers, false success
# claims), so neither instruction invites the
# model to invent anything it has not read from a tool result.
DIAGNOSIS_INSTRUCTION = (
    "SERVER NOTE (inserted by the runtime, not by the user): this loop is "
    "stuck — {detail}. Tools are DISABLED for this turn; a tool call emitted "
    "now will NOT be executed. In at most three short sentences state: "
    "(1) what you have actually CONFIRMED from the tool results already "
    "present in this conversation, (2) why the repeated call is not making "
    "progress, (3) what you can tell the user with only that confirmed "
    "information. Do not state any value you have not read from a tool result."
)
FINAL_INSTRUCTION = (
    "SERVER NOTE (inserted by the runtime): tools remain DISABLED and this is "
    "the LAST turn of this request. Give the user your final answer now, "
    "using ONLY facts that appear in the tool results above. If the task "
    "could not be completed, say plainly what is missing and what you did "
    "confirm. Do not emit a tool call. Do not claim to have done anything the "
    "tool results do not show."
)


# --------------------------------------------------------------- poll whitelist
# Name tokens that mean "this is meant to be called again until it changes".
_POLL_TOKENS = frozenset("""
poll polls polled polling wait waits waited waiting waitfor await awaits
sleep watch watches watching tail follow follows ping pings pinging heartbeat
subscribe subscribes listen listens monitor monitors monitoring
status statuses state progress retry retries refresh refreshes
""".split())

_POLL_DESC_RE = re.compile(
    r"long[-\s]?poll"
    r"|\bpoll(?:s|ed|ing)?\b"
    r"|\bperiodic(?:ally)?\b"
    r"|\brepeatedly\b"
    r"|\bheartbeats?\b"
    r"|\bkeep[-\s]?alive\b"
    r"|\b(?:wait|waits|waiting|block|blocks|blocking)\s+(?:for|until)\b"
    r"|\buntil\s+(?:it|the|they|this|ready|complete|completed|done|finish|"
    r"finished|available|succeeds)\b"
    r"|\bcall\s+(?:this|it)\s+(?:again|repeatedly|periodically|in a loop)\b"
    r"|\bcheck(?:s|ing)?\s+(?:the\s+)?(?:status|progress|state)\b"
    r"|\b(?:current|latest)\s+status\b",
    re.I)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def name_tokens(name) -> list[str]:
    """`getJobStatus` / `mcp__ci__poll-run` -> lowercase word tokens."""
    spaced = _CAMEL.sub(" ", str(name or ""))
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t]


def _descriptions(tools) -> dict:
    """{tool name: description} from OpenAI-shaped or bare tool schemas."""
    out: dict = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if isinstance(name, str) and name:
            desc = fn.get("description")
            out[name] = desc if isinstance(desc, str) else ""
    return out


def is_poll_tool(name, *, descriptions=None, whitelist=(), auto=True) -> bool:
    """True when this tool is legitimately called over and over."""
    if not name:
        return False
    if name in whitelist:
        return True
    if not auto:
        return False
    if _POLL_TOKENS.intersection(name_tokens(name)):
        return True
    desc = (descriptions or {}).get(name) or ""
    return bool(desc and _POLL_DESC_RE.search(desc))


# ------------------------------------------------------------------ error shape
_ERROR_RE = re.compile(
    r"\berrors?\b|\berrno\b|\bexception\b|\btraceback\b|\bfail(?:ed|ure|s)?\b"
    r"|\bfatal\b|\bnot\s+found\b|\bno\s+such\s+(?:file|directory|table|key)\b"
    r"|\bdoes\s+not\s+exist\b|\b(?:permission|access)\s+denied\b|\bdenied\b"
    r"|\bunauthori[sz]ed\b|\bforbidden\b|\binvalid\b|\bmalformed\b"
    r"|\btimed?\s*out\b|\btimeout\b|\brefused\b|\bunreachable\b"
    r"|\bunable\s+to\b|\bcannot\b|\bcan'?t\b|\bunsupported\b"
    r"|\bbad\s+request\b|\brate\s*limit(?:ed)?\b|\bquota\b",
    re.I)

_ERROR_KEYS = ("error", "errors", "is_error", "iserror", "isError")
_ERROR_STATUS = frozenset(("error", "errored", "failed", "failure", "fail"))


def normalise_payload(text) -> str:
    """Lowercased, whitespace-collapsed, length-capped comparison key."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()[:400]


def looks_like_error(text, *, flag=None) -> bool:
    """Is this tool return an error?

    Three ways to say so, cheapest first: an explicit client flag
    (`is_error` on the tool message, an Anthropic/MCP-style convention some
    frameworks forward), a structured `{"error": ...}` / `{"status": "error"}`
    payload, or error vocabulary in free text.
    """
    if flag is not None:
        return bool(flag)
    raw = str(text or "")
    if not raw.strip():
        return False
    stripped = raw.strip()
    if stripped[:1] in "{[":
        try:
            obj = json.loads(stripped)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in _ERROR_KEYS:
                val = obj.get(key)
                if val:
                    return True
            status = obj.get("status") or obj.get("state")
            if isinstance(status, str) and status.strip().lower() in _ERROR_STATUS:
                return True
            ok = obj.get("ok")
            if ok is False:
                return True
    return bool(_ERROR_RE.search(raw))


# ------------------------------------------------------------------ extraction
def canon_args(args) -> str:
    """Canonical JSON for the arguments, whatever shape the client sent.

    `_norm_message` hands us either a parsed object (the common case, because
    `render_assistant_tool_calls` parses tolerantly) or the raw string when it
    could not be parsed. Both must hash the same when they mean the same
    thing, so key order and whitespace are normalised away.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return re.sub(r"\s+", " ", args).strip()
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
    except Exception:                                    # unhashable/odd keys
        return re.sub(r"\s+", " ", repr(args)).strip()


def call_hash(name, args) -> str:
    """The 'exact hash' of signal (a). 16 hex chars is ample for a transcript."""
    blob = f"{name}\x00{canon_args(args)}".encode("utf-8", "replace")
    return hashlib.sha256(blob).hexdigest()[:16]


class Call:
    __slots__ = ("name", "hash", "id", "poll", "turn")

    def __init__(self, name, hash_, id_, poll, turn):
        self.name, self.hash, self.id, self.poll, self.turn = \
            name, hash_, id_, poll, turn

    def __repr__(self):                                  # pragma: no cover
        return f"Call({self.name!r},{self.hash},poll={self.poll})"


class Result:
    __slots__ = ("name", "norm", "error", "poll", "turn")

    def __init__(self, name, norm, error, poll, turn):
        self.name, self.norm, self.error, self.poll, self.turn = \
            name, norm, error, poll, turn

    def __repr__(self):                                  # pragma: no cover
        return f"Result({self.name!r},error={self.error},poll={self.poll})"


def extract(messages, tools=None, *, whitelist=(), auto=True):
    """`(calls, results)` in transcript order from an OpenAI message list.

    Tool returns are paired to their call by `tool_call_id` when the client
    sends one (every mainstream framework does), then by the tool message's
    `name`, and finally positionally against the most recent unpaired call —
    so the error signal still works for a client that sends neither.
    """
    descriptions = _descriptions(tools)

    def poll(name):
        return is_poll_tool(name, descriptions=descriptions,
                            whitelist=whitelist, auto=auto)

    calls: list[Call] = []
    results: list[Result] = []
    by_id: dict = {}
    unpaired: list[Call] = []
    for turn, msg in enumerate(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) \
                    else tc
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    continue
                call = Call(name, call_hash(name, fn.get("arguments")),
                            tc.get("id"), poll(name), turn)
                calls.append(call)
                unpaired.append(call)
                if call.id:
                    by_id[call.id] = call
        elif role == "tool":
            owner = by_id.get(msg.get("tool_call_id"))
            if owner is None and isinstance(msg.get("name"), str):
                want = msg["name"]
                owner = next((c for c in reversed(unpaired)
                              if c.name == want), None)
            if owner is None and unpaired:
                owner = unpaired[-1]
            if owner is not None and owner in unpaired:
                unpaired.remove(owner)
            name = owner.name if owner is not None else (msg.get("name") or "")
            content = msg.get("content")
            results.append(Result(
                name, normalise_payload(content),
                looks_like_error(content, flag=msg.get("is_error")),
                owner.poll if owner is not None else poll(name), turn))
    return calls, results


# --------------------------------------------------------------------- signals
class Trip:
    """One fired signal, ready for stderr/health/report."""

    def __init__(self, signal, *, detail, tool=None, count=0, threshold=0):
        self.signal = signal
        self.detail = detail
        self.tool = tool
        self.count = int(count)
        self.threshold = int(threshold)
        self.also: list[str] = []

    def as_dict(self) -> dict:
        return {"signal": self.signal, "tool": self.tool, "count": self.count,
                "threshold": self.threshold, "detail": self.detail,
                "also": list(self.also)}

    def __repr__(self):                                  # pragma: no cover
        return f"Trip({self.signal},{self.tool!r},{self.count})"


def signal_repeat_hash(calls, threshold) -> Trip | None:
    counts: dict = {}
    for c in calls:
        counts[c.hash] = counts.get(c.hash, 0) + 1
    if not counts:
        return None
    top, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if n < threshold:
        return None
    name = next(c.name for c in calls if c.hash == top)
    return Trip("repeat_hash", tool=name, count=n, threshold=threshold,
                detail=(f"the identical call {name}(...) was issued {n} times "
                        f"(argument-exact, hash {top})"))


def signal_alternating(calls, threshold) -> Trip | None:
    if threshold < 4 or len(calls) < threshold:
        return None
    tail = calls[-threshold:]
    a, b = tail[0].hash, tail[1].hash
    if a == b:
        return None
    for i, c in enumerate(tail):
        if c.hash != (a if i % 2 == 0 else b):
            return None
    names = f"{tail[0].name} <-> {tail[1].name}"
    return Trip("alternating", tool=tail[0].name, count=threshold,
                threshold=threshold,
                detail=(f"the last {threshold} calls alternate between two "
                        f"fixed calls ({names}) with no other action between "
                        f"them"))


def signal_repeat_error(results, threshold) -> Trip | None:
    key = None
    run = 0
    for r in results:
        if not r.error:
            key, run = None, 0
            continue
        this = (r.name, r.norm)
        run = run + 1 if this == key else 1
        key = this
        if run >= threshold:
            snippet = r.norm[:120]
            return Trip("repeat_error", tool=r.name, count=run,
                        threshold=threshold,
                        detail=(f"{r.name} returned the same error {run} times "
                                f"in a row: {snippet!r}"))
    return None


def evaluate(calls, results, *, repeat, alternate, error) -> list[Trip]:
    """Every signal that fires, most-specific first."""
    out = []
    for trip in (signal_repeat_hash(calls, repeat),
                 signal_alternating(calls, alternate),
                 signal_repeat_error(results, error)):
        if trip is not None:
            out.append(trip)
    return out


# -------------------------------------------------------------------- detector
def _log(payload: dict) -> None:
    """One structured line per event, on stderr, always."""
    prefix = f"[{LOG_PREFIX}] " if LOG_PREFIX else ""
    sys.stderr.write(f"{prefix}stuck-detector "
                     f"{json.dumps(payload, sort_keys=True, default=str)}\n")
    sys.stderr.flush()


class StuckDetector:
    """Stateless analysis + process-lifetime counters.

    `mode`:
      enforce  trip -> the server forces a no-tools diagnosis turn and then a
               final answer (the loop cannot continue).
      observe  trip -> logged and counted, output UNCHANGED. The zero-risk
               setting: it tells you how often the detector would have fired
               on your own traffic before you let it act.
      off      no analysis at all.
    """

    def __init__(self, mode: str = "enforce", *, repeat: int = DEFAULT_REPEAT,
                 alternate: int = DEFAULT_ALTERNATE, error: int = DEFAULT_ERROR,
                 whitelist=(), auto_whitelist: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.repeat = max(2, int(repeat))
        self.alternate = max(4, int(alternate))
        self.error = max(2, int(error))
        self.whitelist = frozenset(
            n.strip() for n in (whitelist or ()) if str(n).strip())
        self.auto_whitelist = bool(auto_whitelist)
        self._lock = threading.Lock()
        self.counters = {
            "requests_checked": 0,
            "tool_calls_seen": 0,
            "poll_calls_whitelisted": 0,
            "trips": 0,
            "trips_by_signal": {s: 0 for s in SIGNALS},
            "observed_only": 0,
            "whitelist_suppressed": 0,
            "whitelist_suppressed_by_signal": {s: 0 for s in SIGNALS},
            "diagnosis_turns": 0,
            "forced_answers": 0,
            "last_trip": None,
        }

    # ------------------------------------------------------------------ api
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def is_poll(self, name, tools=None) -> bool:
        return is_poll_tool(name, descriptions=_descriptions(tools),
                            whitelist=self.whitelist, auto=self.auto_whitelist)

    def check(self, messages, tools=None) -> Trip | None:
        """The whole detector. Returns the winning `Trip`, or None.

        Callers act only when `self.mode == "enforce"`; `observe` returns the
        trip too (so a caller can annotate) but is documented as no-change and
        the server does not intervene on it.
        """
        if self.mode == "off":
            return None
        calls, results = extract(messages, tools,
                                 whitelist=self.whitelist,
                                 auto=self.auto_whitelist)
        if not calls and not results:
            return None
        kept_calls = [c for c in calls if not c.poll]
        kept_results = [r for r in results if not r.poll]
        n_polled = (len(calls) - len(kept_calls)) + (len(results) - len(kept_results))

        trips = evaluate(kept_calls, kept_results, repeat=self.repeat,
                         alternate=self.alternate, error=self.error)
        with self._lock:
            self.counters["requests_checked"] += 1
            self.counters["tool_calls_seen"] += len(calls)
            self.counters["poll_calls_whitelisted"] += n_polled

        if not trips:
            if n_polled:
                # Prove the whitelist did something: would the same transcript
                # have tripped WITHOUT it? Counted and logged, because a
                # silent suppressor is indistinguishable from a broken one.
                raw = evaluate(calls, results, repeat=self.repeat,
                               alternate=self.alternate, error=self.error)
                if raw:
                    with self._lock:
                        self.counters["whitelist_suppressed"] += 1
                        self.counters["whitelist_suppressed_by_signal"][
                            raw[0].signal] += 1
                    _log({"event": "suppressed", "mode": self.mode,
                          "signal": raw[0].signal, "tool": raw[0].tool,
                          "count": raw[0].count,
                          "reason": "poll-type tool whitelisted",
                          "detail": raw[0].detail})
            return None

        trip = trips[0]
        trip.also = [t.signal for t in trips[1:]]
        record = trip.as_dict()
        with self._lock:
            self.counters["trips"] += 1
            self.counters["trips_by_signal"][trip.signal] += 1
            if self.mode == "observe":
                self.counters["observed_only"] += 1
            self.counters["last_trip"] = dict(record, mode=self.mode)
        _log({"event": "trip", "mode": self.mode,
              "action": ("diagnosis+forced_answer" if self.mode == "enforce"
                         else "logged only (observe mode)"),
              "tool_calls_in_history": len(calls),
              "poll_calls_whitelisted": n_polled, **record})
        return trip

    # -------------------------------------------------------------- counters
    def note_diagnosis(self, text: str = "") -> None:
        with self._lock:
            self.counters["diagnosis_turns"] += 1
            last = self.counters.get("last_trip")
            if isinstance(last, dict) and text:
                last["diagnosis"] = text[:400]

    def note_forced_answer(self) -> None:
        with self._lock:
            self.counters["forced_answers"] += 1

    def health(self) -> dict:
        with self._lock:
            counters = json.loads(json.dumps(self.counters, default=str))
        return {
            "mode": self.mode,
            "thresholds": {"repeat_hash": self.repeat,
                           "alternating_calls": self.alternate,
                           "consecutive_same_error": self.error},
            "diagnosis_max_tokens": DIAG_MAX_NEW,
            "poll_whitelist": sorted(self.whitelist),
            "poll_whitelist_auto": self.auto_whitelist,
            "on_trip": ("one tools-disabled diagnosis turn, then a forced "
                        "final answer" if self.mode == "enforce"
                        else "nothing (logged and counted only)"),
            "counters": counters,
        }


def from_args(args) -> StuckDetector:
    """Build the detector from parsed `serve` flags (see cli.py)."""
    raw = getattr(args, "stuck_whitelist", None) or ""
    names = [p for p in re.split(r"[,\s]+", str(raw)) if p]
    return StuckDetector(
        getattr(args, "stuck_detector", "enforce"),
        repeat=getattr(args, "stuck_repeat", DEFAULT_REPEAT),
        alternate=getattr(args, "stuck_alt", DEFAULT_ALTERNATE),
        error=getattr(args, "stuck_error", DEFAULT_ERROR),
        whitelist=names,
        auto_whitelist=not getattr(args, "stuck_no_auto_whitelist", False))

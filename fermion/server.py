"""fermion serve: an OpenAI-compatible HTTP endpoint over the published decode path.

Standard library only (`http.server` + `threading`): no FastAPI, no uvicorn,
no new wheel dependency. Every token is produced by `fermion.cli.iter_generate`
— the same function `fermion chat` and `fermion generate` call — so served
output is token-identical to CLI output for the same messages and sampler
settings (gated).

Routes
  POST /v1/chat/completions   (streaming + non-streaming, `tools` supported)
  POST /v1/completions        (legacy plain-text clients)
  GET  /v1/models             (also /v1/models/{id})
  GET  /health                (also /v1/health)
  POST /v1/audio/transcriptions  (speech models only)
  GET  /v1/audio/stream       (speech models only; WebSocket, hand-rolled
                               RFC 6455 in fermion/_ws.py — still stdlib only)

TOOL CALLING (0.1.7). `tools`, `tool_choice`, inbound assistant `tool_calls`
and inbound `role: "tool"` messages are all handled, because every mainstream
agent framework drives its loop off `choices[].message.tool_calls` and nothing
else — some frameworks actively DELETE `<tool_call>` XML out of assistant
content, so a server that only returns the raw text is invisible to them
(measured). Rendering goes through
the model repo's OWN chat template `tools` branch (`tools=` is passed to
`apply_chat_template`; we never hand-build a preamble), and the reply is
decoded by the tolerant parser in `fermion._toolcalls`.

Defaults are the published sampler config (temp 0.01 / top-p 1.0 /
rep-pen 1.05 / pen-window 256). A client that explicitly asks for
`temperature: 0` gets a DETERMINISTIC argmax with the repetition penalty
still applied — the native runtime no-ops its sampler flags at temp 0;
the torch path does not, so
we honour both settings instead of silently dropping one.

LOOP CONTROL + TOOL SAMPLING (0.1.13). Two serve-side scaffolds, both
disclosed, both flagged:

  * `--stuck-detector` (default `enforce`): the request body carries the whole
    conversation, so a client-driven tool loop that is going nowhere is
    VISIBLE here. Three signals (identical call >=3, A-B-A-B alternation >=6,
    same tool + same error >=2 in a row), poll-type tools whitelisted, and on a
    trip the turn becomes one tools-disabled diagnosis generation followed by a
    forced final answer — so the client's loop ends with an answer instead of a
    fourth identical call. `observe` logs without changing output; `off`
    disables the analysis. Every trip prints a structured JSON line to stderr
    and bumps counters on /health. See fermion/_stuck.py.
  * `--tool-profile` (default `graded` = NO CHANGE): an opt-in sampler profile
    applied ONLY to requests that carry `tools`. The default keeps the
    published config, which every published number was measured at; the
    vendor-guided anti-repetition profiles are one flag away and named on
    /health so a run can never be silently mis-attributed.

Bind is 127.0.0.1 by default. The server is unauthenticated unless
`--api-key` is passed; binding a non-loopback host prints a warning.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fermion import cli as _cli
from fermion import _catalog, _stuck, _toolcalls, _ws
from fermion._brand import BRAND, VERSION

# _stuck.py is deliberately brand-free (it is loaded by path in the CPU unit
# tests, where importing `fermion` would drag in torch). The ONE place the two
# meet is here.
_stuck.LOG_PREFIX = BRAND

MAX_BODY = 32 * 1024 * 1024          # 32 MB request cap
OWNER = "fermionresearch"

# A live stream that sends nothing for this long is presumed dead. It matters
# more here than on an HTTP route because a stream session HOLDS the one-at-a-
# time slot (`SpeechEngine.stream_guard`): an abandoned socket must time out
# and release it, or every later stream gets "engine busy" forever.
STREAM_IDLE_S = 90.0

# Hard ceiling on `max_tokens`, in tokens, when the container's config does not
# declare one. This exists because a client asking for 65,536 tokens on a
# ~25 tok/s CPU decode path is ALWAYS a mistake and until 0.1.7 we obliged:
# `max_new()` returned the request value verbatim. A popular agent framework
# requests 65536 on every step (measured), which is
# effectively "run forever" whenever the model misses a stop token. The clamp
# is silent-but-logged and returns the reduced value; it never 400s, because
# the request is otherwise perfectly valid and a well-behaved client should not
# have to know our decode rate.
DEFAULT_CONTEXT = 40_960

# SAMPLER PROFILES FOR TOOL-CARRYING REQUESTS (`--tool-profile`).
#
# `graded` is the default and is a NO-OP: it keeps the banked shipping config
# (temp 0.01 / top-p 1.0 / rep-pen 1.05 / window 256) that every published
# number was measured at. The others are opt-in via `--tool-profile` and apply
# ONLY to requests that carry `tools`; a plain chat request is never touched.
#
# Provenance and honesty notes, because these numbers are borrowed:
#   * `vendor` = Qwen3's own NON-THINKING sampling guidance (temp 0.7,
#     top_p 0.8). Non-thinking is the right row: `cli._chat_ids` renders with
#     `enable_thinking=False`.
#   * `vendor-thinking` = Qwen3's thinking-mode row (0.6 / 0.95).
#   * Qwen's quantized-model guidance is `presence_penalty` up to 1.5. THIS
#     RUNTIME HAS NO presence_penalty: the only repetition lever is the
#     CTRL-style multiplicative windowed penalty (`--rep-penalty`, mirrored by
#     the native runtime). 1.10/1.15 below are the same-direction stand-ins,
#     NOT the vendor's number in different units, and they are UNMEASURED on
#     our containers. Nothing here is a graded configuration.
#   * top_k and min_p are also unimplemented (the CLI pins top_k=0 on the torch
#     path on purpose), so "min-p 0" is already true by construction.
TOOL_PROFILES = {
    "graded": {},
    "vendor": {"temperature": 0.7, "top_p": 0.8,
               "rep_penalty": 1.10, "pen_window": 256},
    "vendor-thinking": {"temperature": 0.6, "top_p": 0.95,
                        "rep_penalty": 1.10, "pen_window": 256},
    "antiloop": {"temperature": 0.7, "top_p": 0.8,
                 "rep_penalty": 1.15, "pen_window": 512},
}

# OpenAI sampler fields this runtime does not implement. Until 0.1.13 they were
# read by nobody and dropped in silence, so a client that set
# `presence_penalty: 1.5` (exactly Qwen's quantized-model advice) got the
# graded config and no hint that its request had been ignored. Now it is told
# once per process per field.
UNSUPPORTED_SAMPLER_KNOBS = {
    "presence_penalty": ("not implemented; the nearest lever is "
                         "--rep-penalty (windowed CTRL-style, multiplicative)"),
    "frequency_penalty": ("not implemented; the nearest lever is "
                          "--rep-penalty"),
    "top_k": "not implemented (the torch path pins top_k=0; top-p is the only "
             "shortlist knob)",
    "min_p": "not implemented (equivalent to min_p=0)",
    "logit_bias": "not implemented",
    "seed": "not implemented (no seeded RNG on either backend)",
    "logprobs": "not implemented (logprobs are never returned)",
    "top_logprobs": "not implemented",
}


# ------------------------------------------------------------------ helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "")


class _UnixThreadingHTTPServer(ThreadingHTTPServer):
    """The same HTTP server over a Unix-domain socket (`fermion serve --unix-socket PATH`).

    For dictation tools and local agents that keep one warm engine per machine: no TCP port to pick, filesystem
    permissions instead of --api-key, and a stale socket file from a crashed server is replaced at bind. Clients:
    `curl --unix-socket PATH http://localhost/v1/audio/transcriptions ...`, requests-unixsocket, or any OpenAI SDK
    with a Unix-socket transport. The handler sees client_address ("unix", 0) so its logging is unchanged."""
    import socket as _socket
    address_family = getattr(_socket, "AF_UNIX", -1)     # absent on some Windows Pythons: serve() refuses --unix-socket there
    allow_reuse_address = False

    def server_bind(self):
        import os as _os
        path = self.server_address
        try:
            if _os.path.exists(path) and _os.stat(path).st_mode & 0o170000 == 0o140000:
                _os.unlink(path)                            # a socket file left by a dead server
        except OSError:
            pass
        super().server_bind()
        try:
            _os.chmod(path, 0o600)                          # owner only: the server is unauthenticated
        except OSError:
            pass
        self.server_address = (path, 0)                     # (host, port) shape the banner code expects

    def get_request(self):
        conn, _ = self.socket.accept()
        return conn, ("unix", 0)

    def server_close(self):
        import os as _os
        super().server_close()
        try:
            _os.unlink(self.server_address[0])
        except OSError:
            pass


def _norm_message(msg) -> dict:
    """OpenAI message -> {role, content} with content flattened to text.

    An assistant turn's `tool_calls` and a tool turn's `tool_call_id` are kept
    (normalised): the chat template renders both — `tool_calls` back to
    `<tool_call>` XML and a `role: "tool"` turn to a `<tool_response>` block
    inside a user turn — so a multi-turn tool loop replays as the model was
    trained to see it, with no re-encoding by us.
    """
    if not isinstance(msg, dict):
        raise ValueError("each item of 'messages' must be an object")
    role = msg.get("role")
    if role not in ("system", "user", "assistant", "developer", "tool"):
        raise ValueError(f"unsupported message role {role!r}")
    content = msg.get("content")
    if content is None:
        content = ""
    if isinstance(content, list):        # multimodal-shaped parts
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in (None, "text"):
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
            else:
                raise ValueError("only text content parts are supported")
        content = "".join(parts)
    if not isinstance(content, str):
        raise ValueError("message 'content' must be a string or text parts")
    if role == "developer":              # newer alias for system
        role = "system"
    out = {"role": role, "content": content}
    if role == "assistant" and msg.get("tool_calls") is not None:
        raw = msg["tool_calls"]
        if not isinstance(raw, list):
            raise ValueError("assistant 'tool_calls' must be an array")
        calls = _toolcalls.render_assistant_tool_calls(raw)
        if calls:
            out["tool_calls"] = calls
        elif not content:
            raise ValueError("assistant 'tool_calls' entries need a "
                             "function.name")
    if role == "tool":
        tcid = msg.get("tool_call_id")
        if tcid is not None and not isinstance(tcid, str):
            raise ValueError("'tool_call_id' must be a string")
        if tcid:
            out["tool_call_id"] = tcid
        if msg.get("name"):
            out["name"] = msg["name"]
    return out


def _norm_tools(body: dict) -> tuple[list[dict], object]:
    """`(tools, tool_choice)` ready for the chat template.

    Returns `([], choice)` whenever nothing should be rendered, which is the
    0.1.6 code path exactly. `tool_choice` semantics on this runtime:

      absent / "auto"        render every tool, parse any calls back  (default)
      "none"                 render NOTHING and parse nothing, so the turn is
                             plain prose. Rendering the tools and then
                             suppressing the calls would leave raw
                             `<tool_call>` XML in `content`, which is worse
                             than either honest option.
      "required" / {...}     BEST EFFORT. This is a plain decode path with no
                             grammar constraint, so a call cannot be *forced*;
                             for a named function we narrow the rendered
                             `tools` to that one function, which is the only
                             real steering available. If the model still
                             answers in prose we return `finish_reason: "stop"`
                             rather than inventing a call.
    """
    tools = body.get("tools")
    choice = body.get("tool_choice")
    if tools is None and body.get("functions") is not None:
        # pre-2023-11 shape: [{name, description, parameters}] with no wrapper
        legacy = body.get("functions")
        if not isinstance(legacy, list):
            raise _BadRequest("'functions' must be an array", "functions")
        tools = [{"type": "function", "function": f} for f in legacy]
        if choice is None and body.get("function_call") is not None:
            choice = body["function_call"]
    if tools is None:
        return [], choice
    if not isinstance(tools, list):
        raise _BadRequest("'tools' must be an array", "tools")

    want_name = None
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise _BadRequest("'tool_choice' must be 'auto', 'none', "
                              "'required', or a {type, function} object",
                              "tool_choice")
        if choice == "none":
            return [], choice
    elif isinstance(choice, dict):
        fn = choice.get("function")
        want_name = fn.get("name") if isinstance(fn, dict) else choice.get("name")
        if not isinstance(want_name, str) or not want_name:
            raise _BadRequest("'tool_choice.function.name' is required when "
                              "'tool_choice' is an object", "tool_choice")
    elif choice is not None:
        raise _BadRequest("'tool_choice' must be a string or an object",
                          "tool_choice")

    out = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            raise _BadRequest(f"tools[{i}] must be an object", "tools")
        if t.get("type") not in (None, "function"):
            raise _BadRequest(f"tools[{i}].type {t['type']!r} is not "
                              f"supported (only 'function')", "tools")
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise _BadRequest(f"tools[{i}].function.name is required", "tools")
        entry = {"type": "function",
                 "function": {k: v for k, v in fn.items()
                              if k in ("name", "description", "parameters")}}
        out.append(entry)
    if want_name is not None:
        picked = [t for t in out if t["function"]["name"] == want_name]
        if not picked:
            raise _BadRequest(f"tool_choice names {want_name!r}, which is not "
                              f"in 'tools'", "tool_choice")
        out = picked
    return out, choice


def _norm_stop(stop) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise ValueError("'stop' must be a string or an array of strings")
    if len(stop) > 4:
        raise ValueError("'stop' accepts at most 4 sequences")
    return [s for s in stop if s]


def _stop_filter(pieces, stops, state):
    """Withhold text that is (or could become) a stop sequence.

    Never abandons `pieces` early: the underlying generator is always drained
    so `iter_generate` can join its worker thread and fill the usage stats.
    """
    buf = ""
    done = False
    for piece in pieces:
        if done:
            continue                       # drain, discard
        buf += piece
        hits = [i for i in (buf.find(s) for s in stops) if i >= 0]
        if hits:
            state["hit"] = True
            out, buf, done = buf[:min(hits)], "", True
            if out:
                yield out
            continue
        hold = 0                           # tail that may start a stop seq
        for s in stops:
            for k in range(min(len(s) - 1, len(buf)), 0, -1):
                if buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        if hold < len(buf):
            yield buf[:len(buf) - hold]
            buf = buf[len(buf) - hold:]
    if not done and buf:
        yield buf


# ------------------------------------------------------------------- engine
class Engine:
    """Loads the container once; serialises generation behind one lock."""

    def __init__(self, args):
        self.args = args
        # 0.2.6: a name that is neither a local path nor a published model is refused in one line, before any Hub
        # request (0.2.5 printed an httpx 401 traceback for `fermion serve nosuchmodel`); the catalog is the authority.
        spec = str(args.model)
        if not Path(spec).expanduser().exists() and _catalog.entry_for(_catalog.resolve_alias(spec)) is None:
            raise SystemExit(f"unknown model {spec!r}. `{BRAND} models` lists the published models; "
                             f"`{BRAND} serve phonon-2` serves speech, `{BRAND} serve neutrino-8b` serves chat.")
        try:
            self.kind, self.path = _cli.resolve_model(args.model)
        except SystemExit:
            raise
        except Exception as exc:                                   # Hub / network errors: one line, no traceback
            raise SystemExit(f"could not fetch {spec}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}") from None
        self.container = self._container()
        self.sha256 = _sha256(self.container)
        self.model_id = args.served_model_name or self.container.name
        self.created = int(time.time())
        self.lock = threading.Lock()
        self.model = self.tokenizer = self.draft = None
        self.tiny_buf = None
        self.header = None
        # Reliability scaffolds. Both are pure serve-side policy: no weights,
        # no kernels, no effect on any other surface.
        self.stuck = _stuck.from_args(args)
        self.tool_profile = getattr(args, "tool_profile", "graded") or "graded"
        if self.tool_profile not in TOOL_PROFILES:      # argparse also guards
            raise SystemExit(f"--tool-profile must be one of "
                             f"{', '.join(TOOL_PROFILES)}")
        self._warned_knobs: set = set()
        if self.kind == "tiny":
            self.tiny_buf = self.container.read_bytes()
            print(f"[{BRAND}] loaded tiny artifact {self.container.name} "
                  f"({len(self.tiny_buf)} bytes, "
                  f"{self.tiny_buf[:4].decode()})", file=sys.stderr)
        else:
            self.model, self.tokenizer = _cli.load_trtc(
                self.kind, self.path, args.device, args.dtype,
                **_cli._backend_kwargs(args))
            try:
                import fermion
                self.header = fermion.read_header(self.container)
            except Exception:                     # pragma: no cover
                self.header = None
            if getattr(args, "draft", None):
                self._load_draft(args.draft)
            if getattr(self.model, "backend", "") == "native" \
                    and not getattr(args, "no_session", False):
                # 0.1.7: ONE resident runner process holds the weights + KV
                # cache across requests; each request prefills only the prompt
                # tail the cache has not seen. This is what collapses an agent
                # step from ~2-4 min (full ~4.3k-token preamble re-prefilled
                # per step, measured) to seconds. Falls back to
                # one-shot-per-request automatically (old runner binary,
                # oversize request, or a session crash).
                self.model.enable_session(getattr(args, "session_ctx", 8192))
                print(f"[{BRAND}] {self.model.session_note}", file=sys.stderr)
                # 0.1.8: the CONFIG TRAP, called out at startup instead of
                # discovered one silent 2-minute turn at a time. The session
                # engages only when prompt + max_tokens + 1 <= session ctx;
                # with the default --max-new-ceiling this can be impossible
                # for every realistic agent preamble (measured live:
                # `--session-ctx 4096` + a client max_tokens 65536 clamped to
                # 4096 = zero KV reuse on every step, ~150 s/step).
                sctx = getattr(self.model, "_session_ctx", 0) or 0
                ceil = int(getattr(args, "max_new_ceiling", 0) or 0)
                if sctx and ceil and sctx - ceil - 1 < 1024:
                    print(f"[{BRAND}] WARNING: --max-new-ceiling {ceil} "
                          f"leaves only {max(0, sctx - ceil - 1)} positions "
                          f"of --session-ctx {sctx} for the prompt; any "
                          f"longer prompt FALLS BACK to one-shot mode (full "
                          f"prefill, no KV reuse — minutes per agent step on "
                          f"CPU). Agent preambles are typically 2-4k tokens: "
                          f"use --max-new-ceiling 512 or raise --session-ctx.",
                          file=sys.stderr)
        self.context = self._context_len()

    def _context_len(self) -> int:
        """Token budget the clamp measures against.

        The MODEL's config is the only source consulted. `tokenizer
        .model_max_length` is deliberately NOT used: it carries the tokenizer
        family's declared maximum (131,072 for the Qwen3 donor tokenizer these
        containers ship), not this model's trained window (40,960), so trusting
        it would have made the clamp 3.2x too loose and let a 65,536-token
        request straight through — exactly the defect being fixed. A config that
        genuinely declares a window is believed; anything absent or nonsensical
        (some configs carry a 1e30 sentinel) falls back to DEFAULT_CONTEXT.
        """
        cfg = getattr(self.model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and not isinstance(v, bool) \
                    and 1024 <= v <= 1 << 22:
                return v
        return DEFAULT_CONTEXT

    def _container(self) -> Path:
        if self.kind == "trtc-dir":
            from fermion._container import find_container
            return find_container(self.path)
        return Path(self.path)

    def _load_draft(self, spec: str) -> None:
        kind, path = _cli.resolve_model(spec)
        if kind == "tiny":
            raise SystemExit("--draft must be a TRTC v4 container, not a tiny "
                             "artifact")
        # the draft is handed to transformers as an `assistant_model`, which
        # is a torch-graph feature: it is always the reference backend.
        draft, dtok = _cli.load_trtc(kind, path, self.args.device,
                                     self.args.dtype, backend="torch")
        if draft.config.vocab_size != self.model.config.vocab_size:
            raise SystemExit(
                f"--draft vocab {draft.config.vocab_size} != main model vocab "
                f"{self.model.config.vocab_size}; speculative decoding needs a "
                f"shared tokenizer")
        self.draft = draft
        print(f"[{BRAND}] draft model attached (speculative decoding): "
              f"{Path(path).name}", file=sys.stderr)

    # -------------------------------------------------------------- sampling
    def profile_for(self, tools: bool) -> dict:
        """The tool-profile sampler overrides in force for this request
        (may be empty).

        Empty for every request without `tools`, and empty for every request
        at all under the default `--tool-profile graded` — so the graded
        defaults are unreachable-by-accident, which is the point.
        """
        if not tools:
            return {}
        return dict(TOOL_PROFILES.get(
            getattr(self, "tool_profile", "graded"), {}))

    def sampler(self, body: dict, *, tools: bool = False) -> dict:
        """Request overrides on top of the defaults in force for this request.

        Precedence, highest first: an explicit field in the request body, then
        the `--tool-profile` value when the request carries `tools`, then the
        `serve` flag (= the graded shipping config unless the operator changed
        it). A client that names a temperature always wins: the profile is a
        DEFAULT, never an override of a stated intent.
        """
        def _num(key, default, lo, hi):
            v = body.get(key)
            if v is None:
                return float(default)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise _BadRequest(f"'{key}' must be a number", key)
            v = float(v)
            if not (lo <= v <= hi):
                raise _BadRequest(f"'{key}' must be between {lo} and {hi}", key)
            return v

        # Called THROUGH THE CLASS on purpose. Three banked serve gate
        # harnesses build an Engine-shaped stub that copies individual real
        # methods
        # (`sampler = S.Engine.sampler`). Those gates must keep passing
        # UNMODIFIED against this file, so nothing new here may require an
        # attribute the stub does not define.
        Engine._warn_unsupported(self, body)
        prof = Engine.profile_for(self, tools)
        return dict(
            temperature=_num("temperature",
                             prof.get("temperature", self.args.temperature),
                             0.0, 2.0),
            top_p=_num("top_p", prof.get("top_p", self.args.top_p), 0.0, 1.0),
            # vLLM/TGI-style extension: 1.0 disables the penalty entirely.
            rep_penalty=_num("repetition_penalty",
                             prof.get("rep_penalty", self.args.rep_penalty),
                             0.01, 2.0),
            # not an OpenAI field, so it is profile/flag-driven only.
            pen_window=int(prof.get("pen_window", self.args.pen_window)),
        )

    def _warn_unsupported(self, body: dict) -> None:
        """Say so, once, when a client sets a knob we do not implement."""
        seen = getattr(self, "_warned_knobs", None)
        if seen is None:
            seen = self._warned_knobs = set()
        for key, why in UNSUPPORTED_SAMPLER_KNOBS.items():
            if key in seen:
                continue
            v = body.get(key)
            if v in (None, 0, 0.0, False, {}, []):
                continue
            seen.add(key)
            print(f"[{BRAND}] NOTE: '{key}' is IGNORED by this runtime — {why}. "
                  f"(said once per process; the sampler actually in force is on "
                  f"/health)", file=sys.stderr)

    def max_new(self, body: dict) -> int:
        """Requested completion budget, CLAMPED to the model context.

        0.1.6 returned the request value verbatim with no ceiling at all. See
        DEFAULT_CONTEXT above for why that is a denial of service you inflict on
        yourself. The clamp is applied twice: here against the whole context,
        and again in `stream()` against `context - prompt_tokens` once the
        prompt has actually been tokenised.
        """
        v = body.get("max_tokens")
        if v is None:
            v = body.get("max_completion_tokens")
        if v is None:
            return int(self.args.max_new)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise _BadRequest("'max_tokens' must be a positive integer",
                              "max_tokens")
        ceiling = int(getattr(self.args, "max_new_ceiling", 0) or 0)
        if ceiling > 0:
            v = self._clamp(v, ceiling, "--max-new-ceiling")
        return self._clamp(v, self.context - 1, "the model context")

    def _clamp(self, want: int, ceiling: int, why: str) -> int:
        ceiling = max(1, int(ceiling))
        if want <= ceiling:
            return want
        print(f"[{BRAND}] max_tokens {want} clamped to {ceiling} ({why})",
              file=sys.stderr)
        return ceiling

    def fit(self, max_new: int, n_prompt: int) -> int:
        """Second half of the clamp: never generate past the context window."""
        return self._clamp(max_new, self.context - n_prompt,
                           f"context {self.context} - {n_prompt} prompt tokens")

    # ------------------------------------------------------------ generation
    def stream(self, *, messages=None, prompt=None, max_new: int,
               temperature: float, top_p: float, rep_penalty: float,
               stop: list[str], result: dict, state: dict, tools=None,
               pen_window: int | None = None):
        """Yield decoded text pieces; fills `result` once fully consumed.

        `pen_window=None` keeps the flag value, which is what every caller did
        before 0.1.13; it is a parameter now only so a tool profile can
        widen the repetition window for tool-carrying requests.
        """
        if pen_window is None:
            pen_window = self.args.pen_window
        if self.kind == "tiny":
            text_in = prompt if prompt is not None else _last_user(messages)
            text, n_prompt, n_gen = _cli.tiny_reply(self.tiny_buf, text_in)
            result.update(prompt_tokens=n_prompt, completion_tokens=n_gen,
                          finish_reason="stop")
            pieces = iter([text] if text else [])
        else:
            if messages is not None:
                # tools=None/[] takes the 0.1.6 call shape unchanged; see
                # cli._chat_ids.
                ids = _cli._chat_ids(self.tokenizer, messages, tools=tools)
            else:
                ids = list(self.tokenizer.encode(prompt))
            max_new = self.fit(max_new, len(ids))
            pieces = _cli.iter_generate(
                self.model, self.tokenizer, ids, max_new=max_new,
                temperature=temperature, device=self.args.device,
                top_p=top_p, rep_penalty=rep_penalty,
                pen_window=pen_window, stop=stop or None,
                assistant_model=self.draft, result=result)
        if stop:
            pieces = _stop_filter(pieces, stop, state)
        return pieces


def _last_user(messages) -> str:
    for m in reversed(messages or []):
        if m["role"] == "user":
            return m["content"]
    return messages[-1]["content"] if messages else ""


# ------------------------------------------------------------------ handler

def _granularities(form) -> list[str]:
    """`timestamp_granularities` from a parsed form: the SDKs' `timestamp_granularities[]` (one part per value), a
    plain field, a comma-separated field, or a JSON list string; lower-cased, de-duplicated, in order."""
    raw: list = []
    for key in ("timestamp_granularities[]", "timestamp_granularities"):
        v = form.get(key)
        if isinstance(v, list):
            raw.extend(v)
        elif isinstance(v, str):
            t = v.strip()
            if t.startswith("["):
                try:
                    import json as _json
                    val = _json.loads(t)
                    raw.extend(val if isinstance(val, list) else [val])
                    continue
                except ValueError:
                    pass
            raw.extend(t.split(","))
    out: list[str] = []
    for g in raw:
        g = str(g).strip().strip('"').lower()
        if g and g not in out:
            out.append(g)
    return out

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"{BRAND}/{VERSION}"
    sys_version = ""

    # ------------------------------------------------------------- plumbing
    @property
    def engine(self) -> Engine:
        return self.server.engine                       # type: ignore[attr-defined]

    def log_message(self, fmt, *a):                     # concise, to stderr
        sys.stderr.write(f"[{BRAND}] {self.address_string()} {fmt % a}\n")

    def _cors(self) -> None:
        if getattr(self.engine.args, "cors", False):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods",
                             "GET, POST, OPTIONS")

    def _send_json(self, status: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, message: str, type_: str = "invalid_request_error",
             param=None, code=None) -> None:
        self._send_json(status, {"error": {"message": message, "type": type_,
                                           "param": param, "code": code}})

    def _sse_open(self) -> None:
        self._sse_started = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()

    def _sse(self, obj) -> None:
        self._raw(f"data: {json.dumps(obj)}\n\n")

    def _raw(self, text: str) -> None:
        b = text.encode()
        self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
        self.wfile.flush()

    def _sse_close(self) -> None:
        self._raw("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _authorised(self) -> bool:
        key = getattr(self.engine.args, "api_key", None)
        if not key:
            return True
        hdr = self.headers.get("Authorization", "")
        return hdr.startswith("Bearer ") and hmac.compare_digest(hdr[7:], key)

    def _drain_body(self) -> None:
        """Consume an unread request body before replying on an early-error
        path (unknown route, bad auth).

        HTTP/1.1 keep-alive DESYNC fix (0.1.8): these paths used to reply
        without reading the body, so the body's bytes were parsed as the next
        request line on the same connection. Measured live with an agent
        framework whose startup probe storm includes Ollama-style POSTs:
        `POST /api/show` got its 404, then its body surfaced as
        `code 400, message Bad request syntax ('{"name":"neutrino-8b_v4.bin"}')`
        — a garbage 400 to a well-behaved
        client, and one desynced framing away from misrouting a real request.
        Anything unreadable/oversized/chunked closes the connection instead:
        dropping a keep-alive is always safe, replying mid-stream never is.
        """
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True
            return
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked" \
                or n > MAX_BODY:
            self.close_connection = True
            return
        try:
            while n > 0:
                chunk = self.rfile.read(min(n, 1 << 16))
                if not chunk:
                    self.close_connection = True
                    return
                n -= len(chunk)
        except Exception:
            self.close_connection = True

    def _read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.close_connection = True   # unread stream: do not reuse it
            self._err(400, "chunked request bodies are not supported")
            return None
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._err(400, "invalid Content-Length")
            return None
        if n <= 0:
            self._err(400, "request body is required")
            return None
        if n > MAX_BODY:
            self._err(413, "request body too large", type_="invalid_request_error")
            return None
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as exc:
            self._err(400, f"invalid JSON body: {exc}")
            return None
        if not isinstance(body, dict):
            self._err(400, "request body must be a JSON object")
            return None
        return body

    # -------------------------------------------------------------- routing
    def do_OPTIONS(self):                                # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):                                    # noqa: N802
        route = urlsplit(self.path).path.rstrip("/") or "/"
        if route in ("/health", "/v1/health", "/healthz"):
            return self._send_json(200, self._health())
        if route == "/v1/audio/stream":
            # BEFORE the generic /v1 auth gate: a WebSocket client cannot
            # always set headers (browser WebSocket() has no header argument),
            # so this one route also accepts the key as `?api_key=` — checked
            # inside, with the same constant-time comparison.
            return self._audio_stream()
        if route.startswith("/v1") and not self._authorised():
            return self._err(401, "missing or invalid API key",
                             type_="invalid_request_error", code="invalid_api_key")
        if route == "/v1/models":
            return self._send_json(200, {"object": "list",
                                         "data": [self._model_card()]})
        if route.startswith("/v1/models/"):
            want = route[len("/v1/models/"):]
            if want == self.engine.model_id:
                return self._send_json(200, self._model_card())
            return self._err(404, f"model {want!r} not found",
                             type_="invalid_request_error", param="model",
                             code="model_not_found")
        if route == "/":
            eng = self.engine
            # The endpoint list is derived from the served model, not a
            # constant: this response is how a client discovers what exists,
            # so advertising chat endpoints on a speech server would be a lie
            # that costs the caller a round trip and a confusing 404.
            endpoints = (eng.endpoints() if getattr(eng, "kind", "") == "speech"
                         else ["/v1/chat/completions", "/v1/completions",
                               "/v1/models", "/health"])
            return self._send_json(200, {
                "service": f"{BRAND} serve", "version": VERSION,
                "model": eng.model_id,
                "model_kind": getattr(eng, "kind", "trtc"),
                "endpoints": endpoints})
        return self._err(404, f"unknown route {route!r}",
                         type_="invalid_request_error", code="not_found")

    def do_POST(self):                                   # noqa: N802
        route = urlsplit(self.path).path.rstrip("/") or "/"
        self._sse_started = False
        if not self._authorised():
            self._drain_body()
            return self._err(401, "missing or invalid API key",
                             type_="invalid_request_error", code="invalid_api_key")
        # THE SERVED MODEL DETERMINES WHICH ENDPOINTS ARE MOUNTED. A speech
        # server has no chat/completions and an LLM server has no
        # transcriptions — and each 404 names the endpoint that DOES exist
        # here, because "unknown route" on a route the client correctly
        # believes in is the least useful error we can return.
        speech = getattr(self.engine, "kind", "") == "speech"
        if speech:
            if route == "/v1/audio/transcriptions":
                return self._transcriptions()
            self._drain_body()
            return self._err(
                404, f"{route} is not served by this model. "
                     f"{self.engine.model_id} is a speech model; this server "
                     f"mounts POST /v1/audio/transcriptions. Start "
                     f"`{BRAND} serve neutrino-8b` for the chat endpoints.",
                type_="invalid_request_error", code="not_found")
        if route == "/v1/audio/transcriptions":
            self._drain_body()
            return self._err(
                404, f"{self.engine.model_id} is a language model, so this "
                     f"server does not transcribe audio. Run "
                     f"`{BRAND} serve phonon-2` for "
                     f"POST /v1/audio/transcriptions (this one was started as "
                     f"`{BRAND} serve neutrino-8b`-class).",
                type_="invalid_request_error", param="model",
                code="not_found")
        if route not in ("/v1/chat/completions", "/v1/completions"):
            self._drain_body()
            return self._err(404, f"unknown route {route!r}",
                             type_="invalid_request_error", code="not_found")
        body = self._read_body()
        if body is None:
            return
        try:
            if route == "/v1/chat/completions":
                self._chat(body)
            else:
                self._text(body)
        except _BadRequest as exc:
            self._fail(400, str(exc), "invalid_request_error", exc.param)
        except (BrokenPipeError, ConnectionResetError):  # client hung up
            self.close_connection = True
        except Exception as exc:                         # noqa: BLE001
            sys.stderr.write(f"[{BRAND}] internal error: {exc!r}\n")
            self._fail(500, f"internal error: {exc}", "server_error", None)

    # -------------------------------------------------- speech: /v1/audio/*
    def _transcriptions(self) -> None:
        """POST /v1/audio/transcriptions — the Whisper API shape.

        Multipart `file` (required) + `model` (accepted, cross-checked) +
        `response_format` (json | text | verbose_json). That is the subset of
        the OpenAI audio API that existing client libraries and every curl
        snippet on the internet actually send, so `client.audio.transcriptions
        .create(model=..., file=open(...,'rb'))` works unmodified against
        `base_url=http://127.0.0.1:8000/v1`.

        Body reading is done HERE rather than through `_read_body()` because
        that helper JSON-decodes; the framing rules it enforces (no chunked,
        Content-Length required, MAX_BODY cap, and `close_connection` on an
        unread stream so HTTP/1.1 keep-alive cannot desync) are reproduced
        exactly, because that desync was a real measured bug (0.1.8).

        Errors use the same `{"error": {...}}` envelope and the same
        `param`/`code` fields as every other route.
        """
        from fermion import _multipart

        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.close_connection = True
            return self._err(400, "chunked request bodies are not supported")
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._err(400, "invalid Content-Length")
        if n <= 0:
            return self._err(400, "an audio file is required",
                             param="file", code="missing_file")
        if n > MAX_BODY:
            self.close_connection = True
            return self._err(
                413, f"request body too large (limit {MAX_BODY // 2**20} MB). "
                     f"Split long recordings; the model is designed for "
                     f"utterances up to 30 s and decodes long audio in chunks.")
        raw = self.rfile.read(n)

        ctype = self.headers.get("Content-Type", "")
        try:
            form = _multipart.parse(raw, ctype)
        except _multipart.MultipartError as exc:
            return self._err(
                400, f"{exc}. /v1/audio/transcriptions takes a "
                     f"multipart/form-data body with a `file` part, exactly "
                     f"like the OpenAI audio API.")

        part = form.get("file")
        if not isinstance(part, dict) or not part.get("content"):
            return self._err(400, "no `file` part in the form",
                             param="file", code="missing_file")

        # `model` is accepted and checked, never silently ignored: a client
        # pointed at the wrong server should be told, not given a transcript
        # from a model it did not ask for.
        want = form.get("model")
        if isinstance(want, str) and want.strip():
            known = {self.engine.model_id, self.engine.repo}
            if want.strip() not in known \
                    and _catalog.resolve_alias(want.strip()) not in known:
                return self._err(
                    404, f"model {want.strip()!r} is not served here; this "
                         f"process serves {self.engine.model_id}",
                    param="model", code="model_not_found")

        fmt = (form.get("response_format") or "json")
        if not isinstance(fmt, str):
            fmt = "json"
        fmt = fmt.strip().lower() or "json"
        if fmt not in ("json", "text", "verbose_json"):
            return self._err(
                400, f"response_format {fmt!r} is not supported; use json "
                     f"(default), text, or verbose_json",
                param="response_format")

        # 0.2.9: hotwords. `hotwords` (comma-separated, a JSON list, or one
        # `hotwords[]` part per term) and, for OpenAI clients, `prompt` read
        # as a vocabulary list: names and terms the decoder favours for this
        # request. `hotwords` wins when both are sent. `hotword_lambda` sets
        # the bonus per matching piece (Phonon-2; default 2.0).
        try:
            hotwords, hotword_lambda = self._request_hotwords(form)
        except ValueError as exc:
            return self._err(400, str(exc), param="hotword_lambda")

        # Unimplemented OpenAI fields, told once rather than dropped in
        # silence — the same posture as UNSUPPORTED_SAMPLER_KNOBS.
        for field, why in (
                ("language", "the model is English-only; `language` is ignored"),
                ("temperature", "transcription is deterministic (temperature 0)")):
            if field in form and field not in self.engine._warned_fields:
                self.engine._warned_fields.add(field)
                sys.stderr.write(f"[{BRAND}] note: `{field}` — {why}\n")
        # 0.2.7: `timestamp_granularities` (the OpenAI field; the SDKs send it as `timestamp_granularities[]`, curl
        # users as a plain or comma-separated string) selects `words` and `segments` in verbose_json. `word` and
        # `segment` are honoured; anything else is noted once and ignored.
        granularities = _granularities(form)
        for g in granularities:
            if g not in ("word", "segment") and ("timestamp_granularities:" + g) not in self.engine._warned_fields:
                self.engine._warned_fields.add("timestamp_granularities:" + g)
                sys.stderr.write(f"[{BRAND}] note: `timestamp_granularities` value {g!r} is not supported; "
                                 f"use word and/or segment\n")
        want_words = "word" in granularities

        try:
            result = self.engine.transcribe_bytes(
                part["content"], part.get("filename") or "upload",
                hotwords=hotwords, hotword_lambda=hotword_lambda)
        except ValueError as exc:                # undecodable audio
            self.engine.stats["errors"] += 1
            return self._err(400, str(exc), param="file",
                             code="invalid_audio")
        text, decode_s, audio_s = result.triple()

        if fmt == "text":
            body = (text + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            return self.wfile.write(body)
        out = {"text": text}
        if fmt == "verbose_json":
            out.update({
                "task": "transcribe", "language": "english",
                "duration": round(audio_s, 3),
                # One entry per decoded window (`start`/`end` in seconds).
                # A file up to 35 s is one window; longer files are cut at
                # pauses into 25-35 s windows (fermion/_speech/segment.py).
                "segments": [{"id": s["id"], "start": s["start"],
                              "end": s["end"], "text": s["text"]}
                             for s in result.segments],
            })
            if want_words:
                if result.timed:
                    # OpenAI's shape: `word`, `start`, `end` (seconds from the start of the file).
                    out["words"] = [{"word": w["text"], "start": w["start"], "end": w["end"]}
                                    for w in result.words]
                elif "timestamp_granularities:unavailable" not in self.engine._warned_fields:
                    self.engine._warned_fields.add("timestamp_granularities:unavailable")
                    sys.stderr.write(f"[{BRAND}] note: word timestamps are not available from "
                                     f"{self.engine.model_id}; `words` is omitted\n")
            out.update({
                # Not in OpenAI's schema, and deliberately here anyway: the
                # decode configuration these words came out of. Same reason
                # `serve` prints its `--tool-profile` — a number must never be
                # attributable to a configuration that did not produce it.
                "x_fermion": {"model": self.engine.repo,
                              "profile": self.engine.profile,
                              "decode_seconds": round(decode_s, 3),
                              "segment_count": len(result.segments),
                              "truncated": bool(result.truncated),
                              **self.engine.speech.describe(),
                              # this request's hotwords (describe() is the
                              # process-wide setting, empty under `serve`)
                              "hotwords": list(hotwords or []),
                              "hotword_lambda": (hotword_lambda if hotword_lambda is not None
                                                 else getattr(self.engine.speech, "hotword_lambda", None))
                              if hotwords else None},
            })
        return self._send_json(200, out)

    def _request_hotwords(self, form):
        """(hotwords or None, hotword_lambda or None) from a transcription form; ValueError on a bad lambda."""
        from fermion._speech.hotwords import MAX_WORDS, parse_list
        words: list[str] = []
        for key in ("hotwords[]", "hotwords"):
            v = form.get(key)
            vals = v if isinstance(v, list) else ([v] if isinstance(v, str) else [])
            for x in vals:
                t = str(x).strip()
                if t.startswith("["):
                    try:
                        import json as _json
                        val = _json.loads(t)
                        for w in (val if isinstance(val, list) else [val]):
                            w = " ".join(str(w).split())
                            if w and w not in words:
                                words.append(w)
                        continue
                    except ValueError:
                        pass
                for w in (parse_list(t) if key == "hotwords" else [" ".join(t.split())]):
                    if w and w not in words:
                        words.append(w)
        if not words:
            p = form.get("prompt")
            if isinstance(p, str) and p.strip():
                words = parse_list(p)
        lam = form.get("hotword_lambda")
        if isinstance(lam, str) and lam.strip():
            try:
                lam = float(lam)
            except ValueError:
                raise ValueError(f"hotword_lambda {lam.strip()!r} is not a number")
            if not (0 <= lam <= 100):
                raise ValueError("hotword_lambda must be between 0 and 100")
        else:
            lam = None
        if len(words) > MAX_WORDS:
            if "hotwords:cap" not in self.engine._warned_fields:
                self.engine._warned_fields.add("hotwords:cap")
                sys.stderr.write(f"[{BRAND}] note: a request sent {len(words)} hotwords; the first {MAX_WORDS} are "
                                 f"used (said once)\n")
            words = words[:MAX_WORDS]
        if words and not hasattr(self.engine.speech, "_with_hotwords"):
            if "hotwords:unavailable" not in self.engine._warned_fields:
                self.engine._warned_fields.add("hotwords:unavailable")
                sys.stderr.write(f"[{BRAND}] note: hotwords are not available for {self.engine.model_id} on this "
                                 f"engine; requests decode without them\n")
            return None, None
        return (words or None), (lam if words else None)

    # ------------------------------------------- speech: WS /v1/audio/stream
    def _ws_authorised(self) -> bool:
        """The `--api-key` rule, WebSocket edition: header OR `?api_key=`.

        The bearer header stays the primary form (identical to every HTTP
        route). The query parameter exists because the browser `WebSocket()`
        constructor cannot set headers at all, and refusing browsers would
        make the endpoint useless to the web dictation demos it exists for.
        Both comparisons are constant-time.
        """
        key = getattr(self.engine.args, "api_key", None)
        if not key:
            return True
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Bearer ") and hmac.compare_digest(hdr[7:], key):
            return True
        q = parse_qs(urlsplit(self.path).query).get("api_key", [])
        return bool(q) and hmac.compare_digest(q[-1], key)

    def _audio_stream(self) -> None:
        """GET /v1/audio/stream — live transcription over a WebSocket.

        THE TRANSPORT IS THIN ON PURPOSE. Everything that makes the stream
        live — rolling partials, the pause-detecting segmenter, the 30 s cap,
        the decode cadence — is `fermion._speech.live.LiveSession`, the exact
        object behind `fermion listen`. This handler only turns WebSocket
        frames into `session.feed()` calls and callbacks into JSON frames, so
        the wire endpoint and the local command cannot drift apart. The decode
        configuration is `transcribe`'s, untouched.

        Protocol (client -> server): one JSON text frame
        `{"sample_rate": 16000, "format": "pcm_f32le"|"pcm_s16le"}` (both
        fields optional; f32 is the default), then binary frames of raw mono
        audio, then `{"type":"end"}` — or a clean close — to finish.
        Server -> client: `{"type":"partial","text":...}` while a phrase is
        in flight, `{"type":"final","text":...,"segment":N}` (N counts from 1)
        when a pause or the 30 s cap closes a segment, `{"type":"done",
        "text":<full transcript>}` at the end, and `{"type":"error",
        "message":...}` followed by a close on any refusal.

        MOUNTED ONLY FOR SPEECH MODELS, exactly like /v1/audio/transcriptions:
        the served model determines which endpoints exist, and the 404 an LLM
        server returns here names the command that does work.
        """
        eng = self.engine
        if not self._ws_authorised():
            return self._err(401, "missing or invalid API key",
                             type_="invalid_request_error",
                             code="invalid_api_key")
        if getattr(eng, "kind", "") != "speech":
            return self._err(
                404, f"{eng.model_id} is a language model, so this server "
                     f"does not stream transcription. Run "
                     f"`{BRAND} serve --model phonon` for "
                     f"GET /v1/audio/stream (WebSocket).",
                type_="invalid_request_error", param="model", code="not_found")
        wkey = self.headers.get("Sec-WebSocket-Key", "")
        upgrade = (self.headers.get("Upgrade") or "").lower()
        connection = (self.headers.get("Connection") or "").lower()
        if upgrade != "websocket" or "upgrade" not in connection or not wkey:
            return self._err(
                426, "/v1/audio/stream is a WebSocket endpoint: send an "
                     "upgrade handshake (Upgrade: websocket + "
                     "Sec-WebSocket-Key). For one-shot files use "
                     "POST /v1/audio/transcriptions.",
                type_="invalid_request_error", code="upgrade_required")
        if self.headers.get("Sec-WebSocket-Version", "").strip() != "13":
            # RFC 6455 §4.2.2: name the version we do speak.
            self.send_response(426)
            self.send_header("Sec-WebSocket-Version", "13")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # A WebSocket connection never returns to HTTP keep-alive.
        self.close_connection = True
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", _ws.accept_key(wkey))
        self.end_headers()
        self.wfile.flush()
        try:
            self._ws_session(eng)
        except (BrokenPipeError, ConnectionResetError, _ws.ConnectionClosed):
            pass                                     # client hung up mid-frame
        except _ws.ProtocolError as exc:
            self._ws_refuse(f"websocket protocol error: {exc}", code=1002)
        except Exception as exc:                             # noqa: BLE001
            sys.stderr.write(f"[{BRAND}] stream internal error: {exc!r}\n")
            self._ws_refuse(f"internal error: {exc}", code=1011)

    def _ws_send(self, obj: dict) -> None:
        self.wfile.write(_ws.encode_frame(_ws.TEXT, json.dumps(obj).encode()))
        self.wfile.flush()

    def _ws_bye(self, code: int = 1000, reason: str = "") -> None:
        """Send our close frame; best-effort, because the peer may be gone."""
        try:
            self.wfile.write(_ws.encode_frame(
                _ws.CLOSE, _ws.close_payload(code, reason)))
            self.wfile.flush()
        except Exception:                                    # noqa: BLE001
            pass

    def _ws_refuse(self, message: str, code: int = 1008) -> None:
        """The error contract: `{"type":"error",...}` then a clean close."""
        try:
            self._ws_send({"type": "error", "message": message})
        except Exception:                                    # noqa: BLE001
            pass
        self._ws_bye(code, "")

    def _ws_next(self):
        """The next DATA message: (opcode, payload), answering pings inline.

        Returns `(None, close_payload)` for a close frame. Fragmented
        messages (FIN=0 or a continuation opcode) are refused via
        `_WSRefusal` — see fermion/_ws.py for why v1 does not reassemble.
        """
        while True:
            fin, op, payload = _ws.read_frame(self.rfile, require_mask=True)
            if op == _ws.PING:
                self.wfile.write(_ws.encode_frame(_ws.PONG, payload))
                self.wfile.flush()
                continue
            if op == _ws.PONG:
                continue
            if op == _ws.CLOSE:
                return None, payload
            if not fin or op == _ws.CONTINUATION:
                raise _WSRefusal("fragmented messages are not supported: "
                                 "send each message in a single frame")
            if op not in (_ws.TEXT, _ws.BINARY):
                raise _WSRefusal(f"unsupported opcode {op:#x}")
            return op, payload

    def _ws_session(self, eng) -> None:
        """Acquire the one-stream slot, run the stream, always release."""
        if not eng.stream_guard.acquire(blocking=False):
            # A second concurrent stream: refused cleanly and immediately —
            # never queued behind a session that may run for minutes. 1013 is
            # "try again later". Decodes themselves already serialise on the
            # engine's single MLX worker; this guard exists so a second
            # client is TOLD, instead of silently starving the first.
            return self._ws_refuse("engine busy — one stream at a time",
                                   code=1013)
        try:
            self.connection.settimeout(STREAM_IDLE_S)
            try:
                self._ws_stream(eng)
            except TimeoutError:
                self._ws_refuse(f"stream idle for over {STREAM_IDLE_S:.0f}s "
                                f"— closing", code=1001)
        finally:
            eng.stream_guard.release()

    def _ws_stream(self, eng) -> None:
        import numpy as np

        from fermion._speech import live
        from fermion._speech.engine import SAMPLE_RATE

        try:
            self._ws_run(eng, np, live, SAMPLE_RATE)
        except _WSRefusal as exc:
            self._ws_refuse(str(exc))

    def _ws_run(self, eng, np, live, sample_rate: int) -> None:
        # ---- 1. the config frame (or an immediate end / close)
        op, payload = self._ws_next()
        if op is None:
            return self._ws_bye()
        if op != _ws.TEXT:
            raise _WSRefusal(
                'the first message must be a JSON text frame, e.g. '
                '{"sample_rate": 16000, "format": "pcm_f32le"}')
        try:
            cfg = json.loads(payload.decode("utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError("not an object")
        except Exception as exc:                             # noqa: BLE001
            raise _WSRefusal(f"invalid JSON config frame: {exc}")
        if cfg.get("type") == "end":                # ended before any audio
            self._ws_send({"type": "done", "text": ""})
            return self._ws_bye()
        fmt = cfg.get("format", "pcm_f32le")
        if fmt not in ("pcm_f32le", "pcm_s16le"):
            raise _WSRefusal(f"format {fmt!r} is not supported: send "
                             f"pcm_f32le (default) or pcm_s16le")
        sr = cfg.get("sample_rate", sample_rate)
        if sr != sample_rate:
            # Streaming resample would need stateful filters to avoid chunk-
            # boundary artifacts; v1 keeps the model's native rate and says so.
            raise _WSRefusal(f"sample_rate must be {sample_rate} (got {sr!r})"
                             f" — resample client-side")

        # ---- 2. the session. First decode on a fresh process pays the Metal
        # graph compile (~seconds); warm once so partials land on cadence,
        # exactly as `fermion listen` does at spin-up.
        eng.warm()
        state = {"segment": 0}

        def on_partial(text: str) -> None:
            self._ws_send({"type": "partial", "text": text})

        def on_final(text: str) -> None:
            state["segment"] += 1
            self._ws_send({"type": "final", "text": text,
                           "segment": state["segment"]})

        session = live.LiveSession(eng.pooled(), on_partial=on_partial,
                                   on_final=on_final)
        bytes_per = 4 if fmt == "pcm_f32le" else 2
        pending = b""                    # sub-sample tail across frame splits
        fed_samples = 0
        while True:
            op, payload = self._ws_next()
            if op is None:               # clean close counts as end (spec'd)
                break
            if op == _ws.TEXT:
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception as exc:                     # noqa: BLE001
                    raise _WSRefusal(f"invalid JSON text frame: {exc}")
                if isinstance(msg, dict) and msg.get("type") == "end":
                    break
                raise _WSRefusal('unexpected text frame mid-stream: only '
                                 '{"type":"end"} is defined')
            pending += payload
            cut = len(pending) - (len(pending) % bytes_per)
            if not cut:
                continue
            chunk, pending = pending[:cut], pending[cut:]
            if fmt == "pcm_f32le":
                block = np.frombuffer(chunk, dtype="<f4")
            else:
                block = (np.frombuffer(chunk, dtype="<i2")
                         .astype(np.float32) / 32768.0)
            fed_samples += block.size
            session.feed(block)

        # ---- 3. finish: decode whatever is in flight, hand back the whole
        # transcript (finals joined with single spaces — `transcribe`'s shape),
        # and close cleanly.
        text = session.finish()
        eng.note_stream(fed_samples / sample_rate)
        self._ws_send({"type": "done", "text": text})
        self._ws_bye()

    def _fail(self, status: int, message: str, type_: str, param) -> None:
        """Error out whether or not the SSE stream has already started."""
        payload = {"error": {"message": message, "type": type_,
                             "param": param, "code": None}}
        try:
            if getattr(self, "_sse_started", False):
                self._sse(payload)                       # in-band SSE error
                self._sse_close()
            else:
                self._send_json(status, payload)
        except Exception:                                # pragma: no cover
            self.close_connection = True

    # --------------------------------------------------------------- bodies
    def _model_card(self) -> dict:
        return {"id": self.engine.model_id, "object": "model",
                "created": self.engine.created, "owned_by": OWNER}

    def _health(self) -> dict:
        eng = self.engine
        if getattr(eng, "kind", "") == "speech":
            # A speech server has no container, no sampler, no tools and no KV
            # session, so it answers with its own block rather than filling
            # this one with Nones that a monitoring script would misread.
            return eng.health()
        out = {"status": "ok", "model": eng.model_id, "kind": eng.kind,
               "sha256_short": eng.sha256[:8], "sha256": eng.sha256,
               "container": str(eng.container), "bytes": eng.container.stat().st_size,
               "version": VERSION, "device": eng.args.device,
               "dtype": eng.args.dtype, "draft": bool(eng.draft),
               # 0.1.7: agent frameworks probe for this before they will drive
               # a custom endpoint at all.
               "tool_calling": True, "context": eng.context,
               # which decode path is actually serving these tokens. The
               # published tok/s numbers describe "native" only.
               "backend": getattr(eng.model, "backend", "torch")
                          if eng.kind != "tiny" else "tiny-reference",
               "native": (eng.model.describe()
                          if getattr(eng.model, "backend", "") == "native"
                          else None),
               "sampler_defaults": {
                   "temperature": eng.args.temperature,
                   "top_p": eng.args.top_p,
                   "repetition_penalty": eng.args.rep_penalty,
                   "pen_window": eng.args.pen_window,
                   "max_tokens": eng.args.max_new},
               # tool profiles: what a request CARRYING TOOLS is sampled at,
               # which is not
               # necessarily `sampler_defaults`. `graded` (the default) means
               # the two are identical and the banked config is in force.
               # (`getattr` throughout: the banked serve harnesses stub the
               # engine and predate these attributes.)
               "tool_sampling": {
                   "profile": getattr(eng, "tool_profile", "graded"),
                   "applies_to": "requests that carry `tools`",
                   "overrides": Engine.profile_for(eng, True),
                   "graded_config": {
                       "temperature": getattr(_cli, "SHIP_TEMPERATURE",
                                              eng.args.temperature),
                       "top_p": getattr(_cli, "SHIP_TOP_P", eng.args.top_p),
                       "repetition_penalty": getattr(_cli, "SHIP_REP_PENALTY",
                                                     eng.args.rep_penalty),
                       "pen_window": getattr(_cli, "SHIP_PEN_WINDOW",
                                             eng.args.pen_window)},
                   "is_graded": getattr(eng, "tool_profile",
                                        "graded") == "graded",
                   "unsupported_fields": sorted(UNSUPPORTED_SAMPLER_KNOBS)}}
        # stuck detector: proof the loop breaker exists, its thresholds, and
        # how often it has fired in this process.
        det = getattr(eng, "stuck", None)
        out["stuck_detector"] = det.health() if det is not None else None
        if eng.header is not None:
            out["container_header"] = {
                "trtc_version": eng.header.version, "arch": eng.header.arch,
                "layers": eng.header.n_layers, "hidden": eng.header.hidden,
                "vocab": eng.header.vocab}
        # 0.1.8: session state + reuse stats as a first-class /health block,
        # so "is the KV cache actually engaging?" is one curl away instead of
        # a serve-log archaeology session. `last_request` carries the reuse
        # ratio of the most recent request; `fallbacks`/`last_fallback` say
        # how often and why requests ran one-shot (full prefill).
        if out["native"] is not None:
            out["session"] = {
                "active": out["native"].get("session", False),
                "ctx": out["native"].get("session_ctx"),
                "note": out["native"].get("session_note"),
                **out["native"].get("session_stats", {})}
        else:
            out["session"] = None
        return out

    def _common(self, body: dict, *, tools: bool = False) -> tuple:
        eng = self.engine
        smp = eng.sampler(body, tools=tools)
        # `pen_window` rides in the sampler dict (and therefore into
        # `Engine.stream`) ONLY when a tool profile actually moves it. At
        # the default it is dropped, so `stream(**smp)` keeps the exact
        # pre-0.1.13 keyword set — the banked serve harnesses stub `stream`
        # with that signature and must keep passing unmodified.
        if smp.get("pen_window") == getattr(eng.args, "pen_window", None):
            smp.pop("pen_window", None)
        max_new = eng.max_new(body)
        try:
            stop = _norm_stop(body.get("stop"))
        except ValueError as exc:
            raise _BadRequest(str(exc), "stop")
        n = body.get("n")
        if n not in (None, 1):
            raise _BadRequest("only n=1 is supported by this runtime", "n")
        if body.get("stream") not in (None, True, False):
            raise _BadRequest("'stream' must be a boolean", "stream")
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return smp, max_new, stop, bool(body.get("stream")), want_usage

    # ------------------------------------------------------- chat completion
    def _chat(self, body: dict) -> None:
        eng = self.engine
        raw = body.get("messages")
        if not isinstance(raw, list) or not raw:
            raise _BadRequest("'messages' is required and must be a non-empty "
                              "array", "messages")
        try:
            messages = [_norm_message(m) for m in raw]
        except ValueError as exc:
            raise _BadRequest(str(exc), "messages")
        tools, _choice = _norm_tools(body)
        smp, max_new, stop, stream, want_usage = self._common(
            body, tools=bool(tools))

        cid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        result: dict = {}
        state: dict = {}

        def make(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": eng.model_id,
                    "choices": [{"index": 0, "delta": delta,
                                 "logprobs": None, "finish_reason": finish}]}

        # Stuck detection: the loop is in the request body (stateless server: the
        # client re-sends the whole conversation every step). Analysis is a
        # pure function of `messages`; see fermion/_stuck.py.
        trip = None
        # `Engine.__init__` always builds the detector; `getattr` is for the
        # Engine-shaped stubs in the banked serve harnesses, which predate it.
        det = getattr(eng, "stuck", None)
        if det is not None and det.enabled and (tools or any(
                isinstance(m, dict) and m.get("tool_calls") for m in messages)):
            trip = det.check(messages, tools or body.get("tools"))
        if trip is not None and det.mode == "enforce":
            return self._chat_stuck(
                messages, trip, cid=cid, created=created, smp=smp,
                max_new=max_new, stop=stop, stream=stream,
                want_usage=want_usage, make=make, choice=_choice)

        with eng.lock:
            pieces = eng.stream(messages=messages, max_new=max_new, tools=tools,
                                stop=stop, result=result, state=state, **smp)
            if tools:
                # A tool call can only be RECOGNISED once its whole
                # `<tool_call>{…}</tool_call>` envelope has arrived, so a call
                # cannot be emitted incrementally. The NON-streaming path
                # therefore buffers the whole completion and parses it (0.1.7
                # behaviour, unchanged).
                #
                # 0.1.9: the STREAMING path no longer buffers the prose. It
                # streams the assistant's prose as normal content deltas and
                # only switches to buffering the instant a `<tool_call` marker
                # appears (see `_chat_with_tools_stream`); the assembled result
                # is byte-identical to the buffered path for every shape (pure
                # prose, pure call, prose-then-call, parallel calls). This does
                # NOT cut time-to-first-token — that wait is prefill, before any
                # token exists — it only makes the generation phase visible.
                if not stream:
                    return self._chat_with_tools(
                        "".join(pieces), cid=cid, created=created,
                        result=result, state=state, stream=False,
                        want_usage=want_usage, make=make, tools=tools)
                return self._chat_with_tools_stream(
                    pieces, cid=cid, created=created, result=result,
                    state=state, want_usage=want_usage, make=make, tools=tools)
            if not stream:
                text = "".join(pieces)
                usage = self._usage(result)
                return self._send_json(200, {
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": eng.model_id,
                    "choices": [{"index": 0, "logprobs": None,
                                 "message": {"role": "assistant",
                                             "content": text},
                                 "finish_reason": self._finish(result, state)}],
                    "usage": usage})
            self._sse_open()
            self._sse(make({"role": "assistant", "content": ""}))
            for piece in pieces:
                if piece:
                    self._sse(make({"content": piece}))
            self._sse(make({}, self._finish(result, state)))
            if want_usage:
                self._sse({"id": cid, "object": "chat.completion.chunk",
                           "created": created, "model": eng.model_id,
                           "choices": [], "usage": self._usage(result)})
            self._sse_close()

    # ---------------------------------------------------------- loop breaking
    def _chat_stuck(self, messages, trip, *, cid: str, created: int, smp: dict,
                    max_new: int, stop, stream: bool, want_usage: bool, make,
                    choice=None) -> None:
        """The tripped turn: ONE no-tools diagnosis, then a forced answer.

        Why two generations inside one request rather than two request/response
        rounds: the client owns the loop, and a client whose model has just
        emitted a third identical call will happily execute a fourth. The only
        place that can guarantee the loop ENDS is here, so the turn the client
        gets back is a finished answer (`finish_reason: "stop"`, no
        `tool_calls`) and its loop has nothing left to execute.

        Neither generation renders `tools`, which also disposes of the
        tool_choice-stickiness loop class: a `tool_choice: "required"`
        that survives into a stuck loop cannot force a fourth call here.
        """
        eng = self.engine
        det = eng.stuck
        if choice not in (None, "auto", "none"):
            print(f"[{BRAND}] stuck-detector: tool_choice {choice!r} is not "
                  f"honoured on the loop-break turns — tools are disabled for "
                  f"both of them (a forced tool_choice must not outlive a "
                  f"stuck loop)", file=sys.stderr)

        diag_msgs = list(messages) + [
            {"role": "user",
             "content": _stuck.DIAGNOSIS_INSTRUCTION.format(detail=trip.detail)}]
        r1: dict = {}
        r2: dict = {}
        s1: dict = {}
        s2: dict = {}

        def usage() -> dict:
            return {"prompt_tokens": int(r1.get("prompt_tokens", 0))
                    + int(r2.get("prompt_tokens", 0)),
                    "completion_tokens": int(r1.get("completion_tokens", 0))
                    + int(r2.get("completion_tokens", 0)),
                    "total_tokens": int(r1.get("prompt_tokens", 0))
                    + int(r1.get("completion_tokens", 0))
                    + int(r2.get("prompt_tokens", 0))
                    + int(r2.get("completion_tokens", 0))}

        # One lock for both phases: they are two halves of one turn, and the
        # native session's KV prefix reuse wants them adjacent.
        with eng.lock:
            diag_raw = "".join(eng.stream(
                messages=diag_msgs, max_new=min(max_new, _stuck.DIAG_MAX_NEW),
                tools=None, stop=stop, result=r1, state=s1, **smp))
            # The model may still emit `<tool_call>` XML out of habit. Nothing
            # will execute it, and it must never reach the user as prose, so it
            # goes through the same tolerant parser and is dropped.
            diag, _dropped = self._parse_and_alias(diag_raw, None)
            diag = diag.strip()
            det.note_diagnosis(diag)
            if _dropped:
                print(f"[{BRAND}] stuck-detector: dropped "
                      f"{len(_dropped)} tool call(s) emitted during the "
                      f"tools-disabled diagnosis turn", file=sys.stderr)

            final_msgs = list(messages) + [
                {"role": "assistant",
                 "content": diag or "(the diagnosis turn produced no text)"},
                {"role": "user", "content": _stuck.FINAL_INSTRUCTION}]
            pieces = eng.stream(messages=final_msgs, max_new=max_new,
                                tools=None, stop=stop, result=r2, state=s2,
                                **smp)

            annot = {"tripped": True, "action": "diagnosis+forced_answer",
                     "diagnosis": diag, **trip.as_dict()}
            if not stream:
                text = "".join(pieces)
                det.note_forced_answer()
                return self._send_json(200, {
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": eng.model_id,
                    "choices": [{"index": 0, "logprobs": None,
                                 "message": {"role": "assistant",
                                             "content": text},
                                 "finish_reason": self._finish(r2, s2)}],
                    "usage": usage(),
                    # Non-standard, present ONLY on a tripped turn: an agent
                    # framework should be able to see the runtime broke its
                    # loop without tailing our stderr. OpenAI clients ignore
                    # unknown top-level fields.
                    "fermion_stuck": annot})
            # Nothing to buffer (no tools are rendered), so the forced answer
            # streams live like any prose turn.
            self._sse_open()
            self._sse(make({"role": "assistant", "content": ""}))
            for piece in pieces:
                if piece:
                    self._sse(make({"content": piece}))
            self._sse(make({}, self._finish(r2, s2)))
            det.note_forced_answer()
            tail = {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": eng.model_id,
                    "choices": [], "fermion_stuck": annot}
            if want_usage:
                tail["usage"] = usage()
            self._sse(tail)
            self._sse_close()

    def _parse_and_alias(self, text: str, tools) -> tuple[str, list[dict]]:
        """`(clean, calls)` from raw model text, with the loud logging and the
        name-alias rewrite both the buffered and the streamed tools paths
        share. THE ONE place tool-call text becomes OpenAI calls in serve, so
        the two paths cannot drift.
        """
        clean, calls, anomalies = _toolcalls.parse_tool_calls(text)
        if anomalies:
            # Loud on purpose: this is the malformed-envelope weakness, and the
            # only way anyone learns how often it fires is if we say so.
            sys.stderr.write(
                f"[{BRAND}] recovered {len(calls)} tool call(s) from a "
                f"malformed envelope (a strict parser would have failed): "
                f"{'; '.join(anomalies)}\n")
        if calls and tools:
            # 0.1.8: name-binding fix. Under namespaced schemas (MCP-style
            # frameworks name every tool `mcp__<server>__<tool>`) the model
            # emits the bare memorized name and never self-corrects — measured
            # 0/24 tool executions, vs 100% first-try with bare names.
            # A unique namespace-suffix
            # match is rewritten to the client's registered name; ambiguous
            # or unknown names pass through untouched.
            names = [t["function"]["name"] for t in tools
                     if isinstance(t, dict) and isinstance(t.get("function"),
                                                           dict)]
            for note in _toolcalls.alias_tool_names(calls, names):
                sys.stderr.write(f"[{BRAND}] {note}\n")
        return clean, calls

    def _chat_with_tools(self, text: str, *, cid: str, created: int,
                         result: dict, state: dict, stream: bool,
                         want_usage: bool, make, tools=None) -> None:
        """Turn a buffered completion into OpenAI `tool_calls`."""
        eng = self.engine
        clean, calls = self._parse_and_alias(text, tools)
        message: dict = {"role": "assistant",
                         "content": (clean or None) if calls else clean}
        if calls:
            message["tool_calls"] = calls
        finish = "tool_calls" if calls else self._finish(result, state)

        if not stream:
            return self._send_json(200, {
                "id": cid, "object": "chat.completion", "created": created,
                "model": eng.model_id,
                "choices": [{"index": 0, "logprobs": None, "message": message,
                             "finish_reason": finish}],
                "usage": self._usage(result)})
        self._sse_open()
        self._sse(make({"role": "assistant", "content": ""}))
        if clean:
            self._sse(make({"content": clean}))
        if calls:
            self._sse(make({"tool_calls": [
                {"index": i, "id": c["id"], "type": "function",
                 "function": dict(c["function"])} for i, c in enumerate(calls)]}))
        self._sse(make({}, finish))
        if want_usage:
            self._sse({"id": cid, "object": "chat.completion.chunk",
                       "created": created, "model": eng.model_id,
                       "choices": [], "usage": self._usage(result)})
        self._sse_close()

    def _chat_with_tools_stream(self, pieces, *, cid: str, created: int,
                                result: dict, state: dict, want_usage: bool,
                                make, tools=None) -> None:
        """Stream prose live, buffering only from the first `<tool_call` on.

        The reconciliation invariant (proven across 1,529 chunk-boundary
        splits): the concatenation of every `content` delta this
        method emits, plus the `tool_calls` delta, is BYTE-IDENTICAL to what
        the buffered `_chat_with_tools` produces for the same completion, for
        all four shapes. It holds because:

          * `emitted` only ever grows by CONTIGUOUS `raw` bytes taken strictly
            before the first `<` (any `<` might begin a `<tool_call>`/
            `</tool_call>` tag the parser removes), with the trailing
            whitespace run held back (the parser strips it once a call is
            present) and with NO leading whitespace ever streamed (pure prose
            keeps it, a call strips it — the two disagree at byte 0, so a turn
            whose very first byte is whitespace is buffered whole). Hence
            `emitted` is always a prefix of the authoritative `clean`.
          * At end-of-generation the FULL buffered text is handed to the SAME
            `_parse_and_alias` the buffered path uses, and the untold remainder
            `clean[len(emitted):]` is emitted as one final content delta. So
            the tolerant parser and the alias layer fire identically here.

        A `<tool_call` split across chunk boundaries can never leak as prose:
        emission stops at the first `<`, so a partial `<too` / `<tool_ca` tail
        is held until the next chunk resolves it.
        """
        marker = _toolcalls.TOOL_CALL_MARKER          # "<tool_call" (one source)
        raw = ""
        emitted = ""
        frozen = False        # stop live-streaming: a '<' (maybe a marker) is open

        self._sse_open()
        self._sse(make({"role": "assistant", "content": ""}))
        for piece in pieces:
            if not piece:
                continue
            raw += piece
            if frozen:
                continue                       # draining into the buffer only
            if not emitted and raw[:1].isspace():
                # leading whitespace: kept by pure-prose parse, stripped once a
                # call appears — undecidable until the turn ends, so buffer it
                # all and let the end-of-turn remainder deliver it (rare).
                frozen = True
                continue
            lt = raw.find("<")                 # first byte that might open a tag
            limit = lt if lt >= 0 else len(raw)
            # rstrip holds the trailing whitespace run (the parser drops it if a
            # call follows); interior whitespace flushes once real prose follows.
            chunk = raw[len(emitted):limit].rstrip()
            if chunk:
                self._sse(make({"content": chunk}))
                emitted += chunk
            if lt >= 0:
                # From the first '<' on we cannot know whether it becomes a
                # `<tool_call` marker without risking a wrong split, so freeze
                # live output and reconcile everything at end-of-turn.
                frozen = True

        clean, calls = self._parse_and_alias(raw, tools)
        # `emitted` is a prefix of `clean` by construction; the guard keeps
        # output whole even if some unforeseen shape violated that.
        remainder = clean[len(emitted):] if clean.startswith(emitted) else clean
        finish = "tool_calls" if calls else self._finish(result, state)

        if remainder:
            self._sse(make({"content": remainder}))
        if calls:
            self._sse(make({"tool_calls": [
                {"index": i, "id": c["id"], "type": "function",
                 "function": dict(c["function"])} for i, c in enumerate(calls)]}))
        self._sse(make({}, finish))
        if want_usage:
            self._sse({"id": cid, "object": "chat.completion.chunk",
                       "created": created, "model": self.engine.model_id,
                       "choices": [], "usage": self._usage(result)})
        self._sse_close()

    # ------------------------------------------------------- text completion
    def _text(self, body: dict) -> None:
        eng = self.engine
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            if len(prompt) != 1 or not isinstance(prompt[0], str):
                raise _BadRequest("only a single string prompt is supported",
                                  "prompt")
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt:
            raise _BadRequest("'prompt' is required and must be a non-empty "
                              "string", "prompt")
        smp, max_new, stop, stream, want_usage = self._common(body)

        cid = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        result: dict = {}
        state: dict = {}

        def make(text, finish=None):
            return {"id": cid, "object": "text_completion", "created": created,
                    "model": eng.model_id,
                    "choices": [{"index": 0, "text": text, "logprobs": None,
                                 "finish_reason": finish}]}

        with eng.lock:
            pieces = eng.stream(prompt=prompt, max_new=max_new, stop=stop,
                                result=result, state=state, **smp)
            if not stream:
                text = "".join(pieces)
                out = make(text, self._finish(result, state))
                out["usage"] = self._usage(result)
                return self._send_json(200, out)
            self._sse_open()
            for piece in pieces:
                if piece:
                    self._sse(make(piece))
            self._sse(make("", self._finish(result, state)))
            if want_usage:
                tail = {"id": cid, "object": "text_completion",
                        "created": created, "model": eng.model_id,
                        "choices": [], "usage": self._usage(result)}
                self._sse(tail)
            self._sse_close()

    # ---------------------------------------------------------------- stats
    @staticmethod
    def _usage(result: dict) -> dict:
        p = int(result.get("prompt_tokens", 0))
        c = int(result.get("completion_tokens", 0))
        return {"prompt_tokens": p, "completion_tokens": c,
                "total_tokens": p + c}

    @staticmethod
    def _finish(result: dict, state: dict) -> str:
        if state.get("hit"):
            return "stop"
        return result.get("finish_reason", "stop")


class _BadRequest(Exception):
    def __init__(self, message: str, param=None):
        super().__init__(message)
        self.param = param


class _WSRefusal(Exception):
    """A stream-session refusal: becomes `{"type":"error",...}` + clean close.

    The WebSocket twin of `_BadRequest` — raised anywhere inside a session,
    caught in one place, so every refusal takes the same wire shape."""


class _PooledSpeech:
    """The one method `LiveSession` needs, routed onto the MLX worker thread.

    `SpeechEngine.__init__` explains why every MLX call must run on the single
    long-lived worker; `LiveSession` calls `speech.transcribe_array()` from
    the request thread, so this facade submits each decode to that worker and
    waits — which also serialises a live stream's decodes behind the same
    single Metal queue the file endpoint uses. Stats ride along the same way
    `transcribe_bytes` counts them."""

    def __init__(self, eng):
        self._eng = eng

    def transcribe_array(self, audio, **kw):
        eng = self._eng
        text, decode_s, audio_s = eng._pool.submit(
            eng.speech.transcribe_array, audio, **kw).result()
        with eng.lock:
            eng.stats["decode_seconds"] += decode_s
        return text, decode_s, audio_s


# ------------------------------------------------------------ speech engine
class SpeechEngine:
    """Loads a Phonon profile once; serialises decodes behind one lock.

    Deliberately duck-types the small part of `Engine` the Handler's shared
    plumbing touches (`args`, `model_id`, `created`, `kind`, `lock`), and
    nothing else. It has no `sampler()`, no `max_new()` and no `context`,
    because a speech decode has none of those — and the routing below never
    asks it for any, since THE SERVED MODEL DETERMINES WHICH ENDPOINTS EXIST.

    The chat/completions handlers are not merely unused on a speech server:
    they are not mounted, so a client that POSTs to `/v1/chat/completions`
    against a Phonon server gets a 404 that names the endpoint it should have
    used, rather than a 500 from a code path that was never going to work.
    """

    kind = "speech"

    def __init__(self, args):
        from fermion._speech import backends, fetch

        self.args = args
        # THE REPO IS THE PROFILE. There is no `--profile` flag; resolving the
        # model resolves everything, and a local directory is accepted too so
        # an unpublished or developer build can be served.
        from fermion.transcribe import _resolve as _resolve_speech
        repo, key, pin, local_dir = _resolve_speech(args.model)
        self.repo = repo or str(local_dir)

        # Engine gate BEFORE the download, and a hard refusal on a machine
        # neither engine can serve: someone typing `fermion serve --model
        # FermionResearch/Phonon-1` asked explicitly. Apple silicon serves
        # through MLX; Linux (and FERMION_DEVICE=cpu) through the CPU
        # engine — same models, same endpoints.
        engine_kind = backends.resolve(f"`{BRAND} serve --model {args.model}`")
        backends.require_engine_for(engine_kind, (pin or {}).get("backend", ""))

        self.profile = key
        self.pin = pin
        self.model_dir = local_dir if local_dir is not None \
            else fetch.ensure(repo, key, pin)

        # ONE THREAD OWNS MLX, FOR ITS WHOLE LIFE. This is not an optimisation,
        # it is a correctness requirement, and it cost a real bug to find:
        # `ThreadingHTTPServer` runs every request on a FRESH thread, and MLX's
        # default stream is THREAD-LOCAL — a model built on the main thread and
        # decoded on a request thread dies with
        #     RuntimeError: There is no Stream(gpu, 0) in current thread.
        # A per-request `set_default_device` would paper over it; a single
        # long-lived worker fixes it at the root and gives the serialisation a
        # single Metal command queue needs anyway (the same reason `Engine`
        # serialises generation behind one lock). The model is LOADED on this
        # thread too, so no MLX object ever crosses a thread boundary.
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="fermion-speech")
        self.engine_kind = engine_kind
        self.speech = self._pool.submit(
            backends.load, engine_kind, self.model_dir, profile=key,
            backend=pin["backend"]).result()
        self.model_id = args.served_model_name or self.repo
        #: The pinned archive digest — the same field `Engine.sha256` carries
        #: for a container, so the startup banner and `/v1/models` need no
        #: branching.
        self.sha256 = pin.get("sha256") or ""
        self.created = int(time.time())
        self.lock = threading.Lock()
        # ONE LIVE STREAM AT A TIME (`/v1/audio/stream`). Decodes already
        # serialise on the single MLX worker above; this guard exists so a
        # SECOND stream is refused with a clean in-band error instead of
        # having its partials starve behind the first one's decodes for the
        # whole session. Non-blocking acquire only — a stream must never hang.
        self.stream_guard = threading.Lock()
        self._warmed = False
        # Counters, in the spirit of `native.session_stats`: a serve process
        # should be able to say what it has done without log archaeology.
        self.stats = {"requests": 0, "streams": 0, "audio_seconds": 0.0,
                      "decode_seconds": 0.0, "errors": 0}
        #: OpenAI request fields this runtime does not implement, warned about
        #: once per process rather than dropped in silence.
        self._warned_fields: set = set()

    # -- what this server can do, for /health and / ------------------------
    def endpoints(self) -> list[str]:
        return ["/v1/audio/transcriptions", "/v1/audio/stream",
                "/v1/models", "/health"]

    def health(self) -> dict:
        cfg = self.speech.describe()
        stats = dict(self.stats)
        rtf = (stats["audio_seconds"] / stats["decode_seconds"]
               if stats["decode_seconds"] > 0 else None)
        return {
            "status": "ok", "model": self.model_id, "kind": "speech",
            "version": VERSION, "repo": self.repo,
            "profile": self.profile,
            "sha256": self.sha256 or None,
            "sha256_short": (self.sha256[:8] if self.sha256 else None),
            "download_bytes": self.pin.get("download_bytes"),
            "model_dir": str(self.model_dir),
            # The configuration these transcripts were actually produced at.
            # Same job as the LLM server naming its `--tool-profile`: a
            # benchmark must never be silently mis-attributed.
            "decode": cfg,
            "endpoints": self.endpoints(),
            "tool_calling": False,
            "stats": {**stats,
                      "realtime_factor": round(rtf, 2) if rtf else None},
        }

    # -- decoding ----------------------------------------------------------
    def transcribe_bytes(self, blob: bytes, filename: str, *, hotwords=None, hotword_lambda=None):
        """A `segment.Transcript` (text, timings, per-window segments,
        truncated flag). Runs on the MLX worker.

        Decoding is submitted to the single-worker pool, which both puts it on
        the thread that owns the MLX stream (see `__init__`) and serialises
        concurrent requests behind one Metal command queue. Audio DECODING is
        done on the request thread on purpose: `soundfile` + `scipy` are plain
        CPU work with no MLX involvement, so a malformed upload is rejected
        without ever occupying the model.
        """
        from fermion._speech import engine as _engine

        from fermion._speech.mlx_memory import clear_mlx_cache

        audio = _engine.read_audio_bytes(blob, filename)
        try:
            if hotwords:
                # per request: set, decode, put the process setting back -- all on the one worker, so concurrent
                # requests never see each other's words
                result = self._pool.submit(
                    lambda: self.speech.transcribe_array_detailed(
                        audio, hotwords=hotwords, hotword_lambda=hotword_lambda)).result()
            else:
                result = self._pool.submit(
                    self.speech.transcribe_array_detailed, audio).result()
        finally:
            clear_mlx_cache(synchronize=True)
        with self.lock:
            self.stats["requests"] += 1
            self.stats["audio_seconds"] += result.audio_seconds
            self.stats["decode_seconds"] += result.decode_seconds
        return result

    # -- streaming (`/v1/audio/stream`) -------------------------------------
    def pooled(self) -> _PooledSpeech:
        """The `LiveSession`-shaped view of this engine (see _PooledSpeech)."""
        return _PooledSpeech(self)

    def warm(self) -> None:
        """One discarded decode before the first stream, once per process.

        The first decode through a freshly loaded model pays the Metal graph
        compile — seconds of wall time that would otherwise land on a
        session's FIRST partial and wreck the 0.35 s cadence. `fermion
        listen` warms up at spin-up for the same reason; here it is lazy so
        a file-only serve process never pays it.
        """
        with self.lock:
            if self._warmed:
                return
            self._warmed = True
        from fermion._speech import live as _live
        took = self._pool.submit(_live.warm_up, self.speech).result()
        print(f"[{BRAND}] stream warm-up decode: {took:.1f}s "
              f"(first stream in this process only)", file=sys.stderr)

    def note_stream(self, audio_s: float) -> None:
        with self.lock:
            self.stats["streams"] += 1
            self.stats["audio_seconds"] += audio_s


# -------------------------------------------------------------------- entry
def _is_local_speech_dir(spec) -> bool:
    """Is `--model` a local directory holding an unpacked Phonon profile?

    `transcribe --model` accepts such a directory (it is how a developer runs
    an unpublished build), so `serve` must resolve it identically — 0.1.17
    dispatched on the catalog alone and sent a local speech directory down the
    LLM path, where it died looking for a TRTC container that was never going
    to be there. The test is `fetch.is_model_dir` (config.json +
    packed_manifest.json side by side), which no TRTC directory satisfies, so
    local LLM directories keep taking the pre-existing path. The speech module
    is imported only once the spec is literally a directory on disk: a
    catalog-named LLM serve still imports nothing speech-shaped.
    """
    try:
        p = Path(str(spec)).expanduser()
        if not p.is_dir():
            return False
        from fermion._speech import fetch
        return fetch.is_model_dir(p)
    except Exception:                                    # noqa: BLE001
        return False


def _speech_serve(args):
    """Build the speech engine, and refuse LLM-only flags rather than drop them.

    `serve`'s parser is shared, so a speech invocation still ACCEPTS
    `--temperature`, `--draft`, `--session-ctx` and the rest. Silently ignoring
    a flag the user typed is the defect `UNSUPPORTED_SAMPLER_KNOBS` exists to
    prevent, so anything that cannot mean something here is named once, on
    stderr, at startup.
    """
    ignored = []
    for flag, attr, default in (("--draft", "draft", None),
                                ("--tool-profile", "tool_profile", "graded"),
                                ("--session-ctx", "session_ctx", 8192),
                                ("--max-new-ceiling", "max_new_ceiling", 4096),
                                ("--kv-dtype", "kv_dtype", "fp16"),
                                ("--backend", "backend", "auto"),
                                ("--yarn-factor", "yarn_factor", None)):
        if getattr(args, attr, default) != default:
            ignored.append(flag)
    if getattr(args, "temperature", None) not in (None, _cli.SHIP_TEMPERATURE):
        ignored.append("--temperature")
    if ignored:
        print(f"[{BRAND}] note: {', '.join(sorted(set(ignored)))} "
              f"{'has' if len(set(ignored)) == 1 else 'have'} no meaning for a "
              f"speech model and {'is' if len(set(ignored)) == 1 else 'are'} "
              f"ignored — transcription runs one fixed, graded decode "
              f"configuration.", file=sys.stderr)
    return SpeechEngine(args)


def serve(args) -> None:
    # `fermion serve` was observed dying SILENTLY once mid-session — connect
    # refused, zero trace in the log. faulthandler costs nothing and turns
    # a native-level
    # crash (SIGSEGV/SIGABRT in torch, the tokenizer, or a bad extension)
    # into a stderr traceback, so the next such death names its cause.
    import faulthandler
    try:
        faulthandler.enable(file=sys.stderr)
    except Exception:                                    # pragma: no cover
        pass
    # THE SERVED MODEL DETERMINES WHICH ENDPOINTS ARE MOUNTED — decided here,
    # from the catalog, with NO network access and NO download: `_catalog
    # .is_speech()` is a static lookup, so `--model phonon` never touches
    # `resolve_model()` (which would try to find a TRTC container in a repo
    # that has none) and `--model <anything else>` reaches `Engine` exactly as
    # it did in 0.1.14. An unknown spec — a local path, a third-party repo —
    # is treated as a language model, which is the pre-existing behaviour,
    # with ONE exception: a local directory that holds an unpacked Phonon
    # profile is speech, because `transcribe --model` already accepts exactly
    # that directory and the two verbs must resolve a model the same way.
    from fermion import _modelarg
    _modelarg.take_model(args, "serve", "fermion serve phonon-2        (or: fermion serve neutrino-8b)")
    print(f"[{BRAND}] serving the model you named: {args.model}", file=sys.stderr)
    if getattr(args, "threads", None):
        os.environ["FERMION_CPU_THREADS"] = str(int(args.threads))
    if _catalog.is_speech(args.model) or _is_local_speech_dir(args.model):
        engine = _speech_serve(args)
    else:
        engine = Engine(args)
    if not _is_loopback(args.host):
        print(f"[{BRAND}] WARNING: binding {args.host} exposes an "
              f"UNAUTHENTICATED model server beyond this machine"
              f"{'' if getattr(args, 'api_key', None) else ' — no --api-key set'}",
              file=sys.stderr)
    unix_path = getattr(args, "unix_socket", None)
    if unix_path:
        import socket as _sock
        if not hasattr(_sock, "AF_UNIX"):
            raise SystemExit(f"[{BRAND}] --unix-socket is not available on this platform's Python (no AF_UNIX); "
                             f"use --host 127.0.0.1 --port PORT instead.")
        httpd = _UnixThreadingHTTPServer(str(Path(unix_path).expanduser()), Handler, bind_and_activate=True)
    else:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.engine = engine                                # type: ignore[attr-defined]
    host, port = httpd.server_address[0], httpd.server_address[1]
    base = f"http+unix://{host}/v1" if unix_path else f"http://{host}:{port}/v1"
    print(f"[{BRAND}] serving {engine.model_id} (sha {engine.sha256[:8]}) "
          f"on {base}", file=sys.stderr)
    print(f"[{BRAND}] OpenAI-compatible: set base_url={base} "
          f"(api key: {'required' if getattr(args, 'api_key', None) else 'any'})",
          file=sys.stderr)
    if getattr(engine, "kind", "") == "speech":
        cfg = engine.speech.describe()
        print(f"[{BRAND}] endpoint: POST {base}/audio/transcriptions "
              f"(multipart `file` + `model`; the Whisper API shape, so "
              f"existing OpenAI clients work unmodified)", file=sys.stderr)
        print(f"[{BRAND}] endpoint: GET  {base}/audio/stream (WebSocket — "
              f"live transcription: 16 kHz mono PCM frames in, "
              f"partial/final/done JSON out; one stream at a time; with "
              f"--api-key, pass it as a Bearer header or ?api_key=. "
              f"Protocol + reference client: README 'Streaming')",
              file=sys.stderr)
        if cfg.get("engine") == "cpu":
            print(f"[{BRAND}] profile {engine.profile} — CPU engine, "
                  f"deterministic decode", file=sys.stderr)
        else:
            # Phonon-2 engines describe themselves without an `audio_tower_bits` key (five-value encoder, no tower):
            # `.get` keeps `fermion serve --model phonon-2` from dying in its own banner.
            tower = cfg.get("audio_tower_bits")
            print(f"[{BRAND}] profile {engine.profile} — backend "
                  f"{cfg.get('backend')}, "
                  + (f"audio tower {tower}-bit: the configuration the Mac app ships" if tower
                     else f"runtime {cfg.get('runtime', 'mlx')}"), file=sys.stderr)
        print(f"[{BRAND}] transcription is deterministic (temperature 0); "
              f"the chat/completions endpoints are NOT mounted for a speech "
              f"model", file=sys.stderr)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n[{BRAND}] shutting down", file=sys.stderr)
        finally:
            httpd.server_close()
        return
    print(f"[{BRAND}] defaults temp={args.temperature} top_p={args.top_p} "
          f"rep_pen={args.rep_penalty} pen_window={args.pen_window} "
          f"max_tokens={args.max_new}", file=sys.stderr)
    print(f"[{BRAND}] tool calling: ON (`tools` / `tool_choice` / "
          f"`role: \"tool\"`); max_tokens clamped to "
          f"min(--max-new-ceiling {getattr(args, 'max_new_ceiling', 4096)}, "
          f"context {engine.context}); a streaming tools request streams "
          f"prose live and buffers only from the first `<tool_call` marker",
          file=sys.stderr)
    det = engine.stuck
    if det.enabled:
        print(f"[{BRAND}] stuck detector: {det.mode} (identical call x"
              f"{det.repeat} | A-B alternation x{det.alternate} | same error x"
              f"{det.error}); poll-type tools whitelisted"
              f"{' + ' + ','.join(sorted(det.whitelist)) if det.whitelist else ''}"
              f"; on trip = "
              f"{'one tools-disabled diagnosis turn then a forced answer' if det.mode == 'enforce' else 'log only'}"
              f"; counters on /health", file=sys.stderr)
    else:
        print(f"[{BRAND}] stuck detector: OFF (--stuck-detector off) — a "
              f"looping client loop will not be interrupted", file=sys.stderr)
    if engine.tool_profile == "graded":
        print(f"[{BRAND}] tool sampling: graded shipping config (requests with "
              f"`tools` are sampled exactly like everything else)",
              file=sys.stderr)
    else:
        prof = engine.profile_for(True)
        print(f"[{BRAND}] tool sampling: --tool-profile {engine.tool_profile} "
              f"-> {prof} for requests that carry `tools`. NOTE: this is NOT "
              f"the graded configuration — no banked number was measured at "
              f"it, so do not report benchmark results from this process",
              file=sys.stderr)
    nat = engine.model if getattr(engine.model, "backend", "") == "native" \
        else None
    if nat is not None:
        print(f"[{BRAND}] kv reuse: {nat.session_note}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{BRAND}] shutting down", file=sys.stderr)
    finally:
        httpd.server_close()

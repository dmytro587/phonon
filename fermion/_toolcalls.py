"""Tolerant `<tool_call>` parser — the ONE place tool-call text is decoded.

Shared by `fermion serve` (which turns the result into OpenAI
`choices[].message.tool_calls[]`) and by any CLI surface that wants to show
the same thing, so there is exactly one recovery policy in the package.

WHY IT IS TOLERANT
------------------
Measured on a live agent loop, the
model's tool-call envelopes fail in four distinct ways, sometimes two at once:

  1. duplicate/stacked `<tool_call>` openers  (`<tool_call>\\n<tool_call>\\n{…}`)
  2. a DROPPED `</tool_call>` closer
  3. single-quoted "JSON" — the canonical Hermes tool-calling system prompt
     contains a single-quoted example, so the model is actively taught to
     emit invalid JSON
  4. truncation mid-closing-tag (`…}\\n</tool_`) when a token budget runs out

A strict `<tool_call>(.*?)</tool_call>` regex crashes or silently drops the
call on 4 of 6 catalogued shapes. The rule that recovered 6/6:

    **IGNORE THE CLOSING TAG. From each opener, scan forward for the first
    BALANCED `{…}` span and parse that.**

That single rule subsumes cases 2 and 4 (there is no closer to find), works
unchanged on case 1 once stacked openers are collapsed, and leaves the
well-formed path byte-identical. Case 3 is a separate `ast.literal_eval`
fallback on the extracted span.

The brace scanner is STRING-AWARE: `{"content": "a } b"}` and
`{"path": "c:\\\\x"}` are common tool arguments, and a naive depth counter
truncates both. Every recovery is reported in `anomalies` so callers can log
how often a strict parser would have broken.
"""
from __future__ import annotations

import json
import re
import uuid

# The literal that flips output from prose to a tool call. This is the ONE
# place the marker is spelled: `parse_tool_calls`'s gate below and the serve
# streamer (`server._chat_with_tools_stream`) both import THIS, so the "is a
# tool call starting?" question has a single answer package-wide. Note the
# opener the model actually emits is `<tool_call\s*>` (see `_OPENER`), but the
# 10-char prefix `<tool_call` is what the tolerant parser keys on, so it is
# also what the streamer must key on to stay byte-identical.
TOOL_CALL_MARKER = "<tool_call"

_OPENER = re.compile(r"<tool_call\s*>")
_STACKED = re.compile(r"(<tool_call\s*>\s*){2,}")
_ANY_TAG = re.compile(r"</?tool_call\s*>")


# ------------------------------------------------------------------ scanning
def first_balanced_object(text: str) -> tuple[str | None, int]:
    """First balanced `{…}` span in `text`, string- and escape-aware.

    Returns `(span, end_index)`; `(None, -1)` when there is no complete object
    (an unterminated one is a truncated generation, not a recoverable call).
    """
    depth = 0
    start = -1
    in_str = False
    quote = ""
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            # only meaningful inside an object we have already opened
            if depth:
                in_str, quote = True, ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], i + 1
    return None, -1


def _iter_objects(text: str):
    """Yield `(start, end, span)` for EVERY balanced `{…}` span in `text`.

    Same string/escape-aware scanner as `first_balanced_object`, applied
    repeatedly from the end of the previous object. Cost is O(len·n_objects)
    from the re-slicing, which is nothing at completion sizes and keeps
    `first_balanced_object`'s pinned single-span contract untouched.
    """
    cursor = 0
    while True:
        span, rel_end = first_balanced_object(text[cursor:])
        if span is None:
            return
        end = cursor + rel_end
        yield end - len(span), end, span
        cursor = end


def _loads_tolerant(span: str) -> tuple[dict | None, bool]:
    """`(obj, was_loose)` — strict JSON first, then a single-quoted literal."""
    span = span.strip().rstrip(",")
    try:
        obj = json.loads(span)
        if isinstance(obj, dict):
            return obj, False
    except Exception:
        pass
    try:
        import ast
        obj = ast.literal_eval(span)
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        pass
    return None, True


def _args_string(value) -> str:
    """OpenAI requires `function.arguments` to be a JSON *string*."""
    if isinstance(value, str):
        obj, _ = _loads_tolerant(value)
        if obj is not None:                      # re-emit as strict JSON
            return json.dumps(obj, ensure_ascii=False)
        return value                             # opaque string, pass through
    if value is None:
        value = {}
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:                            # pragma: no cover
        return "{}"


def _call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _as_call(obj) -> tuple[str, object] | None:
    """`(name, raw_arguments)` when `obj` is a tool call, else `None`.

    Extracted verbatim from `parse_tool_calls`'s old inline body so the
    envelope scanner can ASK whether a candidate object is a call instead of
    committing to the first object it finds.
    """
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function")
    if isinstance(name, dict):               # {"function": {"name": …}}
        inner = name
        name = inner.get("name")
        if obj.get("arguments") is None:
            obj = {**obj, "arguments": inner.get("arguments")}
    if not name or not isinstance(name, str):
        return None
    raw_args = obj.get("arguments")
    if raw_args is None:
        raw_args = obj.get("parameters")
    return name, raw_args


def _batch(obj) -> list | None:
    """Inner list of a `{"tool_calls": [ … ]}`-style wrapper, else `None`.

    Only consulted when the object itself is NOT a call, so a real tool whose
    arguments happen to contain a `tool_calls` key can never be mistaken for a
    wrapper.
    """
    if isinstance(obj, dict):
        for key in ("tool_calls", "calls", "function_calls"):
            v = obj.get(key)
            if isinstance(v, list) and v and all(isinstance(e, dict) for e in v):
                return v
    return None


# -------------------------------------------------------------------- public
_THINK_RE = None  # compiled lazily; module import stays regex-light


def _strip_think(text: str, anomalies: list[str]) -> str:
    """Remove <think>...</think> spans from model output.

    Policy (0.1.13): thinking is private and
    side-effect-free — a call inside a CLOSED think block is never executed
    and the monologue never reaches message.content. But an UNCLOSED
    <think> followed by a <tool_call> opener is the documented Qwen3/Hermes
    end-of-reasoning shape (the model stopped thinking and acted), so the
    opener implicitly closes the think block and the call IS parsed. An
    unclosed <think> with no call after it hides the rest of the turn.
    See parse_tool_calls step (0)."""
    if "<think>" not in text:
        return text
    global _THINK_RE
    if _THINK_RE is None:
        import re as _re
        _THINK_RE = _re.compile(r"<think>.*?</think>", _re.S)
    stripped, n = _THINK_RE.subn("", text)
    idx = stripped.find("<think>")
    if idx != -1:                                 # unclosed block
        opener = stripped.find(TOOL_CALL_MARKER, idx)
        if opener != -1:                          # implicit close at the call
            stripped = stripped[:idx] + stripped[opener:]
        else:
            stripped = stripped[:idx]
        n += 1
    if n:
        anomalies.append(f"think-block stripped x{n} (private, no side effects)")
    return stripped


def parse_tool_calls(content: str | None) -> tuple[str, list[dict], list[str]]:
    """Split model output into `(clean_text, tool_calls, anomalies)`.

    `tool_calls` are OpenAI-shaped and ordered as emitted, so parallel calls
    in one turn come back as several entries. `clean_text` is the prose with
    every tool-call span removed (empty string when the turn was calls only).
    """
    anomalies: list[str] = []
    if not content or TOOL_CALL_MARKER not in content:
        return _strip_think(content or "", anomalies), [], anomalies

    # (0) THINK-TAG HARDENING (0.1.13). We ship thinking OFF everywhere and
    # never advertise a thinking mode, but the Qwen3 heritage means <think>
    # tags exist in-vocabulary and can leak on unguarded paths (measured on
    # the fork's llama-server, 2026-07-29). Rule: anything inside
    # <think>…</think> is private and side-effect-free — a tool call written
    # while "thinking" is NOT executed, and the monologue never reaches
    # message.content. An UNCLOSED <think> hides the rest of the turn (the
    # conservative read: half-open reasoning is still reasoning).
    content = _strip_think(content, anomalies)
    if TOOL_CALL_MARKER not in content:
        return content, [], anomalies

    # (1) collapse stacked openers so each real call has exactly one.
    text, n_dup = _STACKED.subn("<tool_call>", content)
    if n_dup:
        anomalies.append(f"duplicate <tool_call> opener x{n_dup}")

    calls: list[dict] = []
    prose: list[str] = []
    pos = 0
    for m in _OPENER.finditer(text):
        if m.start() < pos:                      # inside a span already taken
            continue
        prose.append(text[pos:m.start()])
        after = text[m.end():]
        # (2) THE RULE: never trust the closer; take balanced `{…}` spans.
        #
        # Parser audit (2026-07-30): scan EVERY object in the envelope, not just the
        # first. Two measured-in-the-wild Qwen3/Hermes shapes broke the
        # first-object-only rule:
        #   P1  several calls packed into ONE envelope — newline-separated
        #       objects or a JSON array — where calls 2..n were dropped AND
        #       their raw JSON leaked into `message.content` as prose;
        #   P2  `<tool_call>` used as an end-of-reasoning marker, so the
        #       envelope opens with reasoning text; if that text contains a
        #       brace (`{total}`) or a non-call object (`{"step": 1}`) the
        #       first object was not a call, the parser gave up, and the REAL
        #       call was lost.
        # Policy: keep scanning while the gap between objects is only
        # whitespace / array punctuation, or while nothing has parsed as a
        # call yet; stop at any `<tool_call>`/`</tool_call>` tag (a different
        # envelope) and stop at prose that FOLLOWS a good call (that prose
        # belongs to the turn, not to the envelope — pinned by test H4).
        accepted: list[tuple[str, object, bool]] = []
        found_any = False
        stopped_at_tag = False
        prev_end = 0
        last_end = 0
        for s, e, span in _iter_objects(after):
            gap = after[prev_end:s]
            if _ANY_TAG.search(gap):
                stopped_at_tag = True             # next envelope starts here
                break
            if accepted and gap.strip(" \t\r\n,[]"):
                break                             # prose after a real call
            found_any = True
            prev_end = e
            obj, loose = _loads_tolerant(span)
            batch = None if obj is None else (
                None if _as_call(obj) else _batch(obj))
            if batch is not None:
                anomalies.append("tool_calls wrapper object unwrapped "
                                 f"({len(batch)} inner call(s))")
            hit = False
            for entry in (batch if batch is not None else [obj]):
                got = _as_call(entry)
                if got is None:
                    continue
                accepted.append((got[0], got[1], loose))
                hit = True
            if hit:
                last_end = e
            elif obj is None:
                anomalies.append(f"unparseable tool_call payload: {span[:120]!r}")
            else:
                anomalies.append(f"tool_call missing a name: {obj!r}")
        if not found_any:
            if stopped_at_tag:
                # An empty envelope: a bare `<tool_call>` with no payload, or
                # prose that merely MENTIONS the tag. Consume the tag only so
                # the prose survives and the next opener still gets its turn.
                pos = m.end()
                continue
            anomalies.append("truncated tool_call: no complete JSON object "
                             f"after opener ({after[:80]!r})")
            pos = len(text)
            break
        if len(accepted) > 1:
            anomalies.append(f"{len(accepted)} tool calls packed into ONE "
                             "<tool_call> envelope (newline-separated objects "
                             "or a JSON array)")
        end = last_end or prev_end
        # `[\s,\]]*` also eats the `]` / `,` left over by an array-packed
        # envelope so the closer still matches and no junk leaks as prose.
        closer = re.compile(r"[\s,\]]*</tool_call\s*>").match(after, end)
        if closer:
            pos = m.end() + closer.end()
        else:
            # dropped closer, or truncated mid-tag ("</tool_"). Consume the
            # object plus any partial CLOSING tag after it and keep going. The
            # pattern must require the "</" prefix: an all-optional-chars
            # pattern also matched a fresh "<tool_call>" opener, which
            # swallowed the NEXT call whenever a dropped closer was directly
            # followed by another call (the T01/T07 double-call shape from the
            # behavioural deep-dive).
            partial = re.match(r"[\s,\]]*</[a-z_]*>?", after[end:])
            anomalies.append("missing or truncated </tool_call> closer")
            pos = m.end() + end + (partial.end() if partial else 0)

        for name, raw_args, loose in accepted:
            if loose:
                anomalies.append("non-strict JSON in tool_call "
                                 "(single quotes / trailing junk)")
            calls.append({"id": _call_id(), "type": "function",
                          "function": {"name": name,
                                       "arguments": _args_string(raw_args)}})
    prose.append(text[pos:])

    clean = _ANY_TAG.sub("", "".join(prose)).strip()
    return clean, calls, anomalies


def strict_would_fail(anomalies: list[str]) -> bool:
    """True when a `<tool_call>(.*?)</tool_call>` regex would have broken."""
    return bool(anomalies)


# ------------------------------------------------------- tool-name aliasing
# WHY THIS EXISTS (0.1.8). Measured on an MCP-based agent framework driving
# the 8B through a standard MCP toolbelt: the model emits bare memorized tool
# names (`read_file`) regardless
# of the names declared in the schema (`mcp__toolbox__read_file`), and cannot
# copy the correct name even from an error message that spells it out — 24
# corrective errors in a row were ignored, 0/24 tool calls connected.
# "schema carries the name" was DISPROVEN by proxy capture. A controlled A/B
# (same schemas, only the names differ) confirmed causality: bare-name
# schemas connect on the first turn. Since `mcp__<server>__<tool>` is how
# most frameworks namespace ALL MCP tools, the model was 0%
# usable with MCP until this layer.
#
# The fix is serve-side and conservative: when a parsed call's name does not
# EXACTLY match a declared tool, resolve it by unique normalized/suffix match
# and rewrite the emitted name to the declared one. Ambiguous (two declared
# tools would match) or no match => pass through unchanged, never guess.

_NS_SPLIT = re.compile(r"__|[./:]")     # mcp__srv__tool, srv.tool, srv/tool…


def _name_key(name: str) -> str:
    """Case- and underscore/hyphen-insensitive comparison key."""
    return name.lower().replace("-", "").replace("_", "")


def _suffix_key(name: str) -> str:
    """Comparison key of the last namespace segment (`mcp__srv__t` -> `t`)."""
    return _name_key(_NS_SPLIT.split(name)[-1])


def alias_tool_names(calls: list[dict], registered: list[str]) -> list[str]:
    """Rebind model-emitted tool names onto the client's registered names.

    THE MEASURED PROBLEM: given namespaced schemas — and
    `mcp__<server>__<tool>` is how most frameworks name EVERY
    MCP tool — the model emits the memorized canonical name (`read_file`)
    instead of copying the schema name (`mcp__toolbox__read_file`), and
    cannot correct itself even when the error message spells the right name
    out verbatim: 0 successful executions in ~24 attempts. A controlled A/B
    with identical schemas/guidance showed bare names connect on the first
    turn, i.e. the ONLY broken layer is name binding. That is fixable here,
    where both the emitted name and the client's schema are visible.

    Rule, tiered — a tier only runs when every earlier tier had ZERO hits,
    and a rewrite needs exactly ONE hit in its tier (2+ is ambiguous: the
    call is left untouched and the client's own unknown-tool handling stays
    authoritative — we never guess between two tools):

      1. raw `__` namespace boundary, either direction (`read_file` ->
         `mcp__toolbox__read_file`, or a fully-qualified emission against a
         bare belt) — the exact measured MCP shape;
      2. normalized whole-name equality (case / `_` / `-` insensitive:
         `Read-File` -> `read_file`);
      3. normalized suffix matches across common namespace separators
         (`__`, `.`, `/`, `:`) — declared-suffix == emitted, emitted-suffix
         == declared, then suffix == suffix.

    Mutates `calls` in place; returns human-readable notes for the serve log
    (observability rule: every recovery is reported, same as the parser's
    anomalies).
    """
    notes: list[str] = []
    reg = [r for r in registered if isinstance(r, str) and r]
    if not calls or not reg:
        return notes
    regset = set(reg)
    for call in calls:
        fn = call.get("function") if isinstance(call, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name or name in regset:
            continue
        hits = [r for r in reg
                if r.endswith("__" + name) or name.endswith("__" + r)]
        why = "unique namespace-suffix match"
        if not hits:
            ek, es = _name_key(name), _suffix_key(name)
            for cand, w in (
                ([r for r in reg if _name_key(r) == ek],
                 "unique case/underscore-normalized match"),
                ([r for r in reg if _suffix_key(r) == ek],
                 "unique normalized namespace-suffix match"),
                ([r for r in reg if _name_key(r) == es],
                 "unique normalized namespace-suffix match"),
                ([r for r in reg if _suffix_key(r) == es],
                 "unique normalized namespace-suffix match"),
            ):
                if cand:
                    hits, why = cand, w
                    break
        if len(hits) == 1:
            fn["name"] = hits[0]
            notes.append(f"tool-call name {name!r} aliased to {hits[0]!r} "
                         f"({why}; the model emits "
                         f"memorized bare names for namespaced schemas)")
        elif len(hits) > 1:
            notes.append(f"tool-call name {name!r} matches {len(hits)} "
                         f"registered tools ({', '.join(hits[:4])}); "
                         f"ambiguous, left unchanged")
    return notes


def render_assistant_tool_calls(tool_calls) -> list[dict]:
    """Normalise inbound assistant `tool_calls` for the chat template.

    The template renders `message.tool_calls[*].function.{name,arguments}` and
    accepts `arguments` as either a string (emitted verbatim) or an object
    (`| tojson`). We hand it parsed objects when we can, so a client that sent
    single-quoted arguments still produces a valid replay of its own history.
    """
    out = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            obj, _ = _loads_tolerant(args)
            args = obj if obj is not None else args
        elif args is None:
            args = {}
        entry = {"type": "function", "function": {"name": name,
                                                 "arguments": args}}
        if tc.get("id"):
            entry["id"] = tc["id"]
        out.append(entry)
    return out

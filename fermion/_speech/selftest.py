"""Self-checks for the speech lane.  `python -m fermion._speech.selftest`

HOUSE PATTERN: a `check()`
closure that prints one line per assertion and a `ran - failures` tally at the
end, returning a non-zero exit code iff anything failed. The package has no
test framework and does not acquire one for this.

TWO TIERS OF CHECK, ON PURPOSE
-------------------------------
Some of these can run anywhere (they only need the installed wheel). Others
compare the wheel against the SOURCE TREES it was cut from — the Mac app's
Swift, the engine's Python, the release `config.json` files. Those
only make sense on a machine that has them, so they are located via
`$FERMION_SPEECH_SOURCES` or by walking up from this file, and are reported as
SKIP (not PASS, never silently) when the trees are absent.

The cross-tree checks are the ones that matter most, because every defect they
guard is a DRIFT defect: someone changes the app's decode line, or re-cuts an
archive, and the CLI keeps confidently printing numbers from the old one.
"""
from __future__ import annotations

import base64
import io
import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

WHEEL_ROOT = Path(__file__).resolve().parents[2]     # .../<tree>/fermion/..


# ------------------------------------------------------ WS reference client
class WsClient:
    """The reference /v1/audio/stream client — plain socket, stdlib only.

    Deliberately NOT built on `fermion._ws`: if the client and the server
    shared one framing implementation, a symmetric encode/decode bug would
    cancel out and the round-trip below would prove nothing. This is an
    independent second implementation of the client side of RFC 6455, and it
    doubles as the reference code the README's "Streaming" section carries.
    """

    def __init__(self, port: int, path: str = "/v1/audio/stream",
                 header: str | None = None, host: str = "127.0.0.1"):
        self.sock = socket.create_connection((host, port), timeout=240)
        self.f = self.sock.makefile("rb")
        self.key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {self.key}\r\n"
               f"Sec-WebSocket-Version: 13\r\n")
        if header:
            req += header + "\r\n"
        self.sock.sendall((req + "\r\n").encode())
        self.status = self.f.readline().decode()
        self.headers = {}
        while True:
            line = self.f.readline()
            if line in (b"\r\n", b""):
                break
            k, _, v = line.decode().partition(":")
            self.headers[k.strip().lower()] = v.strip()

    def send(self, opcode: int, payload: bytes, fin: bool = True) -> None:
        """One frame, client side — always masked (RFC 6455 §5.1)."""
        head = bytes([(0x80 if fin else 0x00) | opcode])
        n = len(payload)
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        self.sock.sendall(head + mask + bytes(
            b ^ mask[i % 4] for i, b in enumerate(payload)))

    def recv(self):
        """-> (opcode, payload). Server frames are never masked."""
        h = self.f.read(2)
        if len(h) < 2:
            return None, b""
        b1, b2 = h
        n = b2 & 0x7F
        if n == 126:
            (n,) = struct.unpack(">H", self.f.read(2))
        elif n == 127:
            (n,) = struct.unpack(">Q", self.f.read(8))
        return b1 & 0x0F, (self.f.read(n) if n else b"")

    def messages(self):
        """Yield decoded JSON text frames until the server closes."""
        while True:
            op, payload = self.recv()
            if op is None or op == 0x8:                       # EOF / close
                return
            if op == 0x1:
                yield json.loads(payload)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------- source trees
def _sources() -> Path | None:
    """The directory holding the app and engine source trees (dev machines)."""
    env = os.environ.get("FERMION_SPEECH_SOURCES")
    if env:
        p = Path(env).expanduser()
        return p if (p / "local_stt").is_dir() else None
    for base in [WHEEL_ROOT, *WHEEL_ROOT.parents]:
        if (base / "local_stt").is_dir() and (base / "phonon_app").is_dir():
            return base
    return None


def main() -> int:                                   # noqa: C901
    ran = failures = skipped = 0

    def check(label: str, ok: bool) -> None:
        nonlocal ran, failures
        ran += 1
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}")

    def skip(label: str, why: str) -> None:
        nonlocal skipped
        skipped += 1
        print(f"SKIP  {label} ({why})")

    src = _sources()
    print(f"# fermion speech selftest — wheel {WHEEL_ROOT}")
    print(f"# source trees: {src or 'NOT FOUND (cross-tree checks skipped)'}\n")

    from fermion import _catalog, _multipart
    from fermion._speech import DECODE, SHIP_BACKEND, decode_kwargs
    from fermion._speech import _engine as engine_pkg
    from fermion._speech import gate

    # ------------------------------------------------ 1. THE FAST-PATH PIN
    # The explicitly demanded check: the CLI's effective speech decode
    # configuration EQUALS the app's.
    print("## 1. the CLI's decode configuration is the app's")
    expected = {"fold_bits": 4, "tiered_head_bits": 4, "tiered_head_k": 32,
                "audio_tower_bits": 8, "audio_tower_group_size": 128}
    got = decode_kwargs(SHIP_BACKEND)
    check("the fold/tiered build decodes at fold4 + tiered4 + 8-bit audio "
          "tower", got == expected)
    check("the module-level DECODE pin matches it", DECODE == expected)
    check("SHIP_BACKEND is the Phonon-1 Big (parity) backend",
          _catalog.SPEECH_PROFILES["parity"]["backend"]
          == SHIP_BACKEND == "parity-fold4-tiered4")
    # THE DEFAULT MOVED (2026-08-28): the unqualified name is Phonon-1, the
    # `audio6` profile. Its decode path is checked here so the path a user
    # actually takes with no `--model` is pinned, not just the Big one.
    # THE DEFAULT MOVED AGAIN (2026-09-28): the unqualified name is Phonon-2, the `five-value` profile.
    check("the default profile is five-value, published as Phonon-2",
          _catalog.DEFAULT_SPEECH_PROFILE == "five-value"
          and _catalog.SPEECH_PROFILES["five-value"]["backend"] == "phonon2-five-value"
          and _catalog.SPEECH_PROFILES["five-value"]["repo"].endswith("/Phonon-2")
          and _catalog.SPEECH_PROFILES["five-value"]["published"])
    check("Phonon-1 (audio6) still decodes with no fold/tiered stage at the same "
          "8-bit audio tower",
          decode_kwargs(_catalog.SPEECH_PROFILES["audio6"]["backend"])
          == {**expected, "fold_bits": None, "tiered_head_bits": None})

    if src is None:
        skip("the app's own source sets exactly this", "no source trees")
        skip("PHONON_ENC=q8 parses to 8 bits / group 128 as stt.py does",
             "no source trees")
    else:
        # (a) the Swift app sets PHONON_ENC=q8
        swift = src / "PhononNative/Sources/Detta/Engine/EngineClient.swift"
        swift_text = swift.read_text() if swift.is_file() else ""
        check("the app sets PHONON_ENC=q8 (app source)",
              'environment["PHONON_ENC"] = "q8"' in swift_text)
        # (b) the Swift app's parity profile runs parity-fold4-tiered4
        catalog_swift = src / "PhononNative/Sources/Detta/Engine/ModelCatalog.swift"
        cat_text = catalog_swift.read_text() if catalog_swift.is_file() else ""
        check("the app's parity profile backend is parity-fold4-tiered4",
              'backend: "parity-fold4-tiered4"' in cat_text)
        # (c) stt.py turns "q8" into exactly our two numbers
        stt = (src / "local_stt/stt.py").read_text()
        m = re.search(r're\.fullmatch\(r"([^"]+)", enc\)', stt)
        check("stt.py's PHONON_ENC regex is the one we reimplemented",
              m is not None and m.group(1) == r"q(\d+)(?:g(\d+))?")
        check("stt.py defaults the group size to 128",
              'int(match.group(2) or 128)' in stt)
        # (d) stt.py maps parity-fold4-tiered4 -> (path, 4, 4)
        check("stt.py maps parity-fold4-tiered4 to fold_bits=4, tiered=4",
              '"parity-fold4-tiered4": (PARITY_MODEL, 4, 4)' in stt)
        # (e) end to end: our kwargs equal what stt.py would have built
        check("our decode kwargs equal the app's, field for field",
              got == {"fold_bits": 4, "tiered_head_bits": 4,
                      "tiered_head_k": 32, "audio_tower_bits": 8,
                      "audio_tower_group_size": 128})

    # $PHONON_ENC is honoured the same way the app honours it.
    old = os.environ.get("PHONON_ENC")
    try:
        os.environ["PHONON_ENC"] = "q6g64"
        check("PHONON_ENC=q6g64 overrides both numbers, as in the app",
              decode_kwargs(SHIP_BACKEND)["audio_tower_bits"] == 6
              and decode_kwargs(SHIP_BACKEND)["audio_tower_group_size"] == 64)
        os.environ["PHONON_ENC"] = ""
        check("PHONON_ENC= (empty) turns the 8-bit tower off",
              decode_kwargs(SHIP_BACKEND)["audio_tower_bits"] is None)
    finally:
        if old is None:
            os.environ.pop("PHONON_ENC", None)
        else:
            os.environ["PHONON_ENC"] = old
    check("removing PHONON_ENC restores the shipped q8 default",
          decode_kwargs(SHIP_BACKEND)["audio_tower_bits"] == 8)

    # ------------------------------------------------------- 2. THE CATALOG
    print("\n## 2. the catalog agrees with the release sources")
    if src is None:
        skip("catalog == the per-repo config.json files", "no source trees")
        skip("catalog == the app-side catalog", "no source trees")
    else:
        repos_root = src / "local_stt/release_ship/hf_repos"
        if not repos_root.is_dir():
            skip("catalog == the per-repo config.json files",
                 "hf_repos/ not staged")
        else:
            for key in _catalog.published_speech_profiles():
                pin = _catalog.SPEECH_PROFILES[key]
                repo = pin["repo"]
                cfg_path = repos_root / repo.split("/")[-1] / "config.json"
                if not cfg_path.is_file():
                    skip(f"{repo} config.json", "not staged")
                    continue
                published = json.loads(cfg_path.read_text())
                check(f"{repo}: config.json model_id matches the catalog",
                      published.get("model_id") == repo)
                art = published.get("artifact") or {}
                check(f"{repo}: publishes the {key!r} profile and only that",
                      art.get("profile") == key)
                for field in ("filename", "sha256", "download_bytes",
                              "unpacked_bytes"):
                    check(f"{repo}: agrees on {field}",
                          art.get(field) == pin[field])
                check(f"{repo}: agrees on backend",
                      art.get("backend") == pin["backend"])
                entries = (published.get("profiles") or {}).get("entries") or []
                check(f"{repo}: lists exactly one profile entry",
                      len(entries) == 1 and entries[0].get("key") == key)
                fam = {f.get("repo_id") for f in published.get("family") or []}
                check(f"{repo}: names all three siblings in `family`",
                      fam == set(_catalog.SPEECH_REPOS))

        # THE MISTAKE THIS WHOLE REFACTOR EXISTS TO PREVENT: two profiles that
        # resolve to one repo would silently fetch a sibling's archive.
        repos = [p["repo"] for p in _catalog.SPEECH_PROFILES.values()
                 if p["repo"]]
        check(f"no two profiles share a repo ({len(repos)} repos, "
              f"{len(set(repos))} distinct)", len(repos) == len(set(repos)))
        check("every published profile carries a repo",
              all(_catalog.SPEECH_PROFILES[k]["repo"]
                  for k in _catalog.published_speech_profiles()))
        check("SPEECH_REPOS inverts SPEECH_PROFILES exactly",
              _catalog.SPEECH_REPOS
              == {p["repo"]: k for k, p in _catalog.SPEECH_PROFILES.items()
                  if p["repo"]})
        for repo, key in _catalog.SPEECH_REPOS.items():
            check(f"{repo} -> {key} round-trips through profile_for_repo()",
                  _catalog.profile_for_repo(repo)[0] == key)

        models_py = src / "phonon_app/models.py"
        if not models_py.is_file():
            skip("catalog == the app-side catalog", "models.py absent")
        else:
            text = models_py.read_text()
            # Parsed textually rather than imported: importing the app's
            # models.py pulls its whole support-path stack, and this test must
            # stay a pure comparison of the two sets of literals.
            for key in _catalog.SPEECH_PROFILES:
                pin = _catalog.SPEECH_PROFILES[key]
                block = re.search(
                    r'Profile\(\s*\n?\s*"' + key + r'".*?\n    \),',
                    text, re.S)
                if block is None:
                    skip(f"models.py carries the {key} profile",
                         "profile not found in CATALOG")
                    continue
                body = block.group(0)
                check(f"models.py agrees on {key}.filename",
                      f'filename="{pin["filename"]}"' in body)
                check(f"models.py agrees on {key}.sha256",
                      f'sha256="{pin["sha256"]}"' in body)
                digits = f"{pin['download_bytes']:,}".replace(",", "_")
                check(f"models.py agrees on {key}.download_bytes",
                      f"download_bytes={digits}" in body)
                check(f"models.py agrees on {key}.backend",
                      f'backend="{pin["backend"]}"' in body)

    print("\n## 2b. published vs retained")
    check("exactly four speech profiles are published and the default is one of them",
          sorted(_catalog.published_speech_profiles()) == ["audio6", "five-value", "micro", "parity"]
          and _catalog.DEFAULT_SPEECH_PROFILE in _catalog.published_speech_profiles())
    check("compact is retained with NO repo, so it can never be downloaded",
          "compact" in _catalog.SPEECH_PROFILES
          and not _catalog.SPEECH_PROFILES["compact"]["published"]
          and _catalog.SPEECH_PROFILES["compact"]["repo"] is None)
    check("compact is bigger than audio6, which is why it was cut",
          _catalog.SPEECH_PROFILES["compact"]["download_bytes"]
          > _catalog.SPEECH_PROFILES["audio6"]["download_bytes"])
    from fermion._speech import fetch as _fetch
    try:
        _fetch.ensure(None, "compact",
                      _catalog.SPEECH_PROFILES["compact"], quiet=True)
        ok = False
    except _fetch.SpeechFetchError as exc:
        ok = "cannot be downloaded" in str(exc)
    except BaseException:                                    # noqa: BLE001
        ok = False
    check("fetching compact refuses instead of constructing a URL", ok)
    check("each published profile maps to a DIFFERENT repo id",
          len({_catalog.SPEECH_PROFILES[k]["repo"]
               for k in _catalog.published_speech_profiles()})
          == len(_catalog.published_speech_profiles()) == 4)

    # ------------------------------------------------- 3. CLOSED KERNELS
    print("\n## 3. the closed kernels are nowhere in what we would ship")
    shipped = sorted(p for p in (WHEEL_ROOT / "fermion").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    for closed in engine_pkg.CLOSED_MODULES:
        present = [p for p in shipped if p.name == f"{closed}.py"]
        check(f"{closed}.py is absent from the package", not present)
    # ...and nothing imports them at module scope, where absence would be a
    # crash rather than an unreachable branch.
    bad = []
    for p in shipped:
        if p.suffix != ".py":
            continue
        for line in p.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if line[:1] not in (" ", "\t") and any(
                    c in stripped for c in engine_pkg.CLOSED_MODULES):
                bad.append(f"{p.name}:{stripped}")
    check(f"no module-scope import of a closed kernel ({bad or 'none'})",
          not bad)
    check("the vendored engine is exactly the five expected modules",
          sorted(p.name for p in engine_pkg.vendored_files())
          == sorted(f"{m}.py" for m in engine_pkg.MODULES))
    for p in engine_pkg.vendored_files():
        check(f"vendored {p.name} is present", p.is_file())

    if src is not None:
        print("\n## 3b. the vendored engine matches its reviewed origin")
        pub = src / "local_stt/release_ship/github_repo"
        priv = src / "local_stt"
        for name in ("quint5_codec", "quint_fold_codec"):
            a = (pub / f"{name}.py")
            b = (WHEEL_ROOT / "fermion/_speech/_engine" / f"{name}.py")
            if a.is_file():
                check(f"{name}.py is byte-identical to the staged public tree",
                      a.read_bytes() == b.read_bytes())
            else:
                skip(f"{name}.py vs the staged public tree", "source absent")
        # package_release_bps.py ships with a neutralized module docstring and
        # pack-mode default out-dir, so the whole file is not byte-identical.
        # `unpack` is the only entry point the CLI calls; assert THAT function
        # is byte-identical, which is the drift that could corrupt an install.
        bps_pub = pub / "package_release_bps.py"
        bps_vnd = WHEEL_ROOT / "fermion/_speech/_engine/package_release_bps.py"
        if bps_pub.is_file():
            def _fn_text(text: str, name: str) -> str:
                i = text.index(f"def {name}(")
                return text[i:text.index("\ndef ", i + 1)]
            check("package_release_bps.py's unpack() is byte-identical to the "
                  "staged public tree",
                  _fn_text(bps_vnd.read_text(), "unpack")
                  == _fn_text(bps_pub.read_text(), "unpack"))
        else:
            skip("package_release_bps.py vs the staged public tree",
                 "source absent")
        # The two back-ported files are NOT byte-identical to the public tree
        # (that is the point). Assert instead that the back-ported function is
        # byte-identical to the PRIVATE tree it came from, and that the public
        # tree's kernel-closure comment survived.
        vr = (WHEEL_ROOT / "fermion/_speech/_engine/v18_runtime.py").read_text()
        pv = priv / "v18_runtime.py"
        if pv.is_file():
            def _fn(text: str, name: str) -> str:
                i = text.index(f"def {name}(")
                return text[i:text.index("\ndef ", i + 1)]
            check("quantize_audio_tower is byte-identical to the private tree",
                  _fn(vr, "quantize_audio_tower")
                  == _fn(pv.read_text(), "quantize_audio_tower"))
        else:
            skip("quantize_audio_tower vs the private tree", "source absent")
        check("the vendored runtime kept the public tree's lazy closed-kernel "
              "imports",
              "not in the public build" in vr)

    # -------------------------------------------- 4. NON-APPLE PLATFORM
    print("\n## 4. a non-Apple platform gets a plain refusal, not a traceback")
    real = platform.machine, platform.system
    try:
        for system, machine in (("Linux", "x86_64"), ("Windows", "AMD64"),
                                ("Darwin", "x86_64")):
            platform.system = lambda s=system: s          # noqa: E731
            platform.machine = lambda m=machine: m        # noqa: E731
            why = gate.why_not()
            check(f"{system}/{machine}: speech is refused",
                  why is not None and not gate.platform_ok())
            check(f"{system}/{machine}: the reason names Apple silicon",
                  bool(why) and "Apple silicon" in why)
            check(f"{system}/{machine}: the reason says the LLM verbs still "
                  f"work",
                  bool(why) and "fermion chat" in why)
            try:
                gate.require("`fermion transcribe`")
                ok = False
            except SystemExit as exc:
                # SystemExit, not a traceback: argparse-grade exit, one line.
                ok = "Apple silicon" in str(exc)
            except BaseException:                          # noqa: BLE001
                ok = False
            check(f"{system}/{machine}: require() raises SystemExit with the "
                  f"reason", ok)
            try:
                gate.try_load(explicit=False)
                ok = True
            except BaseException:                          # noqa: BLE001
                ok = False
            check(f"{system}/{machine}: try_load(explicit=False) returns "
                  f"quietly", ok)
    finally:
        platform.machine, platform.system = real
    check("the real platform is restored after the simulation",
          platform.machine() == real[0]())

    # ------------------------------------------------ 5. MULTIPART PARSER
    print("\n## 5. multipart parsing (no new dependency)")
    boundary = "----fermionTest0123"
    # Adversarial but SPEC-LEGAL: embedded CRLFs, a run of dashes, a
    # near-miss boundary and raw high bytes — everything that trips a
    # line-oriented or text-round-tripping parser — but never the exact
    # delimiter, which RFC 2046 forbids inside a part.
    audio = (b"RIFF\x00\x01\r\n----fermionTest012\r\n--\r\n\r\n"
             b"\xff\xfe\x00\x80not-a-boundary\r\n--" + boundary[:-1].encode()
             + b"\x00\xff")
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"phonon\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="a b.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + audio + \
           f"\r\n--{boundary}--\r\n".encode()
    form = _multipart.parse(body, f"multipart/form-data; boundary={boundary}")
    check("the text field parses", form.get("model") == "phonon")
    check("the file part keeps its filename",
          form["file"]["filename"] == "a b.wav")
    check("binary content survives byte-for-byte, including embedded CRLF "
          "and a boundary-like run", form["file"]["content"] == audio)
    check("the part's content type is kept",
          form["file"]["content_type"] == "audio/wav")
    for bad_ct, why in (("application/json", "wrong content type"),
                        ("multipart/form-data", "no boundary")):
        try:
            _multipart.parse(body, bad_ct)
            ok = False
        except _multipart.MultipartError:
            ok = True
        check(f"a {why} body is rejected as MultipartError", ok)

    # ------------------------------------------------------ 6. THE CLI
    print("\n## 6. the CLI still works and the LLM verbs are unchanged")
    exe = [sys.executable, "-c",
           "import sys; from fermion.cli import main; sys.argv[0]='fermion'; main()"]

    def run(args, expect_rc=0):
        p = subprocess.run(exe + args, capture_output=True, text=True,
                           timeout=120)
        return p, (p.returncode == expect_rc)

    from fermion._brand import VERSION
    p, ok = run(["--version"])
    check(f"`fermion --version` prints {VERSION}",
          ok and VERSION in (p.stdout + p.stderr))
    p, ok = run(["chat", "--help"])
    check("`fermion chat --help` works", ok)
    chat_help = p.stdout
    p, ok = run(["serve", "--help"])
    check("`fermion serve --help` works", ok)
    serve_help = p.stdout
    p, ok = run(["generate", "--help"])
    check("`fermion generate --help` works", ok)
    gen_help = p.stdout
    p, ok = run(["models"])
    check("`fermion models` works with no network and no model", ok)
    models_out = p.stdout
    p, ok = run(["models", "--json"])
    check("`fermion models --json` emits valid JSON",
          ok and isinstance(json.loads(p.stdout), dict))
    p, ok = run(["transcribe", "--help"])
    check("`fermion transcribe --help` works", ok)
    p, ok = run(["listen", "--help"])
    listen_help = p.stdout
    check("`fermion listen --help` works (the command is registered)", ok)
    from fermion import _tools
    check("`listen` dispatches early (owns its model lifecycle) and offers "
          "--wav and --model but never --profile",
          "listen" in _tools.EARLY and "--wav" in listen_help
          and "--model" in listen_help and "--profile" not in listen_help)

    # The LLM surface must be untouched.
    check("chat still defaults to the graded shipping sampler",
          "--temperature TEMPERATURE" in chat_help
          and "0.01" in chat_help)
    check("generate still defaults to deterministic greedy",
          "sampling temperature (default 0.0)" in gen_help)
    check("chat still has --draft", "--draft" in chat_help)
    check("serve still has --tool-profile and --stuck-detector",
          "--tool-profile" in serve_help and "--stuck-detector" in serve_help)
    check("serve still has --api-key, --cors and --max-new-ceiling",
          all(f in serve_help for f in
              ("--api-key", "--cors", "--max-new-ceiling")))
    check("no speech flag leaked into chat",
          "--profile" not in chat_help and "PHONON" not in chat_help)
    check("`fermion models` lists all three speech repos by id",
          all(r in models_out for r in _catalog.SPEECH_REPOS))
    check("`fermion models` does not offer the cut compact profile",
          "Compact" not in models_out)
    p, ok = run(["models", "--all"])
    check("`fermion models --all` explains where compact went",
          ok and "Compact" in p.stdout)
    check("`fermion transcribe --help` has no --profile flag",
          "--profile" not in run(["transcribe", "--help"])[0].stdout)
    check("`fermion serve --help` has no --profile flag",
          "--profile" not in serve_help)

    # A Linux user running `fermion chat` must not be able to tell speech
    # exists. Registration is argparse only: no `fermion._speech` module and no
    # `mlx` may be imported on that path. (`soundfile`/`scipy` may appear —
    # `transformers/audio_utils.py` imports them, which predates this work and
    # is not ours; they are excluded here on purpose rather than ignored.)
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, io, contextlib\n"
         "sys.argv=['fermion','chat','--help']\n"
         "from fermion.cli import main\n"
         "with contextlib.redirect_stdout(io.StringIO()):\n"
         "    try: main()\n"
         "    except SystemExit: pass\n"
         "print([m for m in sys.modules "
         "       if m.startswith('fermion._speech') or m=='mlx'])"],
        capture_output=True, text=True, timeout=120)
    check(f"`chat --help` imports no speech/MLX module "
          f"({probe.stdout.strip()})",
          probe.returncode == 0 and probe.stdout.strip() == "[]")

    # A speech repo must never reach resolve_model (which would download a
    # Neutrino container looking for a .bin that is not there).
    p, ok = run(["generate", "--model", "phonon", "hello"], expect_rc=1)
    check("`--model phonon` on an LLM verb refuses without downloading",
          ok and "speech model" in (p.stdout + p.stderr)
          and "fermion transcribe" in (p.stdout + p.stderr))

    # ------------------------------------------------------- 7. CATALOG API
    print("\n## 7. catalog resolution never touches the network")
    check("the bare aliases resolve to the default, Phonon-2; phonon-1 keeps Phonon-1",
          _catalog.resolve_alias("phonon") == _catalog.SPEECH2_MODEL_ID
          and all(_catalog.resolve_alias(a) == _catalog.SPEECH2_MODEL_ID
                  for a in ("phonon-2", "phonon2", "speech", "stt", "asr"))
          and _catalog.resolve_alias("phonon-1") == _catalog.SPEECH_MODEL_ID
          and _catalog.resolve_alias("8b").endswith("Neutrino-8B"))
    check("each speech repo has its own aliases that do not collide",
          _catalog.resolve_alias("phonon-1-big")
          == _catalog.resolve_alias("big")
          == _catalog.SPEECH_BIG_MODEL_ID
          and _catalog.resolve_alias("phonon-micro")
          == _catalog.resolve_alias("phonon-1-micro")
          == _catalog.SPEECH_MICRO_MODEL_ID)
    # The retired parameter-count names must NOT quietly resolve: a stale
    # alias pointing at a renamed repo is a trap, not compatibility.
    check("the retired 0.8B alias spellings resolve to nothing",
          all(_catalog.resolve_alias(a) == a
              for a in ("phonon-0.8b", "phonon-0.8b-small",
                        "phonon-0.8b-micro", "phonon-small", "small",
                        "fermionresearch/phonon-0.8b")))
    check("a differently-cased repo id resolves to the canonical spelling",
          _catalog.resolve_alias("fermionresearch/phonon-1-big")
          == _catalog.SPEECH_BIG_MODEL_ID)
    check("all three speech repos classify as speech",
          all(_catalog.kind_of(r) == "speech" for r in _catalog.SPEECH_REPOS))
    check("an unknown spec passes through untouched",
          _catalog.resolve_alias("./some/local/dir") == "./some/local/dir")
    check("kind_of() classifies without a network call",
          _catalog.kind_of("phonon") == "speech"
          and _catalog.kind_of("neutrino") == "llm"
          and _catalog.kind_of("./x") == "unknown")
    check("is_speech() is false for an unknown spec (serve keeps 0.1.14 "
          "behaviour)", not _catalog.is_speech("some/other-model"))

    # -------------------------------------- 8. LIVE (only where it can run)
    print("\n## 8. live decode + live endpoint")
    installed_key = None
    if gate.available():
        from fermion._speech import fetch
        for key in _catalog.published_speech_profiles():
            pin = _catalog.SPEECH_PROFILES[key]
            if fetch.installed(pin["repo"], pin["unpack_dir"]):
                installed_key = key
                break
    if installed_key is None:
        why = gate.why_not() or "no profile is downloaded on this machine"
        skip("a real transcript comes out at the pinned configuration", why)
        skip("POST /v1/audio/transcriptions returns that transcript", why)
    else:
        from fermion._speech import engine as _engine
        from fermion._speech import fetch
        pin = _catalog.SPEECH_PROFILES[installed_key]
        model_dir = fetch.profile_dir(pin["repo"], pin["unpack_dir"])
        sample = None
        if src is not None:
            cand = (src / "local_stt/release_ship/github_repo/samples"
                        / "1089-134691-0000.flac")
            sample = cand if cand.is_file() else None
        if sample is None:
            skip("a real transcript comes out at the pinned configuration",
                 "no sample audio in the source tree")
            skip("POST /v1/audio/transcriptions returns that transcript",
                 "no sample audio in the source tree")
        else:
            speech = _engine.load(model_dir, profile=installed_key,
                                  backend=pin["backend"], quiet=True)
            text, decode_s, audio_s = speech.transcribe(sample)
            check(f"a real transcript comes out ({text!r})",
                  "wait no longer" in text.lower())
            cfg = speech.describe()
            check("the live decode really ran at the pinned configuration",
                  cfg["audio_tower_bits"] == 8
                  and cfg["audio_tower_group_size"] == 128
                  and (cfg["fold_bits"] == 4) == (installed_key == "parity"))
            check(f"it decoded faster than real time "
                  f"({audio_s / decode_s:.1f}x)", decode_s < audio_s)

            # A live server round trip, through the real Handler.
            import threading
            import urllib.request
            from http.server import ThreadingHTTPServer

            from fermion import _multipart          # noqa: F401  (shape ref)
            from fermion.server import Handler, SpeechEngine

            class _Args:
                model = pin["repo"]
                served_model_name = None
                api_key = None
                cors = False
                host = "127.0.0.1"
                port = 0
            try:
                eng = SpeechEngine(_Args())
                httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                httpd.engine = eng
                threading.Thread(target=httpd.serve_forever,
                                 daemon=True).start()
                port = httpd.server_address[1]
                bound = "----fermionSelftestBoundary"
                blob = sample.read_bytes()
                body = (
                    f"--{bound}\r\nContent-Disposition: form-data; "
                    f'name="file"; filename="{sample.name}"\r\n'
                    f"Content-Type: audio/flac\r\n\r\n".encode() + blob
                    + f"\r\n--{bound}--\r\n".encode())
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                    data=body,
                    headers={"Content-Type":
                             f"multipart/form-data; boundary={bound}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    served = json.loads(resp.read())
                check(f"POST /v1/audio/transcriptions returns that transcript "
                      f"({served.get('text')!r})",
                      served.get("text", "").strip() == text.strip())
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=30) as resp:
                    health = json.loads(resp.read())
                check("/health reports the speech decode configuration",
                      health["kind"] == "speech"
                      and health["decode"]["audio_tower_bits"] == 8)
                check("/ advertises transcriptions and not chat",
                      "/v1/audio/transcriptions" in health["endpoints"]
                      and "/v1/chat/completions" not in health["endpoints"])
                httpd.shutdown()
                httpd.server_close()
            except Exception as exc:                        # noqa: BLE001
                check(f"the live endpoint serves a transcript ({exc})", False)

    # ------------------------------------ 9. `fermion listen` — the live path
    # `--wav` exists precisely so this machinery is machine-testable: the wav
    # feed and the microphone feed share one `LiveSession.feed()`, so what is
    # asserted here covers everything but the capture device itself. The
    # decode goes through `transcribe_array()` — `transcribe`'s decode — and
    # the final decode of a single-utterance clip sees the clip's complete
    # audio in one call, so the streamed transcript must equal the batch one
    # BYTE FOR BYTE, not merely after normalization.
    print("\n## 9. `fermion listen` — the live path, fed from a file")
    identity_label = ("`listen --wav`'s final transcript is byte-identical "
                      "to the batch decode")
    partial_label = "`listen --wav` emitted rolling partials before the final"
    listen_dir = listen_key = None
    if gate.available():
        from fermion._speech import fetch as _fetch2
        if installed_key is not None:
            pin = _catalog.SPEECH_PROFILES[installed_key]
            listen_dir = _fetch2.profile_dir(pin["repo"], pin["unpack_dir"])
            listen_key = installed_key
        elif src is not None:
            # No downloaded profile, but a dev tree holds unpacked builds:
            # run against those rather than skipping on the machines that
            # care most.
            for key in ("audio6", "parity"):
                cand = (src / "local_stt"
                        / _catalog.SPEECH_PROFILES[key]["unpack_dir"])
                if _fetch2.is_model_dir(cand):
                    listen_dir, listen_key = cand, key
                    break
    fixture = None
    if src is not None:
        cand = (src / "local_stt/release_ship/github_repo/samples"
                / "1089-134691-0000.flac")
        fixture = cand if cand.is_file() else None
    if not gate.available():
        why = gate.why_not() or "speech unavailable"
        skip(identity_label, why)
        skip(partial_label, why)
    elif listen_dir is None or fixture is None:
        why = ("no model on this machine" if listen_dir is None
               else "no sample audio in the source tree")
        skip(identity_label, why)
        skip(partial_label, why)
    else:
        from fermion._speech import engine as _live_engine
        from fermion._speech import live
        speech2 = _live_engine.load(
            listen_dir, profile=listen_key,
            backend=_catalog.SPEECH_PROFILES[listen_key]["backend"],
            quiet=True)
        partials: list[str] = []
        finals: list[str] = []
        session = live.LiveSession(speech2, on_partial=partials.append,
                                   on_final=finals.append)
        # realtime=False: the identical code path minus only the pacing sleep,
        # which would add the clip's full duration to every selftest run.
        streamed = live.run_wav(session, fixture, realtime=False)
        batch, _, _ = speech2.transcribe(fixture)
        check(f"{identity_label} ({streamed!r})",
              bool(streamed) and streamed == batch)
        check(f"{partial_label} ({len(partials)} partials, ending "
              f"{partials[-1][-30:]!r})" if partials else
              f"{partial_label} (none came out)",
              len(partials) >= 2 and streamed == " ".join(finals))

        # --hotwords (0.1.25): an EMPTY list must leave the decode untouched,
        # byte for byte; a non-empty list must be reported by describe() and
        # must still produce a transcript through the same call.
        speech2.set_hotwords([])
        again, _, _ = speech2.transcribe(fixture)
        check("`--hotwords` with no words is the unchanged decode, byte for byte",
              again == batch and speech2.describe()["hotwords"] == 0)
        n = speech2.set_hotwords(["Kushal", "Sigil", "Kushal"], 4.0)
        biased, _, _ = speech2.transcribe(fixture)
        check(f"`--hotwords` reports its words and still decodes ({n} words, "
              f"{biased[:40]!r}...)",
              n == 2 and speech2.describe()["hotwords"] == 2
              and speech2.describe()["hotword_strength"] == 4.0 and bool(biased))
        speech2.set_hotwords([])
        # Hotword policy v4 (0.1.25): the re-fire block is keyed on the word.
        # After the no-space form "Mr." completes, the leading-space form's
        # root " Mr" must not be boosted (v3 boosted it: "Mr. Mr."); after
        # another word it is eligible again. Tokenizer-only, on the loaded
        # engine's own tokenizer; v3 is checked too so the test is shown to
        # discriminate.
        ov = sys.modules[type(speech2.model).__module__]
        tok = speech2.model.model._tokenizer
        import mlx.core as mx
        import numpy as np

        def _ids(s):
            return tok.encode(s, add_special_tokens=False)

        def _boosted(bias, hist):
            out = bias(mx.array(np.array(hist, dtype=np.int32)),
                       mx.zeros((1, 151936), dtype=mx.float32))
            mx.eval(out)
            o = np.array(out)[0]
            return {int(i) for i in np.nonzero(o)[0]}
        asr, mr, smr, and_ = _ids("<asr_text>")[0], _ids("Mr."), _ids(" Mr."), _ids(" and")
        v3 = ov.HotwordBias(tok, ["Mr.", "Kensbok"], strength=4.0, policy="v3")
        v4 = ov.HotwordBias(tok, ["Mr.", "Kensbok"], strength=4.0, policy="v4")
        check("hotword v4: after 'Mr.' completes, ' Mr' is not boosted again "
              "(v3 did — the 'Mr. Mr.' residue)",
              smr[0] in _boosted(v3, [asr] + mr)
              and smr[0] not in _boosted(v4, [asr] + mr))
        check("hotword v4: other words stay boosted, and ' Mr' is eligible "
              "again after another word",
              _ids(" Kensbok")[0] in _boosted(v4, [asr] + mr)
              and smr[0] in _boosted(v4, [asr] + mr + and_))

    # ------------------------------- 10. /v1/audio/stream — WS live endpoint
    # The codec checks run EVERYWHERE (they are pure bytes); the live stream
    # checks are gated exactly like sections 8/9.
    print("\n## 10. /v1/audio/stream — hand-rolled RFC 6455 + the live path")
    from fermion import _ws

    # (a) the opening handshake, against RFC 6455 §1.3's worked example — a
    # known-answer test, so the digest cannot vouch for itself.
    check("the handshake digest matches RFC 6455's worked example",
          _ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==")
          == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    # (b) the frame codec, one round-trip per payload-length class, masked
    # like a client's frames are.
    for label, payload in (("7-bit", b"hi"), ("16-bit", os.urandom(300)),
                           ("64-bit", os.urandom(70_000))):
        frame = _ws.encode_frame(_ws.BINARY, payload, mask=True)
        fin, op, out = _ws.read_frame(io.BytesIO(frame), require_mask=True)
        check(f"a masked {label}-length frame round-trips byte-for-byte",
              fin and op == _ws.BINARY and out == payload)
    fin, op, out = _ws.read_frame(
        io.BytesIO(_ws.encode_frame(_ws.TEXT, b"x", fin=False)))
    check("FIN comes back honestly (fragmentation is refused a layer up)",
          fin is False and op == _ws.TEXT and out == b"x")
    # Hand-built wire bytes, so the decoder is not graded by its own encoder.
    raw16 = bytes([0x82, 126]) + (300).to_bytes(2, "big") + bytes(300)
    fin, op, out = _ws.read_frame(io.BytesIO(raw16))
    check("a hand-built 126-class unmasked frame decodes",
          fin and op == _ws.BINARY and out == bytes(300))
    try:
        _ws.read_frame(io.BytesIO(raw16), require_mask=True)
        ok = False
    except _ws.ProtocolError:
        ok = True
    check("an unmasked client frame is a protocol error (RFC 6455 §5.1)", ok)
    try:
        _ws.read_frame(io.BytesIO(bytes([0x88, 126, 0, 200]) + bytes(200)))
        ok = False
    except _ws.ProtocolError:
        ok = True
    check("a control frame over 125 bytes is a protocol error", ok)
    code, reason = _ws.parse_close(_ws.close_payload(1013, "busy"))
    check("close payloads round-trip (code + reason)",
          code == 1013 and reason == "busy")

    # (c) the live endpoint, end to end, through a real socket.
    ws_labels = (
        "the server completes the WS handshake (101 + correct accept key)",
        "the WS stream produced partial hypotheses before the final",
        "the WS `done` transcript equals the file endpoint's, byte for byte",
        "finals are numbered from 1",
        "a second concurrent stream is refused in-band, not queued",
        "a non-16 kHz config is refused with an in-band error",
        "a fragmented message is refused cleanly",
        "--api-key: no key -> 401; ?api_key= and the Bearer header both pass",
        "serve resolves a local Phonon directory as speech (transcribe's rule)",
    )
    if not gate.available() or listen_dir is None or fixture is None:
        why = (gate.why_not() or ("no model on this machine"
                                  if listen_dir is None
                                  else "no sample audio in the source tree"))
        for lbl in ws_labels:
            skip(lbl, why)
    else:
        import threading
        from http.server import ThreadingHTTPServer

        from fermion import server as _server_mod
        from fermion._speech.engine import read_audio
        from fermion.server import Handler, SpeechEngine

        # 0.1.17 sent a local speech directory down the LLM serve path, where
        # it died looking for a TRTC container; the dispatch now applies
        # transcribe's rule. A TRTC-shaped directory must NOT match it.
        check(ws_labels[8],
              _server_mod._is_local_speech_dir(str(listen_dir))
              and not _server_mod._is_local_speech_dir(str(WHEEL_ROOT)))

        class _SArgs:
            model = str(listen_dir)
            served_model_name = None
            api_key = None
            cors = False
            host = "127.0.0.1"
            port = 0
        try:
            eng10 = SpeechEngine(_SArgs())
            httpd10 = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            httpd10.engine = eng10
            threading.Thread(target=httpd10.serve_forever,
                             daemon=True).start()
            port10 = httpd10.server_address[1]

            # What `fermion transcribe` says about this file, via the same
            # loaded engine (identical decode call, section 8 ties it to the
            # batch CLI output).
            ref_text = eng10.transcribe_bytes(fixture.read_bytes(),
                                              fixture.name).text
            audio = read_audio(fixture)

            a = WsClient(port10)
            check(ws_labels[0],
                  " 101 " in a.status
                  and a.headers.get("sec-websocket-accept")
                  == _ws.accept_key(a.key))
            a.send(0x1, json.dumps({"sample_rate": 16000,
                                    "format": "pcm_f32le"}).encode())
            # Stream the FIRST HALF in 0.5 s chunks, then wait for a partial:
            # once one arrives, the session provably holds the one-stream
            # slot, so the busy check below cannot race the handshake.
            step = 8000                                     # 0.5 s at 16 kHz
            half = (len(audio) // 2 // step) * step
            for i in range(0, half, step):
                a.send(0x2, audio[i:i + step].tobytes())
            gen = a.messages()
            first = next(gen)

            b = WsClient(port10)
            got_b = list(b.messages())
            b.close()
            check(f"{ws_labels[4]} ({got_b and got_b[0].get('message')!r})",
                  len(got_b) == 1 and got_b[0]["type"] == "error"
                  and "busy" in got_b[0]["message"])

            for i in range(half, len(audio), step):
                a.send(0x2, audio[i:i + step].tobytes())
            a.send(0x1, b'{"type":"end"}')
            events = [first] + list(gen)
            a.close()
            kinds = [e["type"] for e in events]
            done = next((e for e in events if e["type"] == "done"), None)
            check(f"{ws_labels[1]} ({kinds.count('partial')} partials, "
                  f"{kinds.count('final')} final)",
                  "partial" in kinds and "final" in kinds
                  and kinds.index("partial") < kinds.index("final"))
            check(f"{ws_labels[2]} ({(done or {}).get('text')!r})",
                  done is not None and done["text"] == ref_text
                  and bool(ref_text))
            check(ws_labels[3],
                  [e["segment"] for e in events if e["type"] == "final"]
                  == list(range(1, kinds.count("final") + 1)))

            c = WsClient(port10)
            c.send(0x1, b'{"sample_rate": 44100}')
            got_c = list(c.messages())
            c.close()
            check(ws_labels[5],
                  bool(got_c) and got_c[0]["type"] == "error"
                  and "16000" in got_c[0]["message"])

            d = WsClient(port10)
            d.send(0x1, b'{"sample_rate":', fin=False)      # half a message
            got_d = list(d.messages())
            d.close()
            check(ws_labels[6],
                  bool(got_d) and got_d[0]["type"] == "error"
                  and "fragment" in got_d[0]["message"])

            eng10.args.api_key = "sekrit-stream-key"
            try:
                e1 = WsClient(port10)
                no_auth = e1.status
                e1.close()
                e2 = WsClient(
                    port10,
                    path="/v1/audio/stream?api_key=sekrit-stream-key")
                ok_query = " 101 " in e2.status
                e2.send(0x1, b'{"type":"end"}')
                got_e2 = list(e2.messages())
                e2.close()
                e3 = WsClient(
                    port10, header="Authorization: Bearer sekrit-stream-key")
                ok_hdr = " 101 " in e3.status
                e3.send(0x1, b'{"type":"end"}')
                list(e3.messages())
                e3.close()
            finally:
                eng10.args.api_key = None
            check(ws_labels[7],
                  " 401 " in no_auth and ok_query and ok_hdr
                  and bool(got_e2) and got_e2[-1]["type"] == "done")
            httpd10.shutdown()
            httpd10.server_close()
        except Exception as exc:                             # noqa: BLE001
            check(f"the WS live endpoint works end to end ({exc!r})", False)

    # ------------------------------------------- 11. long-audio segmentation
    # Pure array checks, so they run everywhere. The contract: up to 35 s is
    # ONE window (the published single-utterance path, untouched); longer
    # audio is contiguous 25-35 s windows that prefer a pause; no pause in
    # the band means a cut at 30 s.
    print("\n## 11. long-audio segmentation (fermion/_speech/segment.py)")
    import numpy as np

    from fermion._speech import segment
    sr = segment.SAMPLE_RATE
    rng = np.random.default_rng(0)
    noise = lambda s: (rng.standard_normal(int(s * sr)) * 0.1).astype(np.float32)  # noqa: E731
    check("35 s of audio is exactly one window (single-shot path unchanged)",
          segment.plan(noise(35.0)) == [(0, 35 * sr)])
    check("an empty array is one empty window", segment.plan(noise(0)) == [(0, 0)])
    plan = segment.plan(noise(100.0))
    check(f"100 s of continuous speech cuts at 30/60/90 s ({plan})",
          plan == [(0, 30 * sr), (30 * sr, 60 * sr), (60 * sr, 90 * sr),
                   (90 * sr, 100 * sr)])
    a = noise(100.0)
    a[28 * sr:29 * sr] = 0.0                      # a 1 s pause at 28-29 s
    a[62 * sr:int(63.5 * sr)] = 0.001             # room tone at 62-63.5 s
    plan = segment.plan(a)
    check(f"cuts land in the middle of the pauses (28.5 s, 62.75 s) ({plan})",
          plan[0] == (0, int(28.5 * sr)) and plan[1][1] == int(62.75 * sr))
    check("windows are contiguous and cover the whole array",
          plan[0][0] == 0 and plan[-1][1] == a.size
          and all(plan[i][1] == plan[i + 1][0] for i in range(len(plan) - 1)))
    check("every window but the last is 25-35 s",
          all(25 * sr <= b - a_ <= 35 * sr for a_, b in plan[:-1]))
    # transcribe_windows: the decode callback sees each window once, the
    # texts join with single spaces, and a window that spends its whole
    # budget flags `truncated`.
    seen = []

    def fake(window):
        seen.append(window.size)
        return f"w{len(seen)}", 10 if len(seen) < 3 else 24, 24
    out = segment.transcribe_windows(a, fake)
    check("transcribe_windows decodes each window once, in order",
          seen == [b - a_ for a_, b in plan])
    check(f"texts join with single spaces ({out.text!r})",
          out.text == " ".join(f"w{i + 1}" for i in range(len(plan))))
    check("segments carry start/end seconds per window",
          [s["start"] for s in out.segments] == [round(a_ / sr, 3) for a_, _ in plan]
          and out.segments[-1]["end"] == round(a.size / sr, 3))
    check("a window that spends its whole budget sets truncated=True",
          out.truncated is True and out.generation_tokens == 10 + 10 + 24 * (len(plan) - 2))
    out2 = segment.transcribe_windows(noise(10.0), lambda w: ("x", 5, 90))
    check("a short clip is one single-shot window and not truncated",
          out2.text == "x" and len(out2.segments) == 1 and out2.truncated is False)
    check("silence returns an empty transcript without a decode",
          segment.transcribe_windows(np.zeros(sr * 40, np.float32),
                                     lambda w: 1 / 0).text == "")

    # ------------------------- 12. 0.1.25: lazy import, shield, cache lock
    print("\n## 12. 0.1.25 — lazy heavy imports, the shield still fires, "
          "the cache lock serialises processes")
    import tempfile
    py = [sys.executable, "-c"]
    heavy_probe = (
        "import sys, time; t=time.perf_counter(); import fermion; "
        "dt=time.perf_counter()-t; "
        "heavy=sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0]"
        " in ('torch','transformers','torchaudio','hf_ternary','mlx')}"
        " | {m for m in sys.modules if m.startswith('fermion._vendor')}); "
        "print(json.dumps({'ms': dt*1000, 'heavy': heavy}))")
    p = subprocess.run(py + ["import json; " + heavy_probe],
                       capture_output=True, text=True, timeout=120)
    probe = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {}
    check(f"`import fermion` loads no torch/transformers/MLX/hf_ternary "
          f"({probe.get('ms', 0):.0f} ms, heavy={probe.get('heavy')})",
          p.returncode == 0 and probe.get("heavy") == [])
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        p = subprocess.run(exe + ["--version"], capture_output=True, text=True,
                           timeout=120)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    check(f"`fermion --version` wall < 0.3 s (best of 3: {best:.2f} s; "
          f"interpreter start included)", p.returncode == 0 and best < 0.3)
    p = subprocess.run(py + [
        "import sys, fermion; h = fermion.read_header; "
        "from transformers import CONFIG_MAPPING; "
        "print('trtc_v4' in CONFIG_MAPPING and 'torch' in sys.modules)"],
        capture_output=True, text=True, timeout=300)
    check("the first language-model symbol imports the loader and registers "
          "trtc_v4 with transformers", p.returncode == 0
          and p.stdout.strip().endswith("True"))
    # The torchaudio shield: a synthetic torchaudio that raises the DLVM's
    # OSError. `import fermion` must not touch it; the first heavy load must
    # mask it, record the note, and the CLI must print that note.
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp)
        (fake / "torchaudio").mkdir()
        (fake / "torchaudio" / "__init__.py").write_text(
            'raise OSError("Could not load this library: '
            'fake/_torchaudio.abi3.so: undefined symbol: torch_library_impl")\n')
        (fake / "torchaudio-2.11.0.dist-info").mkdir()
        (fake / "torchaudio-2.11.0.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: torchaudio\nVersion: 2.11.0\n")
        env = dict(os.environ, PYTHONPATH=str(fake) + os.pathsep
                   + os.environ.get("PYTHONPATH", ""))
        p = subprocess.run(py + [
            "import sys, fermion; from fermion import _compat; "
            "a = 'torchaudio' in sys.modules; b = _compat.TORCHAUDIO_NOTE; "
            "_compat.announce_notes(); _compat.prepare_torch_stack(); "
            "h = fermion.read_header; from transformers import CONFIG_MAPPING; "
            "print(json.dumps([a, b, sys.modules.get('torchaudio', 1) is None, "
            "bool(_compat.TORCHAUDIO_NOTE), 'trtc_v4' in CONFIG_MAPPING]))"
            .replace("import sys, fermion", "import sys, json, fermion")],
            capture_output=True, text=True, timeout=300, env=env)
        got = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
        check(f"broken torchaudio: untouched by `import fermion`, masked on "
              f"the first heavy load, loader still works ({got})",
              got == [False, None, True, True, True])
        check("...and the note is printed once on stderr in CLI mode",
              p.returncode == 0
              and p.stderr.count("torchaudio 2.11.0 is installed but does not load") == 1)
        # Adversarial (found on the DLVM image 2026-09-07): the program imports
        # transformers BEFORE fermion's first heavy use. transformers 5 caches
        # `is_torchaudio_available()` (lru_cache) as True at its own import;
        # the shield must reset that cache, or `transformers.audio_utils`
        # later runs `import torchaudio` into the None sentinel.
        p = subprocess.run(py + [
            "import sys, json, fermion, transformers; "
            "from transformers.utils import import_utils as iu; "
            "before = iu.is_torchaudio_available(); "
            "from fermion import _compat; _compat.prepare_torch_stack(); "
            "after = iu.is_torchaudio_available(); "
            "import transformers.audio_utils; "
            "print(json.dumps([before, after, bool(_compat.TORCHAUDIO_NOTE)]))"],
            capture_output=True, text=True, timeout=300, env=env)
        got = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
        check(f"broken torchaudio with transformers imported FIRST: the cached "
              f"probe is reset and audio_utils imports ({got}; "
              f"rc={p.returncode})", got == [True, False, True])
    # The cache lock: two processes, 1 s critical sections, no overlap, the
    # waiter is told once. (Adversarial: without the lock they DO overlap.)
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "lock.log"
        code = (
            "import time, os, sys; from fermion._lock import CacheLock\n"
            f"L = {str(log)!r}; lock = sys.argv[1] == 'lock'\n"
            "def w(s): open(L, 'a').write(s + '\\n')\n"
            "ctx = CacheLock(L + '.lock', what='the test model', note=lambda m: w('NOTE ' + m))\n"
            "if lock: ctx.__enter__()\n"
            "w(f'ENTER {os.getpid()} {time.time():.4f}'); time.sleep(1.0); w(f'EXIT {os.getpid()} {time.time():.4f}')\n"
            "if lock: ctx.__exit__(None, None, None)\n")

        def contend(mode):
            log.write_text("")
            procs = [subprocess.Popen(py + [code, mode]) for _ in range(2)]
            rcs = [q.wait(timeout=60) for q in procs]
            iv = {}
            notes = 0
            for line in log.read_text().splitlines():
                if line.startswith("NOTE"):
                    notes += 1
                    continue
                kind, pid, t = line.split()
                iv.setdefault(pid, [None, None])[0 if kind == "ENTER" else 1] = float(t)
            a, b = list(iv.values())
            overlap = not (a[1] <= b[0] or b[1] <= a[0])
            return rcs, overlap, notes, len(iv)
        rcs, overlap, notes, n = contend("lock")
        check(f"cache lock: two processes never overlap ({n} entries, "
              f"waiter told {notes}x)", rcs == [0, 0] and not overlap and n == 2 and notes == 1)
        rcs, overlap, notes, n = contend("nolock")
        check("...and the same test WITHOUT the lock does overlap (the check "
              "can fail)", rcs == [0, 0] and overlap)

    print(f"\n{ran - failures}/{ran} speech checks pass"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

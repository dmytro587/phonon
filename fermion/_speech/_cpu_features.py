"""CPU feature detection for the Phonon-2 CPU engine, BEFORE any kernel binary is loaded (0.2.5).

Why this exists: the aarch64 kernel binaries are built for armv8.2-a+dotprod and the x86-64 encoder binary for AVX2.
A binary that assumes an instruction the processor lacks faults on first use (SIGILL), which Python cannot catch.  So the
tier is chosen here from what the processor and OS actually provide, with methods that need no kernel code:

  aarch64  Linux: getauxval(AT_HWCAP / AT_HWCAP2) through the C library (asimddp = dotprod, i8mm, sve, atomics = LSE);
                  /proc/cpuinfo "Features" is only the fallback because it is the HOST's under user-mode emulation.
           macOS: every Apple silicon core implements dotprod and LSE; i8mm from sysctl hw.optional.arm.FEAT_I8MM.
  x86-64   the cpuid and xgetbv instructions themselves, executed from a page of machine code mapped at runtime (the
           same trick the `cpuid` package uses), so the answer is the processor's and the OS's XCR0 state, not a text file;
           /proc/cpuinfo (Linux), sysctl hw.optional.* (Intel Macs) or IsProcessorFeaturePresent (Windows) only when that
           page cannot be mapped.

If nothing can be determined, the answer is "no optional feature" and the baseline tier is chosen: unknown never means
"assume the fast path".  `plan()` maps the features to the kernel tier and the binary names the loader must use, and
`describe()` prints both (`fermion describe`).
"""
from __future__ import annotations

import ctypes
import os
import platform
import struct
import sys

# --- feature bit names (the C side's phonon2_cpu_features() uses the same bits) --------------------------------------
BITS = {"neon": 1 << 0, "dotprod": 1 << 1, "i8mm": 1 << 2, "sve": 1 << 3, "lse": 1 << 4,
        "sse4.1": 1 << 8, "avx2": 1 << 9, "avx512bw": 1 << 10, "avx512-vnni": 1 << 11, "amx-int8": 1 << 12}

#: kernel ids reported by phonon2_cpu_kernel_in_use()
KERNEL_NAMES = {1: "scalar", 2: "neon-dotprod", 3: "avx2", 4: "avx512-vnni", 5: "neon-i8mm", 6: "avx512bw", 7: "amx-int8",
                8: "neon (armv8.0, no dotprod)", 9: "sse4.1"}

SUPPORTED_HOSTS = ("Linux x86-64 (SSE4.1 or newer; AVX2, AVX-512 VNNI and AMX tiers run faster)",
                   "Linux 64-bit Arm (NEON; dotprod and i8mm tiers run faster)",
                   "Windows x86-64 (SSE4.1 or newer; AVX2 tier runs faster)",
                   "Apple silicon Macs",
                   "Intel Macs (macOS x86-64, SSE4.1 or newer; AVX2 tier runs faster)")


def arch() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    return m


# --- aarch64 -----------------------------------------------------------------------------------------------------------
_AT_HWCAP, _AT_HWCAP2 = 16, 26
_HWCAP_ASIMD, _HWCAP_ATOMICS, _HWCAP_ASIMDDP, _HWCAP_SVE = 1 << 1, 1 << 8, 1 << 20, 1 << 22
_HWCAP2_I8MM = 1 << 13


def _aarch64() -> tuple[dict, str]:
    f = {"neon": True, "dotprod": False, "i8mm": False, "sve": False, "lse": False}
    sysname = platform.system()
    if sysname == "Darwin":
        f["dotprod"] = True; f["lse"] = True
        try:
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_I8MM"], capture_output=True, text=True, timeout=5).stdout.strip()
            f["i8mm"] = out == "1"
        except Exception:
            pass
        return f, "apple-silicon"
    if sysname == "Linux":
        try:
            libc = ctypes.CDLL(None)
            getauxval = libc.getauxval
            getauxval.restype = ctypes.c_ulong; getauxval.argtypes = (ctypes.c_ulong,)
            h1, h2 = getauxval(_AT_HWCAP), getauxval(_AT_HWCAP2)
            if h1:
                f["neon"] = bool(h1 & _HWCAP_ASIMD); f["dotprod"] = bool(h1 & _HWCAP_ASIMDDP); f["sve"] = bool(h1 & _HWCAP_SVE)
                f["lse"] = bool(h1 & _HWCAP_ATOMICS); f["i8mm"] = bool(h2 & _HWCAP2_I8MM)
                return f, "hwcap"
        except Exception:
            pass
        try:
            feats = set()
            for line in open("/proc/cpuinfo"):
                if line.startswith("Features"):
                    feats |= set(line.split(":", 1)[1].split())
            if feats:
                f["dotprod"] = "asimddp" in feats; f["i8mm"] = "i8mm" in feats; f["sve"] = "sve" in feats; f["lse"] = "atomics" in feats
                return f, "cpuinfo"
        except Exception:
            pass
    return f, "unknown"


# --- x86-64 ------------------------------------------------------------------------------------------------------------
# cpuid(leaf, subleaf, out[4]) and xgetbv(out[2]) as position-independent machine code, SysV (Linux/macOS) and Win64.
_CPUID_SYSV = bytes.fromhex("53" "4989d0" "89f8" "89f1" "0fa2" "418900" "41895804" "41894808" "4189500c" "5b" "c3")   # rdx is clobbered by cpuid: keep the pointer in r8
_CPUID_WIN = bytes.fromhex("53" "89c8" "89d1" "0fa2" "418900" "41895804" "41894808" "4189500c" "5b" "c3")
_XGETBV_SYSV = bytes.fromhex("31c9" "0f01d0" "8907" "895704" "c3")
_XGETBV_WIN = bytes.fromhex("4989c8" "31c9" "0f01d0" "418900" "41895004" "c3")


class _CodePage:
    """One executable page holding the two stubs; released on close()."""

    def __init__(self):
        win = platform.system() == "Windows"
        code = (_CPUID_WIN if win else _CPUID_SYSV); self.xoff = 64
        code = code + b"\xcc" * (self.xoff - len(code)) + (_XGETBV_WIN if win else _XGETBV_SYSV)
        self._win = win; self._mm = None; self.base = None
        if win:
            k32 = ctypes.windll.kernel32
            k32.VirtualAlloc.restype = ctypes.c_void_p; k32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32)
            self.base = k32.VirtualAlloc(None, 4096, 0x3000, 0x40)                     # MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE
            if not self.base:
                raise OSError("VirtualAlloc failed")
            ctypes.memmove(self.base, code, len(code))
        else:
            import mmap
            self._mm = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
            self._mm.write(code)
            self.base = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        proto = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
        self._cpuid = proto(self.base)
        self._xgetbv = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.base + self.xoff)

    def cpuid(self, leaf: int, sub: int = 0) -> tuple[int, int, int, int]:
        buf = (ctypes.c_uint32 * 4)()
        self._cpuid(leaf, sub, ctypes.addressof(buf))
        return tuple(buf)

    def xgetbv(self) -> int:
        buf = (ctypes.c_uint32 * 2)()
        self._xgetbv(ctypes.addressof(buf))
        return buf[0] | (buf[1] << 32)

    def close(self):
        try:
            self._cpuid = self._xgetbv = None
            if self._win and self.base:
                ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.base), 0, 0x8000)
            elif self._mm is not None:
                self._mm.close()
        except Exception:
            pass


def _x86_cpuid() -> dict | None:
    """Features from cpuid + xgetbv; None when the executable page cannot be set up (hardened runtimes)."""
    page = _CodePage()
    try:
        max_leaf = page.cpuid(0)[0]
        a1, b1, c1, d1 = page.cpuid(1)
        f = {"sse4.1": bool(c1 >> 19 & 1), "sse4.2": bool(c1 >> 20 & 1), "ssse3": bool(c1 >> 9 & 1), "fma": bool(c1 >> 12 & 1),
             "avx": bool(c1 >> 28 & 1), "osxsave": bool(c1 >> 27 & 1), "avx2": False, "avx512f": False, "avx512bw": False,
             "avx512-vnni": False, "amx-tile": False, "amx-int8": False}
        xcr0 = page.xgetbv() if f["osxsave"] else 0
        os_avx = (xcr0 & 0x6) == 0x6; os_avx512 = os_avx and (xcr0 & 0xe0) == 0xe0; os_amx = (xcr0 & 0x60000) == 0x60000
        if max_leaf >= 7:
            a7, b7, c7, d7 = page.cpuid(7, 0)
            f["avx2"] = bool(b7 >> 5 & 1) and os_avx and f["avx"]
            f["avx512f"] = bool(b7 >> 16 & 1) and os_avx512
            f["avx512bw"] = bool(b7 >> 30 & 1) and f["avx512f"]
            f["avx512-vnni"] = bool(c7 >> 11 & 1) and f["avx512bw"]
            f["amx-tile"] = bool(d7 >> 24 & 1) and os_amx
            f["amx-int8"] = bool(d7 >> 25 & 1) and f["amx-tile"]
        f["avx"] = f["avx"] and os_avx
        # the kernels' AMX tier also needs the Linux XTILEDATA permission; the C side requests it, so only Linux counts
        if platform.system() != "Linux":
            f["amx-int8"] = False
        f["xcr0"] = xcr0
        return f
    finally:
        page.close()


def _x86_fallback() -> tuple[dict, str]:
    f = {"sse4.1": False, "sse4.2": False, "ssse3": False, "fma": False, "avx": False, "avx2": False, "avx512f": False,
         "avx512bw": False, "avx512-vnni": False, "amx-tile": False, "amx-int8": False}
    if platform.system() == "Linux":
        try:
            flags = set()
            for line in open("/proc/cpuinfo"):
                if line.startswith("flags"):
                    flags |= set(line.split(":", 1)[1].split())
            if flags:
                f.update({"sse4.1": "sse4_1" in flags, "sse4.2": "sse4_2" in flags, "ssse3": "ssse3" in flags, "fma": "fma" in flags,
                          "avx": "avx" in flags, "avx2": "avx2" in flags, "avx512f": "avx512f" in flags, "avx512bw": "avx512bw" in flags,
                          "avx512-vnni": "avx512_vnni" in flags, "amx-tile": "amx_tile" in flags, "amx-int8": "amx_int8" in flags})
                return f, "cpuinfo"
        except Exception:
            pass
    if platform.system() == "Darwin":                  # Intel Mac whose process cannot map an executable page: sysctl hw.optional.*
        try:
            import subprocess
            def opt(name):
                r = subprocess.run(["sysctl", "-n", f"hw.optional.{name}"], capture_output=True, text=True, timeout=3)
                return r.returncode == 0 and r.stdout.strip() == "1"
            f.update({"sse4.1": opt("sse4_1"), "sse4.2": opt("sse4_2"), "ssse3": opt("supplementalsse3"), "fma": opt("fma"), "avx": opt("avx1_0"),
                      "avx2": opt("avx2_0"), "avx512f": opt("avx512f"), "avx512bw": opt("avx512bw"), "avx512-vnni": opt("avx512vnni")})
            return f, "sysctl"
        except Exception:
            pass
    if platform.system() == "Windows":
        try:
            k32 = ctypes.windll.kernel32; pf = k32.IsProcessorFeaturePresent
            f.update({"ssse3": bool(pf(36)), "sse4.1": bool(pf(37)), "sse4.2": bool(pf(38)), "avx": bool(pf(39)), "avx2": bool(pf(40)),
                      "avx512f": bool(pf(41))})
            return f, "IsProcessorFeaturePresent"
        except Exception:
            pass
    return f, "unknown"


def _x86() -> tuple[dict, str]:
    if os.environ.get("FERMION_CPU_FEATURES_NO_CPUID") != "1":
        try:
            f = _x86_cpuid()
            if f is not None:
                return f, "cpuid"
        except Exception:
            pass
    return _x86_fallback()


# --- public -------------------------------------------------------------------------------------------------------------
def detect() -> dict:
    """{"arch", "system", "method", "features": {...}, "cpu": model string or None}."""
    a = arch()
    if a == "aarch64":
        f, method = _aarch64()
    elif a == "x86_64":
        f, method = _x86()
    else:
        f, method = {}, "unsupported-arch"
    return {"arch": a, "system": platform.system(), "method": method, "features": f, "cpu": cpu_model()}


def cpu_model() -> str | None:
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith(("model name", "Hardware", "CPU part")):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() or None
        elif platform.system() == "Windows":
            return platform.processor() or None
    except Exception:
        pass
    return None


def tier(det: dict) -> str:
    """The kernel tier for these features: the name also says which binaries the loader must use (see plan())."""
    f = det["features"]
    if det["arch"] == "aarch64":
        if f.get("i8mm") and f.get("dotprod"):
            return "i8mm"
        if f.get("dotprod"):
            return "dotprod"
        return "neon"                                            # armv8.0 baseline binaries (-base)
    if det["arch"] == "x86_64":
        if f.get("amx-int8") and f.get("avx512-vnni"):
            return "amx"
        if f.get("avx512-vnni"):
            return "avx512-vnni"
        if f.get("avx512bw"):
            return "avx512bw"
        if f.get("avx2") and f.get("fma", True):
            return "avx2"
        if f.get("sse4.1"):
            return "sse4.1"                                      # baseline encoder binary (-base), SSE4.1 kernel 9
        return "scalar"                                          # baseline encoder binary (-base), scalar kernel 1
    return "unsupported"


#: tiers that run the NATIVE binaries (the ones 0.2.4 shipped, byte for byte); every other tier takes the -base binaries
NATIVE_TIERS = {"i8mm", "dotprod", "amx", "avx512-vnni", "avx512bw", "avx2"}
#: tiers a processor may be asked to run below its own (FERMION_CPU_TIER): never a higher one, which would fault
_X86_ORDER = ("amx", "avx512-vnni", "avx512bw", "avx2", "sse4.1", "scalar"); _ARM_ORDER = ("i8mm", "dotprod", "neon")
_LOWER_TIERS = {**{t: _X86_ORDER[i + 1:] for i, t in enumerate(_X86_ORDER)}, **{t: _ARM_ORDER[i + 1:] for i, t in enumerate(_ARM_ORDER)}}


def plan(det: dict | None = None) -> dict:
    """Which binaries this processor may load.  `base` True = the -base variants; `refusal` = one line when nothing fits."""
    det = det or detect()
    t = tier(det)
    forced = os.environ.get("FERMION_CPU_TIER", "").strip().lower()      # diagnostic (0.2.8): run a LOWER tier's binaries on purpose
    if forced and forced != t and forced in _LOWER_TIERS.get(t, ()):
        t = forced
    base = t not in NATIVE_TIERS and t != "unsupported"
    out = {"tier": t, "base": base, "detect": det, "refusal": None}
    if t == "unsupported":
        out["refusal"] = (f"the Phonon-2 CPU kernels do not run on {det['system']}/{platform.machine()}; supported hosts: "
                          + "; ".join(SUPPORTED_HOSTS) + ". FERMION_P2_CPU=fp32 runs the full-precision path.")
    return out


def missing_line(det: dict, what: str) -> str:
    """One line naming the missing feature and the supported hosts (used when a base binary is not shipped)."""
    f = det["features"]
    if det["arch"] == "aarch64":
        miss = "dotprod (asimddp)"
    else:
        miss = "AVX2" if not f.get("avx2") else "SSE4.1"
    return (f"Phonon-2 CPU engine: this {det['arch']} processor ({det.get('cpu') or 'model unknown'}) has no {miss} and the "
            f"{what} for it is not shipped in this package; supported hosts: {'; '.join(SUPPORTED_HOSTS)}. "
            f"FERMION_P2_CPU=fp32 runs the full-precision path.")


def features_word(det: dict) -> int:
    """The C-side bitmask for these features (compare with phonon2_cpu_features())."""
    w = 0
    for name, bit in BITS.items():
        if det["features"].get(name):
            w |= bit
    return w


def describe(load: bool = True) -> dict:
    """Everything `fermion describe` prints: host, features, method, tier, the binaries and (optionally) the loaded
    library's own view (phonon2_cpu_features / kernel_in_use), which is safe to call because the tier was chosen for it."""
    p = plan(); det = p["detect"]
    out = {"host": {"system": det["system"], "machine": platform.machine(), "cpu": det["cpu"], "python": sys.version.split()[0],
                    "cores": os.cpu_count()},
           "features": det["features"], "detection": det["method"], "tier": p["tier"], "baseline_binaries": p["base"],
           "refusal": p["refusal"], "binaries": {}, "library": None}
    try:
        from fermion._speech import _engine_phonon2_cpu as pk
        pins = pk._pins()
        for key, prefix in (("encoder", "phonon2_enc"), ("gemm", "phonon2_cpu"), ("tdt", "phonon2_tdt")):
            b = p["base"] and (key == "encoder" or (key == "tdt" and det["arch"] in pk._BASE_NEEDS_TDT))
            name = pk._platform_name(prefix, base=b)
            if name is None:
                continue
            path = pk.HERE / name
            out["binaries"][key] = {"file": name, "shipped": path.is_file(), "pinned": name in pins}
        out["selected"] = {"gemm_encoder": pk.default_library().name, "tdt": pk.default_tdt_library().name}
        if load and not p["refusal"]:
            try:
                lib = pk.load_library()
                info = {"kernel_id": int(lib.phonon2_cpu_kernel_in_use()), "kernel": KERNEL_NAMES.get(int(lib.phonon2_cpu_kernel_in_use()), "?")}
                if hasattr(lib, "phonon2_cpu_features"):
                    lib.phonon2_cpu_features.restype = ctypes.c_uint; lib.phonon2_cpu_build_features.restype = ctypes.c_uint
                    info["c_features"] = f"0x{lib.phonon2_cpu_features():x}"; info["c_build_features"] = f"0x{lib.phonon2_cpu_build_features():x}"
                    info["python_features"] = f"0x{features_word(det):x}"
                out["library"] = info
            except Exception as exc:
                out["library"] = {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        out["binaries"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out

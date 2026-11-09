#!/usr/bin/env python3
"""System check for a Strata Metal serve setup: one quick, comprehensive report.

Reads every strata-*.json serve config in the repo root (so it follows model and
quant swaps without editing), then checks the checkout, the model files, the
sessions, the live serve, the log, and - with --parity - the kernel parity tests.

Run:
    .venv/bin/python metal/tools/system_check.py              # quick report (~1 min with prompts)
    .venv/bin/python metal/tools/system_check.py --parity     # + kernel parity battery (safe with serve live)
    .venv/bin/python metal/tools/system_check.py --json       # machine-readable result

Exit code: 0 all green, 1 warnings, 2 failures.

Scrub policy: the token-0 filler run is never printed raw (it would re-enter model
context and be imitated); it shows as [filler] here.
"""
import argparse
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FILLER_RE = re.compile(r"!{3,}")
# Crash/corruption signatures for the serve log (case-insensitive).
BAD_LOG_RE = re.compile(
    r"sigsegv|sigbus|sigtrap|abort|exception|traceback|assert|panic|corrupt|killed", re.I)
# Informational lines that mention a crash word in prose; benign.
BENIGN_LOG_RE = re.compile(r"resident footprint|probe failed", re.I)
# Last measured performance lines the engine writes per request.
PROMPT_LINE_RE = re.compile(
    r"prompt (\d+) tokens.*?\(([\d.]+) tok/s\), (\d+) generated in [\d.]+ ms \(([\d.]+) tok/s\)"
    r"(?:, drafts accepted (\d+) of (\d+))?")
RESTORE_RE = re.compile(r"session restored (\d+) tokens")
CACHE_RE = re.compile(r"conversation cache: restored (\d+) tokens")

# Regression guards: a file must contain a marker, or the check fails. Add one
# line per past incident so a merge that drops the fix is caught before a serve.
GUARDS = [
    ("metal/kernels/moe_mm.metal", "MMEXP(T_Q5_0)",
     "mm_expand for Q5_0 down experts (missing one aborted the first prompt chunk)"),
]

# Parity battery. moe_mm_parity needs the quant families the packs use; the
# sets below cover both shipped slots.
PARITY = [
    ("route_parity", ["build-metal/route_parity"], 120),
    ("gemv_parity", ["build-metal/gemv_parity"], 180),
    ("window_cells_test", ["build-metal/window_cells_test"], 180),
    ("moe_mm_parity", ["build-metal/moe_mm_parity",
                       "--gu", "IQ3_XXS", "IQ4_XS", "IQ2_S", "--dn", "Q5_0", "IQ4_NL", "Q2_0"], 240),
]

RESULTS = []  # (status, section, label, detail)


def add(status, section, label, detail=""):
    RESULTS.append({"status": status, "section": section, "label": label, "detail": detail})


def scrub(s):
    return FILLER_RE.sub("[filler]", s)


def fmt_bytes(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def port_open(host, port, timeout=0.4):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def api_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def api_chat(base, model, messages, max_tokens=240, temperature=0.0, timeout=300):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def engine_pid_for(pack_path):
    out = subprocess.run(["pgrep", "-f", "--", f"--pack {pack_path}"],
                         capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def proc_env(pid):
    if not pid:
        return ""
    return subprocess.run(["ps", "eww", "-p", str(pid)], capture_output=True,
                          text=True).stdout


# ---------------- sections ----------------

def check_repo():
    s = "checkout"
    try:
        head = subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"],
                              capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "-C", REPO, "branch", "--show-current"],
                                capture_output=True, text=True).stdout.strip()
        add("PASS", s, "git", f"{branch}: {head[:80]}")
    except OSError as e:
        add("WARN", s, "git", str(e))

    exe = None
    for cfg in CONFIGS:
        exe = cfg.get("exe") or exe
    if not exe or not os.path.exists(exe):
        add("FAIL", s, "engine binary", f"{exe} missing")
        return exe
    bt = os.path.getmtime(exe)
    stale = []
    for root in ("src", "include", "metal", "sycl"):
        d = os.path.join(REPO, root)
        for dirpath, _, files in os.walk(d):
            if os.path.join(REPO, "metal", "tools") in dirpath or "/tests" in dirpath or "/build" in dirpath:
                continue
            for f in files:
                if f.endswith((".cpp", ".hpp", ".h", ".m", ".mm", ".metal", ".c")):
                    p = os.path.join(dirpath, f)
                    if os.path.getmtime(p) > bt:
                        stale.append(os.path.relpath(p, REPO))
    if stale:
        add("FAIL", s, "binary freshness",
            f"{exe} older than {len(stale)} source file(s): {', '.join(stale[:4])}"
            " (kernels JIT at start, but C++ needs a rebuild)")
    else:
        add("PASS", s, "binary freshness", f"{os.path.basename(exe)} newer than all sources")

    for rel, marker, why in GUARDS:
        p = os.path.join(REPO, rel)
        try:
            ok = marker in open(p, errors="replace").read()
        except OSError:
            ok = False
        add("PASS" if ok else "FAIL", s, f"guard {os.path.basename(rel)}",
            f"{marker} present" if ok else f"{marker} MISSING - {why}")
    return exe


def check_files(cfg):
    s = "files"
    name = cfg["__name__"]
    args = cfg.get("args", [])
    paths = {}
    for i, a in enumerate(args):
        if a in ("--pack", "--native", "--ple-gguf", "--native-dense-gguf", "--expert-profile") and i + 1 < len(args):
            paths[a] = args[i + 1]
    missing = []
    for flag, p in paths.items():
        if not os.path.exists(p):
            missing.append(f"{flag}={p}")
    if missing:
        add("FAIL", s, name, "missing: " + ", ".join(missing))
    else:
        sizes = ", ".join(f"{k}: {fmt_bytes(os.path.getsize(paths[k]) if os.path.isfile(paths[k]) else sum(
            os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(paths[k]) for f in fs))}"
            for k in ("--pack", "--native") if k in paths)
        add("PASS", s, name, sizes)
    pack = paths.get("--pack", "")
    need = ["index.txt", "conversions.json", "compat-bf16.json", "tokenizer"]
    lack = [f for f in need if pack and not os.path.exists(os.path.join(pack, f))]
    if lack:
        add("FAIL", s, f"{name} pack", f"missing {lack} (repack with metal_repack.py + iq_pack.py --compat-bf16)")
    elif pack:
        add("PASS", s, f"{name} pack", "index/conversions/compat/tokenizer present")
    ple = paths.get("--ple-gguf")
    if ple and os.path.exists(ple):
        n = os.stat(ple).st_nlink
        if n >= 2:
            add("PASS", s, f"{name} ple nlink", f"{n} links (n-gram table shared, no duplicate on disk)")
        else:
            add("WARN", s, f"{name} ple nlink",
                "link count 1 - the table is a private copy; if another slot ships, "
                f"{fmt_bytes(os.path.getsize(ple))} is duplicated on disk")
    tok = cfg.get("tokenizer", "")
    if tok and not os.path.exists(tok):
        add("FAIL", s, f"{name} tokenizer", f"{tok} missing")
    ss = cfg.get("slot_save_path")
    if ss:
        if os.path.isdir(ss):
            ents = os.listdir(ss)
            quar = [e for e in ents if "quarantine" in e]
            auto = [e for e in ents if e.startswith("auto-session") and "quarantine" not in e]
            det = f"auto-session files: {len(auto)}"
            if quar:
                det += f"; quarantined: {quar}"
            add("WARN" if quar else "PASS", s, f"{name} sessions", det)
        else:
            add("PASS", s, f"{name} sessions", "no slot_save_path dir yet (first save creates it)")


def check_disk():
    du = shutil.disk_usage("/")
    free = du.free
    add("PASS" if free > 150e9 else ("WARN" if free > 60e9 else "FAIL"),
        "disk", "free space", fmt_bytes(free))


def check_serve(cfg, do_live):
    s = "serve"
    name = cfg["__name__"]
    port = cfg.get("port", 8180)
    host = cfg.get("host", "127.0.0.1")
    live = port_open(host, port) or (host != "127.0.0.1" and port_open("127.0.0.1", port))
    base = f"http://{host if port_open(host, port) else '127.0.0.1'}:{port}"
    if not live:
        add("PASS", s, name, f"port {port} closed - slot not running (file checks above still apply)")
        return
    try:
        models = api_get(base + "/v1/models")
        ids = [m["id"] for m in models.get("data", [])]
        want = cfg.get("model_name", "")
        if want in ids:
            add("PASS", s, name, f"serving as {want}")
        else:
            owners = [c["__name__"] for c in CONFIGS if c is not cfg and c.get("model_name") in ids]
            if owners:
                add("PASS", s, name, f"standby - port {port} is held by {'/'.join(owners)}")
                return
            add("FAIL", s, name, f"ids {ids} != config model_name {want!r}")
    except (OSError, ValueError) as e:
        add("FAIL", s, name, f"port open but /v1/models failed: {e}")
        return

    pid = engine_pid_for(cfg.get("args", [])[1] if "--pack" not in cfg.get("args", []) else
                         cfg["args"][cfg["args"].index("--pack") + 1])
    env = proc_env(pid)
    bad_env = [f"{k}={v}" for k, v in (cfg.get("env") or {}).items()
               if pid and f"{k}={v}" not in env]
    if not pid:
        add("WARN", s, f"{name} engine pid", "engine process not found by --pack path")
    elif bad_env:
        add("FAIL", s, f"{name} tuning env", "engine env missing: " + ", ".join(bad_env))
    else:
        add("PASS", s, f"{name} tuning env",
            f"{len(cfg.get('env') or {})} STRATA_METAL_* vars live on pid {pid}")
    et = subprocess.run(["ps", "-o", "etime=", "-p", str(pid)],
                        capture_output=True, text=True).stdout.strip() if pid else ""
    if et:
        # BSD etime: [[DD-]hh:]mm:ss
        days, _, rest = et.rpartition("-")
        parts = [int(x) for x in rest.split(":")]
        secs = sum(v * 60 ** (len(parts) - 1 - i) for i, v in enumerate(parts)) + int(days or 0) * 86400
        started = time.time() - secs
        py = max((os.path.getmtime(p) for p in glob.glob(os.path.join(REPO, "serve", "*.py"))), default=0)
        if py > started:
            add("WARN", s, f"{name} serve code",
                "serve/*.py edited after the engine started - restart to pick it up")
        else:
            add("PASS", s, f"{name} serve code", "running code matches serve/*.py on disk")

    log = cfg.get("log", "")
    if log and os.path.exists(log):
        lines = open(log, errors="replace").read().splitlines()[-4000:]
        lines = [scrub(l) for l in lines]
        bad = [l for l in lines if BAD_LOG_RE.search(l) and not BENIGN_LOG_RE.search(l)]
        if bad:
            add("FAIL", s, f"{name} log", f"{len(bad)} crash/corruption lines in last 4000, last: {bad[-1][:120]}")
        else:
            add("PASS", s, f"{name} log", "no crash/corruption lines in last 4000")
        perf = [PROMPT_LINE_RE.search(l) for l in lines]
        perf = [m for m in perf if m][-1:]
        if perf:
            m = perf[0]
            det = (f"prompt {m.group(2)} tok/s, decode {m.group(4)} tok/s, "
                   f"{int(m.group(5) or 0)}/{int(m.group(6) or 0)} drafts accepted")
            okd = float(m.group(4)) >= 20.0
            add("PASS" if okd else "WARN", s, f"{name} last request", det + ("" if okd else " (< 20 tok/s)"))
        res = [RESTORE_RE.search(l) for l in lines]
        res = [m for m in res if m][-1:]
        if res:
            add("PASS", s, f"{name} session restore", f"last restore {res[0].group(1)} tokens, clean")
        else:
            add("PASS", s, f"{name} session restore", "no restore in recent log (fresh session)")

    if do_live:
        needle = "zephyr-quokka-%d" % (int(time.time()) % 10000)
        try:
            d1, t1 = api_chat(base, cfg["model_name"], [
                {"role": "user", "content": f"Remember the code {needle}. "
                 "Without repeating anything else, answer: what is 17*23? Think briefly, then give the number."}])
            msg = d1["choices"][0]["message"]
            text = scrub((msg.get("content") or "") + " " + (msg.get("reasoning_content") or msg.get("reasoning") or ""))
            if "391" in text.replace(" ", ""):
                add("PASS", s, f"{name} live math", f"391 correct in {t1:.1f}s")
            else:
                add("FAIL", s, f"{name} live math", f"no 391 in {text[:120]!r}")
            d2, t2 = api_chat(base, cfg["model_name"], [
                {"role": "user", "content": f"What code did I mention earlier? Reply with just the code. Context: {needle}."}])
            text2 = scrub((d2["choices"][0]["message"].get("content") or ""))
            if needle in text2:
                add("PASS", s, f"{name} live recall", f"needle recalled in {t2:.1f}s")
            else:
                add("FAIL", s, f"{name} live recall", f"needle missing: {text2[:120]!r}")
            filler = len(re.findall(r"\[filler\]", text)) + len(re.findall(r"\[filler\]", text2))
            if filler:
                add("FAIL", s, f"{name} filler", f"{filler} filler runs in live answers")
            else:
                add("PASS", s, f"{name} filler", "no token-0 filler in live answers")
        except (OSError, ValueError, KeyError) as e:
            add("FAIL", s, f"{name} live prompts", f"{e}")


def check_parity():
    s = "parity"
    for label, argv, timeout in PARITY:
        if not os.path.exists(os.path.join(REPO, argv[0])):
            add("WARN", s, label, "binary not built")
            continue
        try:
            p = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=timeout)
            out = p.stdout + p.stderr
            ok = p.returncode == 0 and "PASS" in out and "FAIL" not in out
            add("PASS" if ok else "FAIL", s, label,
                ("PASS" if ok else (out.strip().splitlines() or ["rc=%d" % p.returncode])[-1][:160]))
        except subprocess.TimeoutExpired:
            add("FAIL", s, label, f"timeout {timeout}s")


CONFIGS = []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parity", action="store_true", help="run the kernel parity battery too")
    ap.add_argument("--skip-live", action="store_true", help="skip live prompts (keep the needle test out)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a report")
    ap.add_argument("--config", action="append", default=None,
                    help="serve config(s) to check; default: every strata-*.json in the repo root")
    args = ap.parse_args()

    global CONFIGS
    paths = args.config or sorted(glob.glob(os.path.join(REPO, "strata-*.json")))
    for p in paths:
        try:
            cfg = json.load(open(p))
            cfg["__name__"] = os.path.basename(p)
            CONFIGS.append(cfg)
        except (OSError, ValueError) as e:
            add("FAIL", "configs", p, str(e))
    if not CONFIGS:
        add("FAIL", "configs", "discovery", "no strata-*.json in repo root")

    check_repo()
    for cfg in CONFIGS:
        check_files(cfg)
    check_disk()
    for cfg in CONFIGS:
        check_serve(cfg, do_live=not args.skip_live)
    if args.parity:
        check_parity()

    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    worst = max((order[r["status"]] for r in RESULTS), default=2)
    if args.json:
        print(json.dumps({"exit": worst, "results": RESULTS}, indent=1))
    else:
        cur = None
        for r in RESULTS:
            if r["section"] != cur:
                cur = r["section"]
                print(f"[{cur}]")
            mark = {"PASS": "ok  ", "WARN": "WARN", "FAIL": "FAIL"}[r["status"]]
            print(f"  {mark}  {r['label']:<38} {r['detail'][:110]}")
            worst = max(worst, order[r["status"]])
        print({"0": "ALL GREEN", "1": "WARNINGS - see above", "2": "FAILURES - see above"}[str(worst)])
    sys.exit(worst)


if __name__ == "__main__":
    main()

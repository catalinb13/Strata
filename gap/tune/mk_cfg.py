#!/usr/bin/env python3
"""gap/tune/mk_cfg.py - write a server config JSON from the live one, overridden.

    python3 gap/tune/mk_cfg.py OUT.json --port 8199 --parallel 2 \
        --set --max-context 16384 --set --kv fp16

`--set FLAG [VALUE]` replaces FLAG's value in args (or appends it), and existing
occurrences are removed first so the last one wins.
"""
import json
import sys

BASE = "/Users/pc/src/Strata-fork/strata-ud-iq4_xs-huihui.json"


def main():
    out = sys.argv[1]
    rest = sys.argv[2:]
    port = None
    sets = []
    i = 0
    while i < len(rest):
        if rest[i] == "--port":
            port = int(rest[i + 1]); i += 2
        elif rest[i] == "--set":
            flag = rest[i + 1]
            val = None
            if i + 2 < len(rest) and not rest[i + 2].startswith("--"):
                val = rest[i + 2]
            sets.append((flag, val))
            i += 2 if val is None else 3
        else:
            sys.exit(f"bad arg {rest[i]}")

    cfg = json.load(open(BASE))
    args = list(cfg["args"])
    for flag, val in sets:
        while flag in args:
            j = args.index(flag)
            end = j + 1
            if j + 1 < len(args) and not args[j + 1].startswith("--"):
                end = j + 2
            del args[j:end]
        args.append(flag)
        if val is not None:
            args.append(val)
    cfg["args"] = args
    if port is not None:
        cfg["port"] = port
    cfg["auto_session"] = False
    cfg["open_browser"] = False
    json.dump(cfg, open(out, "w"), indent=1)
    print("wrote", out)
    print("  kv/ctx flags:", [a for a in args if a in ("--kv", "fp16", "int8", "--max-context")
                              or a.isdigit() and len(a) <= 6][:8])


if __name__ == "__main__":
    main()

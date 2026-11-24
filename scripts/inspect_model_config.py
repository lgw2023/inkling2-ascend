#!/usr/bin/env python3
"""Inspect a local config's fixed-loop visits and logical KV memory; no device access."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.planning import kv_estimate, visit_plan  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--element-bytes", type=int, default=2)
    parser.add_argument("--show-visits", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = kv_estimate(config, tokens=args.tokens, tp=args.tp,
                             requests=args.requests, element_bytes=args.element_bytes)
        if args.show_visits:
            report["visits"] = visit_plan(config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

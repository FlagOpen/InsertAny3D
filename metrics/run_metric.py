#!/usr/bin/env python3
"""Central metric dispatcher for InsertAny3D.

The registry intentionally distinguishes future comparison metrics from the
currently available unsupervised HPSv2 metric.
"""

from __future__ import annotations

import argparse
import sys


METRICS = {
    "hpsv2": ("unsupervised", "evaluate_hpsv2"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="InsertAny3D metrics central entry point")
    parser.add_argument("--metric", choices=sorted(METRICS))
    args, rest = parser.parse_known_args(argv)
    if not args.metric:
        parser.error("必须指定 --metric")
    evaluation_type, module_name = METRICS[args.metric]
    if evaluation_type != "unsupervised":
        parser.error(f"当前 metric 未实现：{args.metric}")
    from evaluate_hpsv2 import main as metric_main
    return metric_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry point.

Research:    loadcast download | build | backtest | tune | report
Operations:  loadcast train | publish --run ID | fetch | forecast | dashboard
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from dotenv import load_dotenv

from loadcast.config import Config

COMMANDS = [
    "download",
    "build",
    "backtest",
    "tune",
    "report",
    "train",
    "publish",
    "fetch",
    "forecast",
    "dashboard",
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="loadcast")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--country", action="append", help="Restrict to these countries")
    parser.add_argument("--model", action="append", help="Restrict to these models")
    parser.add_argument("--end", help="Override data.end, e.g. 2025-06-30 or 'yesterday'")
    parser.add_argument(
        "--param",
        action="append",
        help="Override a hyper-parameter, e.g. --param xgboost.max_depth=6",
    )
    parser.add_argument(
        "--grid",
        action="append",
        help="tune only: candidate values, e.g. --grid xgboost.max_depth=6,8",
    )
    parser.add_argument(
        "--run", help="Training run id (train: name it; publish, backtest: which one)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="download only: show what is cached and what would be fetched",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    cfg = Config.load(args.config).with_end(args.end)
    if args.country:
        cfg = replace(cfg, countries={c: cfg.countries[c] for c in args.country})
    if args.model:
        cfg = replace(cfg, models=args.model)
    cfg = cfg.with_params(args.param)

    # Imported lazily so that data commands do not need torch to be importable.
    if args.command in ("download", "build"):
        from loadcast.data import dataset

        if args.command == "download":
            dataset.download(cfg, dry_run=args.dry_run)
        else:
            dataset.build(cfg)
    elif args.command == "backtest":
        from loadcast import backtest

        backtest.run(cfg, args.run)
    elif args.command == "tune":
        from loadcast import tune

        tune.run(cfg, args.grid)
    elif args.command == "report":
        from loadcast import report

        report.run(cfg)
    elif args.command in ("train", "publish", "fetch", "forecast"):
        from loadcast import live

        if args.command == "train":
            live.train(cfg, args.run)
        elif args.command == "publish":
            if not args.run:
                parser.error("publish needs --run")
            live.publish(args.run)
        elif args.command == "fetch":
            live.fetch()
        else:
            live.forecast(cfg)
    else:
        from loadcast import dashboard

        print(dashboard.build(cfg))


if __name__ == "__main__":
    main()

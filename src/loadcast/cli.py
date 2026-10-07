"""Command line entry point: `loadcast {download,build,backtest,report}`."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from loadcast.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(prog="loadcast")
    parser.add_argument("command", choices=["download", "build", "backtest", "report"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--country", action="append", help="Restrict to these countries")
    parser.add_argument("--model", action="append", help="Restrict to these models")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    cfg = Config.load(args.config)
    countries = args.country or list(cfg.countries)

    # Imported lazily so that `download` does not need torch to be importable.
    if args.command in ("download", "build"):
        from loadcast.data import dataset

        getattr(dataset, args.command)(cfg)
    elif args.command == "backtest":
        from loadcast import backtest

        for code in countries:
            backtest.run(cfg, code, args.model)
    else:
        from loadcast import report

        report.run(cfg)


if __name__ == "__main__":
    main()

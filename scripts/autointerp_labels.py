"""CLI wrapper for SAE autointerp labeling."""

from __future__ import annotations

from scripts.autointerp.runner import run_autointerp_labels


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_autointerp_labels(args.config_path))


if __name__ == "__main__":
    main()

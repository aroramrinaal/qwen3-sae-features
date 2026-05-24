"""CLI wrapper for SAE feature dashboard generation."""

from __future__ import annotations

from scripts.dashboard.runner import run_feature_dashboard

__all__ = ["run_feature_dashboard"]


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_feature_dashboard(args.config_path))


if __name__ == "__main__":
    main()

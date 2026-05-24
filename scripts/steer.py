"""CLI wrapper for SAE decoder-direction steering runs."""

from __future__ import annotations

from scripts.steering.runner import run_steering

__all__ = ["run_steering"]


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_steering(args.config_path))


if __name__ == "__main__":
    main()

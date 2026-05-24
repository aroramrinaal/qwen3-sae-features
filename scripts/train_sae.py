"""CLI wrapper for SAELens SAE training."""

from __future__ import annotations

from scripts.training.runner import run_train

__all__ = ["run_train"]


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_train(args.config_path))


if __name__ == "__main__":
    main()

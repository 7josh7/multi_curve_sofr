from __future__ import annotations

import argparse

from .export import export_market_snapshot


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for exporting validated SOFR market snapshots."""

    parser = argparse.ArgumentParser(description="SOFR multi-curve utilities.")
    parser.add_argument(
        "command",
        nargs="?",
        default="export-snapshot",
        choices=["export-snapshot"],
        help="Command to run. Defaults to export-snapshot.",
    )
    parser.add_argument("--project-root", default=None, help="Path to the multi_curve_sofr project root.")
    parser.add_argument("--output", default=None, help="Output path for the market snapshot JSON.")

    args = parser.parse_args(argv)
    if args.command == "export-snapshot":
        path = export_market_snapshot(project_root=args.project_root, output_path=args.output)
        print(f"Wrote market snapshot: {path.resolve()}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

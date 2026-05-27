import argparse
from pathlib import Path

from .dblp_geometry_summary import run_geometry_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Seldon DBLP geometry robustness and validation bundle."
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Geometry config path. Repeat for each venue/semantics combination.",
    )
    parser.add_argument("--output-dir", required=True, help="Validation output directory.")
    args = parser.parse_args()
    outdir = run_geometry_summary(
        configs=[Path(config) for config in args.config],
        output_dir=Path(args.output_dir),
    )
    print(f"Seldon geometry validation bundle complete. Outputs in {outdir}")


if __name__ == "__main__":
    main()

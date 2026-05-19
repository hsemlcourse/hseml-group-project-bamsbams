"""Raw-to-processed parsing entrypoint for reproducible dataset creation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json

from src.preprocessing import DomainConfig, build_processed_dataset


def load_domain_config(path: Path) -> DomainConfig:
    """Load domain configuration from JSON file in data/raw."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DomainConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse raw config and build processed dataset"
    )
    parser.add_argument(
        "--raw-config",
        default="data/raw/domain_config.json",
        help="Path to raw JSON config",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-collocation", type=int, default=25000)
    parser.add_argument("--n-initial", type=int, default=2000)
    parser.add_argument("--n-boundary-per-side", type=int, default=2000)
    parser.add_argument("--fdm-nx", type=int, default=51)
    parser.add_argument("--fdm-ny", type=int, default=51)
    args = parser.parse_args()

    raw_path = Path(args.raw_config)
    cfg = load_domain_config(raw_path)

    out_path = build_processed_dataset(
        output_dir=args.output_dir,
        n_collocation=args.n_collocation,
        n_initial=args.n_initial,
        n_boundary_per_side=args.n_boundary_per_side,
        fdm_nx=args.fdm_nx,
        fdm_ny=args.fdm_ny,
        seed=args.seed,
        cfg=cfg,
    )

    manifest = {
        "raw_config_path": str(raw_path),
        "raw_config": asdict(cfg),
        "output_dataset": str(out_path),
        "seed": args.seed,
    }
    Path(args.output_dir, "parsing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Processed dataset written to: {out_path}")


if __name__ == "__main__":
    main()

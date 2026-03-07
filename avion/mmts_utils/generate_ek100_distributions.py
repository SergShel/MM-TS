"""Generate and save class-distribution plots for EK100 annotations."""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

DISTRIBUTION_CONFIGS = [
    {
        "column": "verb_class",
        "title": "Verbs",
        "plot": "ek100_verb_distribution.png",
        "freq": "ek100_verb_freq.csv",
    },
    {
        "column": "noun_class",
        "title": "Nouns",
        "plot": "ek100_noun_distribution.png",
        "freq": "ek100_noun_freq.csv",
    },
    {
        "column": "verb_noun",
        "title": "Verb-Noun Pairs",
        "plot": "ek100_verb_noun_distribution.png",
        "freq": "ek100_verb_noun_freq.csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate class-distribution plots for EK100 annotations.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the EK100 annotation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the plots.",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["verb", "noun", "verb_noun"],
        default=["verb", "noun", "verb_noun"],
        help="Which distributions to plot (default: all).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures.",
    )
    return parser.parse_args()


def load_annotations(csv_path: str) -> pd.DataFrame:
    """Load the annotation CSV and derive the verb_noun column."""
    logger.info("Loading annotations from %s", csv_path)
    df = pd.read_csv(csv_path)
    df["verb_noun"] = df["verb_class"].astype(str) + "_" + df["noun_class"].astype(str)
    return df


def save_distribution(distribution: pd.Series, title: str, save_path: Path) -> None:
    """Save a frequency distribution to a CSV file."""
    freq_df = distribution.reset_index()
    freq_df.columns = ["class_id", "count"]
    freq_df.to_csv(save_path, index=False)
    logger.info(
        "Saved %s frequencies → %s (%d classes)", title, save_path, len(freq_df)
    )


def plot_distribution(
    distribution: pd.Series,
    title: str,
    save_path: Path,
    dpi: int = 300,
    show: bool = True,
    color: str = "steelblue",
    figsize: tuple = (8, 5),
) -> None:
    """Plot and save a single class-frequency distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(distribution))
    ax.plot(x, distribution.values, color=color, linewidth=2)
    ax.fill_between(x, distribution.values, alpha=0.2, color=color)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel("# of Samples", fontsize=12)
    ax.set_xlim(0, len(distribution))
    ax.set_ylim(0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    logger.info("Saved %s → %s", title, save_path)
    plt.show()
    plt.close(fig)


def generate_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    distributions: List[str],
    dpi: int = 300,
) -> None:
    """Compute and plot requested distributions."""
    # Map short names to config entries
    name_to_config = {
        cfg["column"].replace("_class", ""): cfg for cfg in DISTRIBUTION_CONFIGS
    }

    for name in distributions:
        cfg = name_to_config[name]
        dist = df[cfg["column"]].value_counts()
        plot_distribution(dist, cfg["title"], output_dir / cfg["plot"], dpi=dpi)
        save_distribution(dist, cfg["title"], output_dir / cfg["freq"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(args.csv)
    generate_distributions(
        df,
        output_dir=output_dir,
        distributions=args.distributions,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

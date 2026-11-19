"""Reliability diagram image (also useful in the README / report)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_reliability_plot(before: dict, after: dict, path, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, ev in (("Before", before), ("After", after)):
        pts = [(b["mean_confidence"], b["accuracy"]) for b in ev["reliability_curve"] if b["count"]]
        ax.plot(*zip(*pts), marker="o", label=f"{name} (ECE={ev['ece']:.3f})")
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram (test set){title_suffix}", fontsize=10)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

import matplotlib.pyplot as plt


SERIES_STYLES = {
    "blue": {"color": "#20639B", "marker": "o", "linewidth": 1.8},
    "red": {"color": "#D1495B", "marker": "s", "linewidth": 1.8},
    "green": {"color": "#2A9D8F", "marker": "^", "linewidth": 1.8},
    "slate": {"color": "#2F4858", "marker": "D", "linewidth": 1.9},
}
HORIZONTAL_GRID = {"axis": "y", "color": "0.88", "linewidth": 0.7}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 140,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.975,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )

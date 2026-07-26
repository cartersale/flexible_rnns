"""Shared style for the paper's panel SVGs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"; BAND = "#eef4fc"; SURF = "#fcfcfb"
BLUE = "#2a78d6"; YELLOW = "#eda100"; GREEN = "#008300"; MAGENTA = "#e87ba4"
ORANGE = "#eb6834"; AQUA = "#1baf7a"; VIOLET = "#4a3aa7"; RED = "#e34948"; GRAY = "#9a9a97"
SIZE_COL = {128: BLUE, 256: GREEN, 512: MAGENTA, 1024: YELLOW}


def clean(ax, grid="y"):
    if grid == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    elif grid == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=SURF)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"wrote {path}")

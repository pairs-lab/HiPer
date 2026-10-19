"""
Shared matplotlib style for HiPer RA-L figures.

IEEEtran column width is 3.5 in (88.9 mm); full text width 7.16 in.
Fonts are embedded as TrueType (fonttype 42) so IEEE PDF eXpress
accepts them; serif family matches the Times-like body text.
"""

import matplotlib as mpl

COLUMN_W = 3.5      # inches, single column
TEXT_W = 7.16       # inches, double column

# Okabe-Ito colour-blind-safe palette
C = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "sky":     "#56B4E9",
    "yellow":  "#F0E442",
    "grey":    "#7F7F7F",
}


def use_ieee_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 300,
    })


def save(fig, stem):
    """Write both the vector PDF (for the paper) and a PNG preview."""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=220)
    print(f"  wrote {stem}.pdf / .png")

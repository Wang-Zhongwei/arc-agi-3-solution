"""Ten recorded matching cases, drawn from study results and verified on replay."""

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from run_study import changed, engines, metrics, pixels, render

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "cases"
OUT.mkdir(exist_ok=True)
BLUE, GOOD, BAD, MISS, MUTED = "#2878bc", "#15836d", "#cb4045", "#d29425", "#8895a4"
INK = "#172c42"
PALETTE = ["#f3f6fa", BLUE, "#985acb", "#27978e"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.titlecolor": INK,
        "savefig.facecolor": "white",
    }
)
CASES = [
    (
        "stationary_duplicate/0",
        "One of two identical objects moves",
        "Both methods match the moving object and omit the stationary object.",
        "A stays at column 8. B moves from column 12 to 16. Removing unchanged A first prevents it from competing for B’s destination.",
    ),
    (
        "rotation_90/0",
        "A single object rotates 90° clockwise",
        "Both methods get the pair and rotation right.",
        "There is only one compatible foreground pair. Rotation is verified from the masks before assignment, so the choice of matcher makes no difference.",
    ),
    (
        "mirror_axis_|/0",
        "A single object is reflected left–right",
        "Both methods get the pair and reflection right.",
        "The chiral L changes handedness. A reflection across the vertical axis matches the masks; rotation alone does not.",
    ),
    (
        "scale_3_to_2/0",
        "An object shrinks from scale 3 to scale 2",
        "Both methods get the pair and scale factor 2/3 right.",
        "The colored pixels differ, but the oriented reduced masks are identical. The supported change is exact uniform scaling.",
    ),
    (
        "local_competition_ordered/0",
        "Nearest locally, wrong globally",
        "Hungarian gets both identities right. Mutual nearest chooses one wrong pair.",
        "True moves: A 8→11 and B 12→16. Mutual nearest picks B→1 because its distance is 1. Hungarian picks A→1 and B→2, total distance 7 rather than 9.",
    ),
    (
        "local_tie_global_unique/0",
        "A local tie hides a unique global assignment",
        "Hungarian gets both right. Mutual nearest returns neither pair.",
        "Before columns are 8 and 12; after columns are 10 and 16. Node 1 has two equally close predecessors. The complete minimum-cost assignment is nevertheless unique.",
    ),
    (
        "local_competition_crossed/0",
        "Same images as case 5, different true history",
        "Mutual nearest gets one right. Hungarian gets both identities wrong.",
        "The actual moves are A 8→16 and B 12→11. The frames are identical to case 5: only the hidden fixture history differs. Our B→1 pair is now correct.",
    ),
    (
        "crossing/0",
        "Objects cross: shorter paths give wrong identities",
        "Both methods make wrong matches; Hungarian makes two.",
        "True moves: A 8→12 and B 16→10. The distance heuristic instead favors A→1 at column 10. Hungarian also pairs B→2 at column 12.",
    ),
    (
        "unrelated_changed_color/0",
        "Replacement looks exactly like recoloring",
        "Both methods incorrectly claim that the old object was recolored.",
        "The fixture deletes the blue object and creates a different purple object on the same cells. The images are indistinguishable from recoloring one persistent object.",
    ),
    (
        "rotate+recolor/0",
        "Rotation and recoloring happen together",
        "Both methods leave the true pair unmatched, as the current rules require.",
        "A both rotates and changes color. There is no combined-change bucket, so the true pair is reported as disappearance/appearance. This miss is a policy limit, not an assignment failure.",
    ),
]


def label_maps(scene):
    def order(objects):
        return sorted(objects, key=lambda o: min(pixels(o)))
    a, b = order(scene["before"]), order(scene["after"])
    return (
        {o["identity"]: chr(65 + i) for i, o in enumerate(a)},
        {o["identity"]: str(i + 1) for i, o in enumerate(b)},
    )


def draw_grid(ax, objects, labels, bounds, title):
    r0, c0, r1, c1 = bounds
    grid = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for o in objects:
        for r, c in pixels(o):
            grid[r - r0, c - c0] = o["color"]
    ax.imshow(
        grid,
        cmap=ListedColormap(PALETTE),
        vmin=0,
        vmax=3,
        interpolation="nearest",
        extent=(c0 - 0.5, c1 + 0.5, r1 + 0.5, r0 - 0.5),
    )
    ax.set_xticks(np.arange(c0 - 0.5, c1 + 1, 1), minor=True)
    ax.set_yticks(np.arange(r0 - 0.5, r1 + 1, 1), minor=True)
    ax.grid(which="minor", color="#dbe3eb", linewidth=0.55)
    step = max(1, (c1 - c0) // 8)
    ax.set_xticks(range(c0, c1 + 1, step))
    ax.set_yticks(range(r0, r1 + 1, max(1, (r1 - r0) // 4)))
    ax.tick_params(which="both", length=0, labelsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("column", fontsize=9, labelpad=2)
    ax.set_ylabel("row", fontsize=9, labelpad=2)
    for o in objects:
        coords = pixels(o)
        minr = min(r for r, c in coords)
        minc = min(c for r, c in coords)
        maxc = max(c for r, c in coords)
        ax.text(
            (minc + maxc) / 2,
            minr - 1.1,
            labels[o["identity"]],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_links(ax, scene, maps, title, measurement=None):
    la, lb = maps
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    before = {o["identity"]: o for o in scene["before"]}
    after = {o["identity"]: o for o in scene["after"]}

    def positions(labels):
        keys = list(labels)
        xs = np.linspace(0.22, 0.78, len(keys)) if len(keys) > 1 else [0.5]
        return dict(zip(keys, xs))

    xa, xb = positions(la), positions(lb)
    positives = {
        (key, key)
        for key in before.keys() & after.keys()
        if changed(before[key], after[key])
    }
    stationary = {
        (key, key)
        for key in before.keys() & after.keys()
        if not changed(before[key], after[key])
    }

    def arrow(pair, color, style="-", width=2, curve=0):
        i, j = pair
        ax.add_patch(
            FancyArrowPatch(
                (xa[i], 0.76),
                (xb[j], 0.23),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=width,
                linestyle=style,
                color=color,
                connectionstyle=f"arc3,rad={curve}",
                zorder=2,
            )
        )

    for pair in stationary:
        arrow(pair, MUTED, ":", 1.3)
    if measurement is None:
        for pair in sorted(positives):
            arrow(pair, BLUE)
        if not positives:
            ax.text(
                0.5,
                0.48,
                "Old object disappears.\nNew object appears.",
                ha="center",
                va="center",
                fontsize=11,
            )
        summary = "Known fixture history"
    else:
        predictions = {(p["before"], p["after"]) for p in measurement["predicted"]}
        for pair in sorted(positives - predictions):
            arrow(pair, MISS, "--", 1.6, 0.1 if len(xa) > 1 else 0)
        for pair in sorted(predictions):
            arrow(pair, GOOD if pair in positives else BAD)
        if not predictions:
            ax.text(
                0.5,
                0.49,
                "No pair reported",
                ha="center",
                va="center",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
            )
        summary = f"{measurement['TP']} correct   {measurement['FP']} wrong   {measurement['FN']} missed"
    for objs, labels, xs, y in ((before, la, xa, 0.85), (after, lb, xb, 0.13)):
        for key, label in labels.items():
            ax.add_patch(
                Rectangle(
                    (xs[key] - 0.045, y - 0.045),
                    0.09,
                    0.09,
                    facecolor=PALETTE[objs[key]["color"]],
                    edgecolor="none",
                    zorder=3,
                )
            )
            ax.text(
                xs[key],
                y + (0.095 if y > 0.5 else -0.095),
                label,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
    ax.text(0.01, 0.85, "Before", fontsize=9, color=MUTED, ha="left", va="center")
    ax.text(0.01, 0.13, "After", fontsize=9, color=MUTED, ha="left", va="center")
    ax.text(
        0.5, -0.16, summary, ha="center", va="center", fontsize=11, fontweight="bold"
    )


def figure(index, scene, title, subtitle, explanation):
    # Case 10 contains two stationary distractors: preserve them in scoring and
    # replay, but crop them out of the display so the changing shape is readable.
    shown = scene
    note = "Frames are zoomed for display; row/column coordinates remain original."
    if scene["name"] == "rotate+recolor/0":
        shown = {
            **scene,
            "before": [o for o in scene["before"] if o["identity"] == "target"],
            "after": [o for o in scene["after"] if o["identity"] == "target"],
        }
        note = "Target zoom. Two unchanged distractors are outside this view; both methods received the full frames."
    maps = label_maps(shown)
    coords = set().union(*(pixels(o) for o in shown["before"] + shown["after"]))
    r0, c0 = min(r for r, c in coords), min(c for r, c in coords)
    r1, c1 = max(r for r, c in coords), max(c for r, c in coords)
    bounds = (r0 - 2, c0 - 2, r1 + 1, c1 + 2)
    fig = plt.figure(figsize=(12.8, 8.7))
    fig.suptitle(f"{index:02d} · {title}", fontsize=20, fontweight="bold", y=0.99)
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=13)
    draw_grid(
        fig.add_axes([0.065, 0.55, 0.39, 0.30]),
        shown["before"],
        maps[0],
        bounds,
        "BEFORE",
    )
    draw_grid(
        fig.add_axes([0.565, 0.55, 0.39, 0.30]),
        shown["after"],
        maps[1],
        bounds,
        "AFTER",
    )
    fig.text(0.5, 0.485, note, ha="center", fontsize=10, color=MUTED)
    for x, title, measurement in (
        (0.04, "Ground truth", None),
        (0.37, "Mutual nearest", scene["measurements"]["mutual"]),
        (0.70, "Hungarian", scene["measurements"]["hungarian"]),
    ):
        draw_links(
            fig.add_axes([x, 0.205, 0.27, 0.24]), shown, maps, title, measurement
        )
    for x, text, color in (
        (0.09, "Solid green: correct", GOOD),
        (0.32, "Solid red: wrong", BAD),
        (0.55, "Dashed amber: missed", MISS),
        (0.80, "Dotted gray: unchanged", MUTED),
    ):
        fig.text(x, 0.115, text, color=color, fontsize=10, fontweight="bold")
    fig.text(
        0.5,
        0.066,
        textwrap.fill(explanation, 145),
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.4,
    )
    fig.text(
        0.5,
        0.014,
        f"Recorded case: {scene['name']}  ·  Fixture identity is used for scoring, never supplied to the matcher.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    return fig


def main():
    records = {
        r["name"]: r
        for r in map(json.loads, (ROOT / "data" / "cases.jsonl").read_text().splitlines())
    }
    matchers = engines()
    md = [
        "# Ten visual matching cases",
        "\nThese figures use recorded test fixtures. Mutual nearest is the evaluated alternative; Hungarian is retained in production. "
        "Green arrows are correct matches, red arrows are wrong matches, amber dashed arrows are missed true pairs, "
        "and gray dotted arrows are unchanged objects omitted from the diff. Arrows below the frames show correspondence, not physical motion.",
        "\n[Download all ten figures as a PDF](figures/matching_examples.pdf). "
        "The PNG and SVG files are in [figures/cases/](figures/cases/).",
        "\n| Case | Example | Mutual nearest: correct / wrong / missed | Hungarian: correct / wrong / missed |",
        "|---|---|---:|---:|",
    ]
    details = []
    with PdfPages(ROOT / "figures" / "matching_examples.pdf") as pdf:
        for index, (name, title, subtitle, explanation) in enumerate(CASES, 1):
            scene = records[name]
            for obj in scene["before"] + scene["after"]:
                obj["mask"] = tuple(tuple(row) for row in obj["mask"])
            # Verify figures are backed by current frozen-study outputs.
            ga, na, ia = render(scene["before"])
            gb, nb, ib = render(scene["after"])
            for method, engine in matchers.items():
                actual = metrics(scene, engine(ga, gb, na, nb), ia, ib)
                assert actual == scene["measurements"][method], (name, method)
            filename = f"{index:02d}_" + name.split("/")[0].replace("|", "vertical")
            fig = figure(index, scene, title, subtitle, explanation)
            fig.savefig(OUT / (filename + ".png"), dpi=140, bbox_inches="tight")
            fig.savefig(OUT / (filename + ".svg"), bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            cells = []
            for method in ("mutual", "hungarian"):
                x = scene["measurements"][method]
                cells.append(f"{x['TP']} / {x['FP']} / {x['FN']}")
            md.append(f"| {index:02d} | {title} | {cells[0]} | {cells[1]} |")
            details.extend(
                [
                    f"\n## {index:02d}. {title}",
                    f"\n{subtitle}",
                    f"\n![{title}](figures/cases/{filename}.png)",
                    f"\n{explanation}",
                    f"\nSource fixture: `{name}`.",
                ]
            )
            print(f"{index:02d}: replay verified; figure saved")
    md += [
        "\nCases 5 and 7 deliberately show identical observed frames with different hidden identities. "
        "They demonstrate an information limit, not a contradiction. Case 10 is an intentional single-change-policy rejection. "
        "Background records are excluded from the foreground matching counts, as in the study."
    ] + details
    (ROOT / "visual_cases.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()

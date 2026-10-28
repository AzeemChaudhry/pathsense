"""Tables and charts from runs/tracker-study/results.csv (charts only: no video frames, so
they can go in the public docs)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = Path("docs/images/tracker-study")
# The dataviz reference palette: first three categorical slots (validated all-pairs), the
# blue sequential ramp, and neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUNGS = ["default", "tuned", "+stitch", "+bounce", "+stitch+bounce", "+guard+stitch+bounce"]
NUMBERS = ("tp", "fp", "fn")


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False,
    })  # fmt: skip


def read(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def rung(setup: str) -> tuple[str, str]:
    """'bytetrack-tuned+stitch' -> ('bytetrack', '+stitch')."""
    if "-tuned" not in setup:
        return setup, "default"
    tracker, rest = setup.split("-tuned", 1)
    return tracker, rest or "tuned"


def pooled(rows: list[dict]) -> dict:
    """Sum the crossing counts over scenes, then work out the rates; average the rest."""
    out = {f"{d}_{n}": sum(r[f"{d}_{n}"] for r in rows) for d in ("enter", "exit") for n in NUMBERS}
    for d in ("enter", "exit"):
        tp, fp, fn = (out[f"{d}_{n}"] for n in NUMBERS)
        p, r = (tp / (tp + fp) if tp + fp else 1.0), (tp / (tp + fn) if tp + fn else 1.0)
        out[f"{d}_precision"], out[f"{d}_recall"] = p, r
        out[f"{d}_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    out["macro_f1"] = float(np.mean([r["macro_f1"] for r in rows]))  # mean of scene scores
    tp, fp, fn = (sum(out[f"{d}_{n}"] for d in ("enter", "exit")) for n in NUMBERS)
    out["f1"] = 2 * tp / max(2 * tp + fp + fn, 1)  # every crossing weighs the same
    out["wrong"] = out["enter_fp"] + out["exit_fp"]
    out["missed"] = out["enter_fn"] + out["exit_fn"]
    for key in ("HOTA", "IDF1", "MOTA", "AssA", "track_ms", "zone_error"):
        out[key] = float(np.mean([r[key] for r in rows]))
    for key in ("IDSW", "Frag", "ids"):
        out[key] = sum(r[key] for r in rows)
    return out


def make_report(out: Path, study: dict) -> None:
    style()
    rows = read(out / "results.csv")
    tune = set(study["tune"])
    groups = {
        "tune": lambda r: r["scene"] in tune,
        "check": lambda r: r["scene"] not in tune,
        "all": lambda r: True,
    }
    setups = list(dict.fromkeys(r["setup"] for r in rows))
    table = {
        (s, g): pooled([r for r in rows if r["setup"] == s and keep(r)])
        for s in setups
        for g, keep in groups.items()
    }
    folder = out / "report"
    folder.mkdir(parents=True, exist_ok=True)
    trackers = list(study["trackers"])
    ladder(table, trackers, folder)
    matrix(table, trackers, folder)
    identity_speed(table, trackers, folder)
    final = pick_final(table, trackers)
    per_scene(rows, study, study["reference"]["name"], final, folder)
    write_tables(table, setups, rows, study, final, folder)
    (folder / "final.json").write_text(json.dumps({"final": final}))
    DOCS.mkdir(parents=True, exist_ok=True)
    for chart in folder.glob("*.png"):
        shutil.copy(chart, DOCS / chart.name)
    print(f"report -> {folder}, charts -> {DOCS}; final setup: {final}")


def pick_final(table: dict, trackers: list[str]) -> str:
    """The setup with the best crossing F1 on the tune scenes (check scenes not looked at)."""
    names = {s for s, _ in table}
    return max(names, key=lambda s: (table[(s, "tune")]["f1"], table[(s, "tune")]["IDF1"]))


def ladder(table: dict, trackers: list[str], folder: Path) -> None:
    """One panel per tracker: wrong and missed crossings (all scenes) as layers are added."""
    fig, axes = plt.subplots(1, len(trackers), figsize=(3.0 * len(trackers), 3.6), sharey=True)
    for ax, tracker in zip(axes, trackers, strict=True):
        names = [tracker if r == "default" else f"{tracker}-tuned" + ("" if r == "tuned" else r)
                 for r in RUNGS]  # fmt: skip
        x = np.arange(len(RUNGS))
        for key, colour, label in (("wrong", ORANGE, "wrong"), ("missed", BLUE, "missed")):
            y = [table[(n, "all")][key] for n in names]
            ax.plot(x, y, color=colour, lw=2, marker="o", ms=5, label=label)
            ax.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)  # fmt: skip
        ax.set_title(tracker)
        ax.set_xticks(x, ["def", "tuned", "+st", "+bo", "+st+bo", "+gu+st+bo"], rotation=45)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("crossings, 7 scenes")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Wrong and missed line crossings as each layer is stacked on the tracker",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(folder / "ladder.png", dpi=150)
    plt.close(fig)


def matrix(table: dict, trackers: list[str], folder: Path) -> None:
    """Tracker x rung: crossing F1 per direction, averaged, on the five check scenes."""
    grid = np.array([
        [table[(t if r == "default" else f"{t}-tuned" + ("" if r == "tuned" else r), "check")]
         ["f1"] for r in RUNGS] for t in trackers
    ])  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(trackers) + 1.6))
    lo, hi = grid.min(), grid.max()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    ax.imshow(grid, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            dark = (grid[i, j] - lo) / max(hi - lo, 1e-9) > 0.55
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if dark else INK)  # fmt: skip
    best = np.unravel_index(grid.argmax(), grid.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2))
    ax.set_xticks(range(len(RUNGS)), RUNGS, rotation=20)
    ax.set_yticks(range(len(trackers)), trackers)
    ax.grid(False)
    ax.set_title("Crossing F1 on the five check scenes (every crossing weighs the same)")
    fig.tight_layout()
    fig.savefig(folder / "matrix.png", dpi=150)
    plt.close(fig)


def identity_speed(table: dict, trackers: list[str], folder: Path) -> None:
    """IDF1 against time per frame: each tracker as shipped, and tuned with our layers."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for tracker in trackers:
        a, b = table[(tracker, "all")], table[(f"{tracker}-tuned+stitch+bounce", "all")]
        ax.annotate("", (b["track_ms"], b["IDF1"]), (a["track_ms"], a["IDF1"]),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})  # fmt: skip
        ax.plot(a["track_ms"], a["IDF1"], "o", ms=8, color=RAMP[2], mec=SURFACE, mew=2)
        ax.plot(b["track_ms"], b["IDF1"], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
        ax.annotate(tracker, (b["track_ms"], b["IDF1"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)  # fmt: skip
    ax.plot([], [], "o", color=RAMP[2], label="as shipped")
    ax.plot([], [], "o", color=BLUE, label="tuned + stitch + bounce")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("tracking time per frame, ms (one CPU core, detector not included)")
    ax.set_ylabel("IDF1, mean of 7 scenes")
    ax.set_title("Identity kept against time spent")
    fig.tight_layout()
    fig.savefig(folder / "identity_speed.png", dpi=150)
    plt.close(fig)


def per_scene(rows: list[dict], study: dict, reference: str, final: str, folder: Path) -> None:
    """Per scene: wrong crossings and count error, the 0.3.0 reference against the chosen one."""
    scenes = study["scenes"]
    get = {(r["setup"], r["scene"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x, w = np.arange(len(scenes)), 0.38
    for ax, key, title in (
        (axes[0], "wrong", "Wrong crossings"),
        (axes[1], "count_error", "Count error, IN + OUT (abs)"),
    ):
        for k, (setup, colour) in enumerate(((reference, RAMP[2]), (final, BLUE))):
            values = []
            for scene in scenes:
                r = get[(setup, scene)]
                values.append(
                    r["enter_fp"] + r["exit_fp"] if key == "wrong" else
                    abs(r["enter_ours"] - r["enter_true"]) + abs(r["exit_ours"] - r["exit_true"])
                )  # fmt: skip
            ax.bar(x + (k - 0.5) * w, values, w - 0.04, color=colour, label=setup)
        ax.set_xticks(x, [s.replace("MOT", "") for s in scenes], rotation=30)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "per_scene.png", dpi=150)
    plt.close(fig)


def write_tables(table, setups, rows, study, final, folder: Path) -> None:
    head = ("| setup | IN P | IN R | OUT P | OUT R | F1 tune | F1 check | wrong | missed "
            "| HOTA | IDF1 | IDSW | ms |")  # fmt: skip
    lines = [head, "|" + " --- |" * 13]
    for s in setups:
        a, t, c = table[(s, "all")], table[(s, "tune")], table[(s, "check")]
        lines.append(
            f"| {s} | {a['enter_precision']:.2f} | {a['enter_recall']:.2f} "
            f"| {a['exit_precision']:.2f} | {a['exit_recall']:.2f} | {t['f1']:.3f} "
            f"| {c['f1']:.3f} | {a['wrong']:.0f} | {a['missed']:.0f} | {100 * a['HOTA']:.1f} "
            f"| {100 * a['IDF1']:.1f} | {a['IDSW']:.0f} | {a['track_ms']:.1f} |"
        )
    scene_lines = ["| scene | setup | IN ours / true | OUT ours / true | wrong | missed | IDF1 |",
                   "|" + " --- |" * 7]  # fmt: skip
    get = {(r["setup"], r["scene"]): r for r in rows}
    for scene in study["scenes"]:
        for s in (study["reference"]["name"], final):
            r = get[(s, scene)]
            scene_lines.append(
                f"| {scene} | {s} | {r['enter_ours']:.0f} / {r['enter_true']:.0f} "
                f"| {r['exit_ours']:.0f} / {r['exit_true']:.0f} "
                f"| {r['enter_fp'] + r['exit_fp']:.0f} | {r['enter_fn'] + r['exit_fn']:.0f} "
                f"| {100 * r['IDF1']:.1f} |"
            )
    (folder / "tables.md").write_text("\n".join(lines) + "\n\n" + "\n".join(scene_lines) + "\n")


__all__ = ["make_report"]

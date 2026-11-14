from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
HIST = json.loads((RES / "convergence.json").read_text(encoding="utf-8"))
SOLUTIONS = list(csv.DictReader((RES / "pareto_solutions.csv").open(encoding="utf-8-sig")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig, ax = plt.subplots(figsize=(7.2, 4.2))
for method, color in [("NSGA-II", "#1769aa"), ("Weighted sum GA", "#e07a24")]:
    vals = np.asarray(HIST[method], dtype=float)
    x = np.arange(vals.shape[1])
    med, lo, hi = np.median(vals, axis=0), np.quantile(vals, .25, axis=0), np.quantile(vals, .75, axis=0)
    ax.plot(x, med, label=method, color=color, lw=2)
    ax.fill_between(x, lo, hi, color=color, alpha=.16)
ax.set(title="Сходимость: hypervolume Парето-фронта", xlabel="Поколение", ylabel="Hypervolume (больше — лучше)")
ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "convergence.png"); plt.close(fig)

fig = plt.figure(figsize=(8.2, 5.4))
for k, (xkey, ykey, xlabel, ylabel) in enumerate([
    ("hours", "risk", "Срок, ч", "Риск"),
    ("hours", "value", "Срок, ч", "Ценность"),
    ("risk", "value", "Риск", "Ценность"),
], 1):
    ax = fig.add_subplot(2, 2, k)
    for method, color, marker in [("NSGA-II", "#1769aa", "o"), ("Weighted sum GA", "#e07a24", "x")]:
        rows = [r for r in SOLUTIONS if r["method"] == method]
        ax.scatter([float(r[xkey]) for r in rows], [float(r[ykey]) for r in rows], s=13, alpha=.38,
                   label=method, color=color, marker=marker)
    ax.set(xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.2)
    if k == 1: ax.legend()
ax = fig.add_subplot(2, 2, 4); ax.axis("off")
ax.text(0, .88, "Каждая точка — недоминируемое решение\nитоговой популяции одного запуска.\n\nNSGA-II хранит разнообразие фронта\nодновременно; взвешенная сумма строит\nточки при нескольких заданных весах.", va="top", fontsize=10)
fig.suptitle("Объединённые Парето-решения по независимым запускам")
fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT / "pareto_projections.png"); plt.close(fig)
print(f"Сохранены графики в {OUT}")

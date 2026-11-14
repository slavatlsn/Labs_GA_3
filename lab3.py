from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "features.json"


@dataclass(frozen=True)
class Feature:
    id: str
    name: str
    value: float
    hours: float
    risk: float
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    mandatory: bool = False


def feature_set() -> list[Feature]:
    """Детерминированный учебный экземпляр для продукта промышленной автоматизации."""
    rows = [
        ("F01", "Базовая модель оборудования", 9, 20, 2, (), (), True),
        ("F02", "Ролевой доступ операторов", 8, 24, 3, (), (), True),
        ("F03", "Журнал действий и аудита", 8, 28, 3, ("F02",), (), True),
        ("F04", "Импорт конфигурации PLC", 10, 40, 6, ("F01",), (), False),
        ("F05", "Драйвер Modbus TCP", 9, 32, 4, ("F04",), (), False),
        ("F06", "Драйвер OPC UA", 10, 48, 7, ("F04",), (), False),
        ("F07", "Коннектор MQTT", 7, 28, 4, ("F01",), (), False),
        ("F08", "Потоковая телеметрия", 9, 44, 6, ("F07",), (), False),
        ("F09", "Буферизация при потере связи", 8, 36, 5, ("F08",), (), False),
        ("F10", "Дашборд состояния линии", 8, 32, 3, ("F01",), (), False),
        ("F11", "Настраиваемые пороговые тревоги", 9, 28, 3, ("F10",), (), False),
        ("F12", "Уведомления в Telegram", 6, 20, 4, ("F11",), (), False),
        ("F13", "Уведомления по электронной почте", 5, 18, 2, ("F11",), ("F12",), False),
        ("F14", "Исторические тренды", 8, 40, 4, ("F08",), (), False),
        ("F15", "Экспорт отчёта CSV", 5, 16, 2, (), (), False),
        ("F16", "Экспорт отчёта PDF", 6, 22, 3, ("F15",), (), False),
        ("F17", "Плановое обслуживание", 8, 36, 5, ("F01",), (), False),
        ("F18", "Прогноз отказов по телеметрии", 10, 64, 9, ("F08", "F14"), (), False),
        ("F19", "Ручное подтверждение команд", 8, 30, 4, ("F02",), (), False),
        ("F20", "Автоматическое исполнение команд", 10, 58, 9, ("F19",), (), False),
        ("F21", "Двухфакторная аутентификация", 7, 30, 4, ("F02",), (), False),
        ("F22", "Резервное копирование конфигурации", 7, 26, 3, ("F01",), (), False),
        ("F23", "Синхронизация с корпоративным каталогом", 7, 38, 6, ("F02",), (), False),
        ("F24", "Мобильная панель оператора", 7, 42, 5, ("F10",), (), False),
        ("F25", "Редактор мнемосхем", 9, 56, 7, ("F01",), (), False),
        ("F26", "Симулятор технологического процесса", 8, 52, 6, ("F01",), (), False),
        ("F27", "Песочница для проверки сценариев", 7, 38, 5, ("F26",), (), False),
        ("F28", "Версионирование сценариев", 6, 28, 3, ("F26",), (), False),
        ("F29", "Контроль качества входных данных", 8, 34, 4, ("F08",), (), False),
        ("F30", "Обнаружение аномалий", 9, 54, 8, ("F08", "F29"), (), False),
        ("F31", "Локальный режим без облака", 7, 36, 4, ("F01",), (), False),
        ("F32", "Интеграция с облачной аналитикой", 8, 46, 7, ("F08",), ("F31",), False),
        ("F33", "Мультитенантность", 6, 44, 7, ("F02",), (), False),
        ("F34", "Управление версиями ПО контроллеров", 8, 48, 6, ("F04",), (), False),
        ("F35", "Расписание регламентных работ", 7, 34, 4, ("F17",), (), False),
        ("F36", "Панель KPI производства", 8, 38, 4, ("F10", "F14"), (), False),
    ]
    return [Feature(*r) for r in rows]


FEATURES = feature_set()
INDEX = {f.id: i for i, f in enumerate(FEATURES)}
N = len(FEATURES)
BUDGET_HOURS = 520.0
TOTAL_VALUE = sum(f.value for f in FEATURES)
TOTAL_HOURS = sum(f.hours for f in FEATURES)
TOTAL_RISK = sum(f.risk for f in FEATURES)
MANDATORY = np.array([f.mandatory for f in FEATURES], dtype=bool)
REQUIRES = [(i, INDEX[d]) for i, f in enumerate(FEATURES) for d in f.requires]
# Конфликт симметричен, даже если указан только у одной функции.
CONFLICTS = sorted({
    (min(i, INDEX[c]), max(i, INDEX[c]))
    for i, f in enumerate(FEATURES) for c in f.conflicts
})


def save_dataset() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Синтетический набор функций системы промышленной автоматизации",
        "instance_version": "517-24-v1",
        "construction": "Детерминированный учебный сценарий: названия, зависимости и альтернативы заданы явно; параметры ценности, времени и риска зафиксированы в таблице исходных данных.",
        "budget_hours": BUDGET_HOURS,
        "features": [f.__dict__ for f in FEATURES],
    }
    DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def decode(bits: np.ndarray) -> np.ndarray:
    """Ремонтирует хромосому до допустимого набора функций."""
    x = np.asarray(bits, dtype=bool).copy()
    x |= MANDATORY

    # Добавляем обязательные зависимости.
    changed = True
    while changed:
        changed = False
        for i, j in REQUIRES:
            if x[i] and not x[j]:
                x[j] = True
                changed = True

    def remove_with_dependents(idx: int) -> None:
        x[idx] = False
        changed_inner = True
        while changed_inner:
            changed_inner = False
            for i, j in REQUIRES:
                if x[i] and not x[j]:
                    x[i] = False
                    changed_inner = True

    # Разрушаем конфликты, сохраняя обязательные функции и более выгодный элемент.
    for i, j in CONFLICTS:
        if x[i] and x[j]:
            if MANDATORY[i]:
                remove_with_dependents(j)
            elif MANDATORY[j]:
                remove_with_dependents(i)
            else:
                fi, fj = FEATURES[i], FEATURES[j]
                si = fi.value / (fi.hours * (1.0 + 0.08 * fi.risk))
                sj = fj.value / (fj.hours * (1.0 + 0.08 * fj.risk))
                remove_with_dependents(j if si >= sj else i)

    hours = np.array([f.hours for f in FEATURES], dtype=float)
    values = np.array([f.value for f in FEATURES], dtype=float)
    # Снимаем малополезные необязательные блоки, пока не выполнен лимит времени.
    while float(hours @ x) > BUDGET_HOURS + 1e-9:
        options = []
        for i in np.flatnonzero(x & ~MANDATORY):
            trial = x.copy()
            trial[i] = False
            again = True
            while again:
                again = False
                for child, parent in REQUIRES:
                    if trial[child] and not trial[parent]:
                        trial[child] = False
                        again = True
            removed = x & ~trial
            freed = float(hours @ removed)
            if freed > 0:
                lost_value = float(values @ removed)
                options.append((lost_value / freed, -freed, int(i), trial))
        if not options:
            raise RuntimeError("Не удалось восстановить допустимость при заданном бюджете")
        options.sort(key=lambda z: (z[0], z[1], z[2]))
        x = options[0][3]

    # Удаление необязательной функции не должно нарушать конфликтность/зависимости.
    if np.any(MANDATORY & ~x) or any(x[i] and not x[j] for i, j in REQUIRES):
        raise AssertionError("Ремонт нарушил обязательную функцию или зависимость")
    if any(x[i] and x[j] for i, j in CONFLICTS):
        raise AssertionError("Ремонт оставил конфликт функций")
    if float(hours @ x) > BUDGET_HOURS + 1e-9:
        raise AssertionError("Ремонт превысил бюджет времени")
    return x


def evaluate(bits: np.ndarray) -> tuple[float, float, float]:
    x = decode(bits)
    value = sum(f.value for f, keep in zip(FEATURES, x) if keep)
    hours = sum(f.hours for f, keep in zip(FEATURES, x) if keep)
    risk = sum(f.risk for f, keep in zip(FEATURES, x) if keep)
    return (1.0 - value / TOTAL_VALUE, hours / TOTAL_HOURS, risk / TOTAL_RISK)


def raw_metrics(bits: np.ndarray) -> dict[str, float]:
    x = decode(bits)
    return {
        "value": float(sum(f.value for f, keep in zip(FEATURES, x) if keep)),
        "hours": float(sum(f.hours for f, keep in zip(FEATURES, x) if keep)),
        "risk": float(sum(f.risk for f, keep in zip(FEATURES, x) if keep)),
        "selected": int(x.sum()),
    }


def dominates(a: Iterable[float], b: Iterable[float]) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_sort(F: np.ndarray) -> list[list[int]]:
    F = np.asarray(F, dtype=float)
    n = len(F)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt
    dominated_by = dom.sum(axis=0).astype(int)
    dominates_set = [np.flatnonzero(dom[i]).tolist() for i in range(n)]
    first = np.flatnonzero(dominated_by == 0).tolist()
    fronts = [first]
    k = 0
    while k < len(fronts) and fronts[k]:
        nxt = []
        for p in fronts[k]:
            for q in dominates_set[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    nxt.append(q)
        if nxt:
            fronts.append(nxt)
        k += 1
    return fronts


def crowding_distance(F: np.ndarray, front: list[int]) -> dict[int, float]:
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for m in range(F.shape[1]):
        order = sorted(front, key=lambda i: F[i, m])
        distance[order[0]] = distance[order[-1]] = math.inf
        lo, hi = F[order[0], m], F[order[-1], m]
        if hi == lo:
            continue
        for j in range(1, len(order) - 1):
            distance[order[j]] += (F[order[j + 1], m] - F[order[j - 1], m]) / (hi - lo)
    return distance


def environmental_selection(X: np.ndarray, F: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    chosen = []
    for front in nondominated_sort(F):
        if len(chosen) + len(front) <= size:
            chosen.extend(front)
        else:
            d = crowding_distance(F, front)
            chosen.extend(sorted(front, key=lambda i: d[i], reverse=True)[:size - len(chosen)])
            break
    idx = np.asarray(chosen, dtype=int)
    return X[idx].copy(), F[idx].copy()


def selection_metadata(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fronts = nondominated_sort(F)
    rank = np.empty(len(F), dtype=int)
    crowd = np.zeros(len(F))
    for r, front in enumerate(fronts):
        rank[front] = r
        cd = crowding_distance(F, front)
        for i, val in cd.items():
            crowd[i] = val
    return rank, crowd


def tournament_nsga(X: np.ndarray, F: np.ndarray, rng: np.random.Generator,
                    rank: np.ndarray | None = None, crowd: np.ndarray | None = None) -> np.ndarray:
    if rank is None or crowd is None:
        rank, crowd = selection_metadata(F)
    a, b = rng.integers(len(X), size=2)
    if rank[a] != rank[b]:
        return X[a if rank[a] < rank[b] else b]
    return X[a if crowd[a] >= crowd[b] else b]


def crossover_mutate(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, mutation_rate: float) -> np.ndarray:
    child = np.where(rng.random(N) < 0.5, a, b)
    flip = rng.random(N) < mutation_rate
    child = np.logical_xor(child, flip)
    return decode(child)


def pareto_front(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = np.array([evaluate(x) for x in X])
    keep = []
    for i in range(len(X)):
        if not any(dominates(F[j], F[i]) for j in range(len(X)) if j != i):
            keep.append(i)
    # Collapse identical phenotypes so the reported front is readable.
    seen = set()
    uniq = []
    for i in keep:
        key = tuple(decode(X[i]).astype(int))
        if key not in seen:
            seen.add(key)
            uniq.append(i)
    return X[uniq].copy(), F[uniq].copy()


def hypervolume_3d(F: np.ndarray, ref: float = 1.05) -> float:
    """Точный объём объединения прямоугольных боксов для 3D минимизации."""
    if len(F) == 0:
        return 0.0
    F = np.asarray(F, dtype=float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    xs = sorted(set([float(v) for v in F[:, 0]] + [ref]))
    volume = 0.0
    for x0, x1 in zip(xs[:-1], xs[1:]):
        active = F[F[:, 0] <= x0 + 1e-12, 1:]
        if len(active) == 0:
            continue
        ys = sorted(set([float(v) for v in active[:, 0]] + [ref]))
        area = 0.0
        for y0, y1 in zip(ys[:-1], ys[1:]):
            zs = active[active[:, 0] <= y0 + 1e-12, 1]
            if len(zs):
                area += (y1 - y0) * max(0.0, ref - float(np.min(zs)))
        volume += (x1 - x0) * area
    return float(volume)


def run_nsga(seed: int, pop_size: int = 100, generations: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    X = np.array([decode(rng.random(N) < 0.35) for _ in range(pop_size)], dtype=bool)
    F = np.array([evaluate(x) for x in X])
    history = [hypervolume_3d(F[nondominated_sort(F)[0]])]
    for _ in range(generations):
        rank, crowd = selection_metadata(F)
        children = []
        for _ in range(pop_size):
            p1 = tournament_nsga(X, F, rng, rank, crowd)
            p2 = tournament_nsga(X, F, rng, rank, crowd)
            children.append(crossover_mutate(p1, p2, rng, 1.0 / N))
        Y = np.asarray(children, dtype=bool)
        G = np.array([evaluate(x) for x in Y])
        X, F = environmental_selection(np.vstack([X, Y]), np.vstack([F, G]), pop_size)
        front = nondominated_sort(F)[0]
        history.append(hypervolume_3d(F[front]))
    X, F = pareto_front(X)
    return {"X": X, "F": F, "history": np.asarray(history), "evaluations": pop_size * (generations + 1)}


def scalar_ga(seed: int, weights: np.ndarray, pop_size: int = 20, generations: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    X = np.array([decode(rng.random(N) < 0.35) for _ in range(pop_size)], dtype=bool)
    F = np.array([evaluate(x) for x in X])
    history = []
    for gen in range(generations + 1):
        score = F @ weights
        order = np.argsort(score)
        X, F = X[order], F[order]
        history.append((X.copy(), F.copy()))
        if gen == generations:
            break
        nxt = [X[0].copy(), X[1].copy()]
        while len(nxt) < pop_size:
            a, b = rng.integers(pop_size, size=2)
            p1 = X[min(a, b)]
            a, b = rng.integers(pop_size, size=2)
            p2 = X[min(a, b)]
            nxt.append(crossover_mutate(p1, p2, rng, 1.0 / N))
        X = np.asarray(nxt, dtype=bool)
        F = np.array([evaluate(x) for x in X])
    return {"X": X, "F": F, "history": history, "evaluations": pop_size * (generations + 1)}


WEIGHTS = np.array([
    [0.70, 0.15, 0.15],
    [0.15, 0.70, 0.15],
    [0.15, 0.15, 0.70],
    [0.45, 0.35, 0.20],
    [1 / 3, 1 / 3, 1 / 3],
], dtype=float)


def as_rows(X: np.ndarray, F: np.ndarray, method: str, seed: int) -> list[dict]:
    rows = []
    for bits, obj in zip(X, F):
        m = raw_metrics(bits)
        rows.append({"method": method, "seed": seed, **m,
                     "value_share": m["value"] / TOTAL_VALUE,
                     "objective_value_min": float(obj[0]),
                     "objective_hours": float(obj[1]),
                     "objective_risk": float(obj[2]),
                     "features": ";".join(f.id for f, keep in zip(FEATURES, decode(bits)) if keep)})
    return rows


def run_campaign(runs: int = 30, first_seed: int = 3000, pop_size: int = 100, generations: int = 100) -> dict:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(first_seed, first_seed + runs))
    records, histories, pooled = [], {"NSGA-II": [], "Weighted sum GA": []}, {"NSGA-II": [], "Weighted sum GA": []}
    start = time.perf_counter()
    for seed in seeds:
        method_start = time.perf_counter()
        ns = run_nsga(seed, pop_size, generations)
        nsga_seconds = time.perf_counter() - method_start
        method_start = time.perf_counter()
        wx, wf, wh = [], [], []
        scalar_pop = pop_size // len(WEIGHTS)
        scalar_budget = scalar_pop * (generations + 1) * len(WEIGHTS)
        nsga_budget = pop_size * (generations + 1)
        if scalar_budget != nsga_budget:
            raise AssertionError("Бюджеты сравниваемых методов не совпадают")
        for wi, w in enumerate(WEIGHTS):
            res = scalar_ga(seed + 100_000 + wi, w, scalar_pop, generations)
            wx.extend(res["X"])
            wf.extend(res["F"])
            wh.append(res["history"])
        weighted_seconds = time.perf_counter() - method_start
        wx, wf = pareto_front(np.asarray(wx, dtype=bool))
        ns_hv = hypervolume_3d(ns["F"])
        ws_hv = hypervolume_3d(wf)
        records.append({"seed": seed, "nsga_hv": ns_hv, "weighted_hv": ws_hv,
                        "nsga_front_size": len(ns["F"]), "weighted_front_size": len(wf),
                        "nsga_evaluations": ns["evaluations"], "weighted_evaluations": scalar_budget,
                        "nsga_time_s": nsga_seconds, "weighted_time_s": weighted_seconds})
        histories["NSGA-II"].append(ns["history"])
        # объединяем промежуточные популяции 5 скаляризаций в каждом поколении.
        w_hist = []
        for g in range(generations + 1):
            gx = np.asarray([h[g][0] for h in wh], dtype=bool).reshape(-1, N)
            gf = np.asarray([h[g][1] for h in wh], dtype=float).reshape(-1, 3)
            _, frontf = pareto_front(gx)
            w_hist.append(hypervolume_3d(frontf))
        histories["Weighted sum GA"].append(np.asarray(w_hist))
        pooled["NSGA-II"].append(as_rows(ns["X"], ns["F"], "NSGA-II", seed))
        pooled["Weighted sum GA"].append(as_rows(wx, wf, "Weighted sum GA", seed))
        if seed == seeds[0]:
            write_csv(out / "front_nsga2_seed3000.csv", as_rows(ns["X"], ns["F"], "NSGA-II", seed))
            write_csv(out / "front_weighted_seed3000.csv", as_rows(wx, wf, "Weighted sum GA", seed))
        print(f"seed={seed} NSGA-II HV={ns_hv:.5f} weighted={ws_hv:.5f}")
    elapsed = time.perf_counter() - start
    for r in records:
        r["nsga_time_s"] = round(r["nsga_time_s"], 6)
        r["weighted_time_s"] = round(r["weighted_time_s"], 6)
    write_csv(out / "runs.csv", records)
    allrows = [row for method in pooled.values() for seedrows in method for row in seedrows]
    write_csv(out / "pareto_solutions.csv", allrows)
    hist = {k: np.asarray(v, dtype=float).tolist() for k, v in histories.items()}
    (out / "convergence.json").write_text(json.dumps(hist), encoding="utf-8")
    summary = summarize(records, pooled, elapsed, pop_size, generations)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(records: list[dict], pooled: dict, elapsed: float, pop_size: int, generations: int) -> dict:
    a = np.array([r["nsga_hv"] for r in records])
    b = np.array([r["weighted_hv"] for r in records])
    delta = a - b
    rng = np.random.default_rng(90417)
    boot = np.array([np.mean(rng.choice(delta, size=len(delta), replace=True)) for _ in range(10000)])
    def stats(x: np.ndarray) -> dict:
        return {"mean": float(np.mean(x)), "median": float(np.median(x)), "std": float(np.std(x, ddof=1)),
                "min": float(np.min(x)), "max": float(np.max(x))}
    fronts = {}
    for name, groups in pooled.items():
        rows = [r for group in groups for r in group]
        fronts[name] = rows
    nsga_rows = fronts["NSGA-II"]
    # недоминируемая объединённая популяция из всех seed используется только для подбора примеров.
    feature_rows = []
    seen = set()
    for r in nsga_rows:
        key = r["features"]
        if key not in seen:
            seen.add(key)
            feature_rows.append(r)
    # Отбираем глобальные компромиссные решения по трём понятным правилам.
    value_best = max(feature_rows, key=lambda r: r["value"])
    time_best = min(feature_rows, key=lambda r: r["hours"])
    risk_best = min(feature_rows, key=lambda r: r["risk"])
    balanced = min(feature_rows, key=lambda r: math.sqrt(r["objective_value_min"]**2 + r["objective_hours"]**2 + r["objective_risk"]**2))
    reps = []
    for label, r in [("Максимальная ценность", value_best), ("Минимальный срок", time_best),
                     ("Минимальный риск", risk_best), ("Сбалансированное", balanced)]:
        if not any(x["features"] == r["features"] for x in reps):
            reps.append({"label": label, **r})
    return {
        "title": "Многокритериальный выбор функций продукта промышленной автоматизации",
        "group": "517", "student_number": 24, "variant": 2, "Q": 1, "Y": 26,
        "variant_formula": "1 + ((17*24 + 7*1 + 26) mod 20) = 2",
        "problem": {"feature_count": N, "mandatory_count": int(MANDATORY.sum()), "time_budget_hours": BUDGET_HOURS,
                    "objectives": ["максимизация ценности", "минимизация времени", "минимизация технического риска"],
                    "data": "synthetic, fixed in data/features.json", "dataset_version": "517-24-v1",
                    "construction": "Детерминированный учебный сценарий с явно заданными параметрами; случайная генерация исходных оценок не применяется."},
        "experiment": {"runs": len(records), "seeds": [records[0]["seed"], records[-1]["seed"]],
                       "population_nsga": pop_size, "scalar_population_per_weight": pop_size // len(WEIGHTS),
                       "generations": generations, "weights_count": len(WEIGHTS),
                       "evaluations_per_run_each": pop_size * (generations + 1), "weights": WEIGHTS.tolist(),
                       "elapsed_seconds": elapsed},
        "metrics": {"hypervolume_reference": [1.05, 1.05, 1.05], "nsga": stats(a), "weighted_sum": stats(b),
                    "paired_delta_nsga_minus_weighted": stats(delta),
                    "paired_delta_bootstrap_95ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                    "nsga_wins": int(np.sum(delta > 0)), "ties": int(np.sum(np.isclose(delta, 0))),
                    "weighted_wins": int(np.sum(delta < 0))},
        "representative_solutions": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР3: NSGA-II против взвешенной суммы")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=3000)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 20:
        raise SystemExit("Для отчётной серии требуется не менее 20 независимых запусков")
    if args.population < len(WEIGHTS) * 4:
        raise SystemExit("Популяция слишком мала для равного сравнения пяти весовых векторов")
    save_dataset()
    summary = run_campaign(args.runs, args.first_seed, args.population, args.generations)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

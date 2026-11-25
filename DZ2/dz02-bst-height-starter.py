#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ДЗ 2. Эксперимент: высота BST и деградация операций.

Запуск: python dz02_bst_height.py --variant N
Результаты: results_vN.csv, fig_height_vN.png, fig_depth_vN.png, fig_search_vN.png
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time

# ---------------------------------------------------------------------------
# 1. Узел и итеративная вставка
# ---------------------------------------------------------------------------


class Node:
    """Узел BST: ключ и ссылки на левое и правое поддеревья."""

    __slots__ = ("key", "left", "right")

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None


def insert_iter(root: Node | None, key: int) -> Node:
    """Итеративная вставка ключа в BST; возвращает корень дерева.

    Дубликаты игнорируются (ключи в эксперименте уникальны).
    """
    new = Node(key)
    if root is None:
        return new
    node = root
    while True:
        if key < node.key:
            if node.left is None:
                node.left = new
                break
            node = node.left
        elif key > node.key:
            if node.right is None:
                node.right = new
                break
            node = node.right
        else:  # дубликат
            break
    return root


def build_bst(keys: list[int]) -> Node | None:
    """Строит BST последовательной итеративной вставкой ключей."""
    root: Node | None = None
    for key in keys:
        root = insert_iter(root, key)
    return root


# ---------------------------------------------------------------------------
# 2. Измеряемые величины: высота и средняя глубина узла
# ---------------------------------------------------------------------------


def _depth_walk(root: Node | None):
    """Итеративный обход в глубину, выдаёт глубину каждого узла."""
    if root is None:
        return
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        yield d
        if node.left is not None:
            stack.append((node.left, d + 1))
        if node.right is not None:
            stack.append((node.right, d + 1))


def tree_height(root: Node | None) -> int:
    """Высота: число рёбер на самом длинном пути от корня. Пустое: -1."""
    h = -1
    for d in _depth_walk(root):
        if d > h:
            h = d
    return h


def sum_depths(root: Node | None) -> int:
    """Сумма глубин всех узлов (глубина корня — 0)."""
    return sum(_depth_walk(root))


def count_nodes(root: Node | None) -> int:
    """Число узлов дерева."""
    return sum(1 for _ in _depth_walk(root))


def avg_depth(root: Node | None) -> float:
    """Средняя глубина узла: sum_depths / count_nodes (0.0 для пустого)."""
    n = count_nodes(root)
    return sum_depths(root) / n if n else 0.0


def inorder_keys(root: Node | None) -> list[int]:
    """Ключи in-order-обхода (итеративно, с явным стеком)."""
    out: list[int] = []
    stack: list[Node] = []
    node = root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.key)
        node = node.right
    return out


# ---------------------------------------------------------------------------
# 3. Верификация
# ---------------------------------------------------------------------------


def self_check() -> None:
    """Граничные деревья, аналитические значения, сверка с sorted()."""
    assert tree_height(None) == -1
    assert avg_depth(None) == 0.0
    assert count_nodes(None) == 0
    assert inorder_keys(None) == []
    single = build_bst([42])
    assert tree_height(single) == 0
    assert count_nodes(single) == 1
    assert avg_depth(single) == 0.0

    small = build_bst([2, 1, 3])
    assert tree_height(small) == 1
    assert sum_depths(small) == 2

    # дубликаты игнорируются
    dup = build_bst([5, 5, 5])
    assert count_nodes(dup) == 1

    # вырожденные деревья: вправо (1..n) и влево (n..1), разные n
    for n in (2, 10, 100, 10_000):
        for keys in (list(range(1, n + 1)), list(range(n, 0, -1))):
            d = build_bst(keys)
            assert tree_height(d) == n - 1
            assert avg_depth(d) == (n - 1) / 2
            assert count_nodes(d) == n

    # идеально сбалансированное дерево 2^k - 1 узлов: h = k-1, sum = sum i*2^i
    k = 6

    def balanced(lo: int, hi: int) -> list[int]:
        if lo > hi:
            return []
        mid = (lo + hi) // 2
        return [mid] + balanced(lo, mid - 1) + balanced(mid + 1, hi)

    m = 2**k - 1
    perfect = build_bst(balanced(1, m))
    assert tree_height(perfect) == k - 1
    assert sum_depths(perfect) == sum(i * 2**i for i in range(k))

    # инвариант BST: in-order == sorted() на случайных наборах
    rng = random.Random(0)
    for _ in range(200):
        keys = rng.sample(range(-1000, 1000), rng.randint(1, 60))
        root = build_bst(keys)
        assert inorder_keys(root) == sorted(keys)
        assert count_nodes(root) == len(keys)
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 4. Эксперимент
# ---------------------------------------------------------------------------

SIZES = [100, 300, 1_000, 3_000, 10_000]
REPEATS = 5
N_QUERIES = 1_000
TRIALS = 20  # независимых случайных деревьев на каждое n (для среднего и разброса)


def random_keys(n: int, rng: random.Random) -> list[int]:
    """n уникальных случайных ключей (детерминированно по seed варианта)."""
    return rng.sample(range(10 * n), n)


def ordered_keys(n: int) -> list[int]:
    """n упорядоченных ключей 1..n — худший поток вставок для BST."""
    return list(range(1, n + 1))


def bench_search(root: Node | None, queries: list[int]) -> float:
    """Медиана времени поиска всех ключей queries по REPEATS запускам."""

    def run() -> None:
        for q in queries:
            node = root
            while node is not None and node.key != q:
                node = node.left if q < node.key else node.right

    run()  # прогрев
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def expected_random_height(n: int) -> float:
    """Асимптотика высоты случайного BST (Reed, 2003): 4.311 ln n - 1.953 ln ln n."""
    return 4.311 * math.log(n) - 1.953 * math.log(math.log(n))


def expected_random_avg_depth(n: int) -> float:
    """Точное E[средняя глубина] случайного BST: 2(1+1/n)H_n - 4."""
    h = sum(1.0 / i for i in range(1, n + 1))
    return 2 * (1 + 1 / n) * h - 4


def run_experiment(seed: int, variant: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    print(f"{'n':>7} {'h_случ':>7} {'h_ср20':>7} {'d_случ':>7} {'h_упор':>7} "
          f"{'d_упор':>8} {'log2n':>6} {'t_случ,мс':>10} {'t_упор,мс':>10}")
    for n in SIZES:
        # основная выборка варианта — первое дерево; остальные — для усреднения
        trial_h, trial_d = [], []
        for t in range(TRIALS):
            r = build_bst(random_keys(n, rng))
            trial_h.append(tree_height(r))
            trial_d.append(avg_depth(r))
            if t == 0:
                root_rand = r
                keys_rand = inorder_keys(r)
        root_ord = build_bst(ordered_keys(n))
        # одинаковые по позиции запросы: индексы случайные, ключи из своего дерева
        idx = [rng.randrange(n) for _ in range(N_QUERIES)]
        q_rand = [keys_rand[i] for i in idx]
        q_ord = [i + 1 for i in idx]
        t_rand = bench_search(root_rand, q_rand) * 1000
        t_ord = bench_search(root_ord, q_ord) * 1000
        row = {
            "n": n,
            "h_random": tree_height(root_rand),
            "h_random_mean": statistics.mean(trial_h),
            "h_random_min": min(trial_h),
            "h_random_max": max(trial_h),
            "avgdepth_random": avg_depth(root_rand),
            "avgdepth_random_mean": statistics.mean(trial_d),
            "h_ordered": tree_height(root_ord),
            "avgdepth_ordered": avg_depth(root_ord),
            "log2n": math.log2(n),
            "h_theory_random": expected_random_height(n),
            "avgdepth_theory_random": expected_random_avg_depth(n),
            "avl_bound": 1.44 * math.log2(n),
            "search_ms_random": t_rand,
            "search_ms_ordered": t_ord,
        }
        rows.append(row)
        print(f"{n:>7} {row['h_random']:>7} {row['h_random_mean']:>7.1f} "
              f"{row['avgdepth_random']:>7.1f} {row['h_ordered']:>7} "
              f"{row['avgdepth_ordered']:>8.1f} {row['log2n']:>6.1f} "
              f"{t_rand:>10.2f} {t_ord:>10.2f}")
    with open(f"results_v{variant}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def make_plots(rows: list[dict], variant: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    grid = [10 ** (2 + 2 * i / 100) for i in range(101)]

    # h(n)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(ns, [r["h_random"] for r in rows], "o-", color="#1f77b4", label="BST, случайный поток")
    ax.plot(ns, [r["h_ordered"] for r in rows], "s-", color="#d62728", label="BST, упорядоченный поток")
    ax.plot(grid, [math.log2(x) for x in grid], "--", color="gray", label="log₂(n)")
    ax.plot(grid, grid, ":", color="black", label="n")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (число ключей)")
    ax.set_ylabel("высота h")
    ax.set_title(f"Высота BST, вариант {variant} (лог-лог шкала)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"fig_height_v{variant}.png", dpi=150)
    plt.close(fig)

    # средняя глубина
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(ns, [r["avgdepth_random"] for r in rows], "o-", color="#1f77b4", label="случайный поток")
    ax.plot(ns, [r["avgdepth_ordered"] for r in rows], "s-", color="#d62728", label="упорядоченный поток")
    ax.plot(grid, [math.log2(x) for x in grid], "--", color="gray", label="log₂(n)")
    ax.plot(grid, [(x - 1) / 2 for x in grid], ":", color="black", label="(n − 1)/2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (число ключей)")
    ax.set_ylabel("средняя глубина узла")
    ax.set_title(f"Средняя глубина узла, вариант {variant} (лог-лог шкала)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"fig_depth_v{variant}.png", dpi=150)
    plt.close(fig)

    # время поиска
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(ns, [r["search_ms_random"] for r in rows], "o-", color="#1f77b4", label="случайный поток")
    ax.plot(ns, [r["search_ms_ordered"] for r in rows], "s-", color="#d62728", label="упорядоченный поток")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (число ключей)")
    ax.set_ylabel(f"время {N_QUERIES} поисков, мс (медиана)")
    ax.set_title(f"Время поиска, вариант {variant}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"fig_search_v{variant}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    rows = run_experiment(seed, args.variant)
    make_plots(rows, args.variant)


if __name__ == "__main__":
    main()
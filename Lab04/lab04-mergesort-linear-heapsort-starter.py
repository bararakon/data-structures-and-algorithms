#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности — стартовая заготовка.

Запуск: python lab04-mergesort-linear-heapsort-starter.py --variant 9
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# 1. Сортировки (реализуются вручную; sorted() используется только как эталон)
# ---------------------------------------------------------------------------


def merge(left: list, right: list, key) -> list:
    """Стабильное слияние двух отсортированных списков в новый список.

    Стабильность: при равенстве ключей элемент из left идёт раньше.
    Ожидаемая сложность: O(n) (обосновать в отчёте).
    """
    out = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def merge_sort(a: list, key=lambda x: x) -> list:
    """MergeSort: возвращает новый отсортированный список. Ожидаемая сложность: O(n log n)."""
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], key=key)
    right = merge_sort(a[mid:], key=key)
    return merge(left, right, key)


def counting_sort(a: list[int], key_max: int) -> list[int]:
    """Устойчивая сортировка подсчётом: ключи — целые из диапазона [0, key_max].

    Схема: массив счётчиков -> префиксные суммы -> расстановка с конца.
    Ожидаемые время и память: Время O(n + k), Память O(n + k) — где n и k = key_max + 1;
    """
    if not a:
        return []
    count = [0] * (key_max + 1)
    for x in a:
        count[x] += 1
    for i in range(1, key_max + 1):
        count[i] += count[i - 1]
    
    out = [0] * len(a)
    for x in reversed(a):
        count[x] -= 1
        out[count[x]] = x
    return out


def radix_sort_lsd(a: list[int]) -> list[int]:
    """RadixSort LSD по десятичным разрядам, неотрицательные целые.

    Поверх устойчивой поразрядной counting-сортировки: от младшего разряда
    к старшему. Ожидаемая сложность: O(d * (n + b)) где d - кол-во разрядов, b - основание (10).
    """
    if not a:
        return []
    max_val = max(a)
    exp = 1
    out = [0] * len(a)
    while max_val // exp > 0:
        count = [0] * 10
        for x in a:
            count[(x // exp) % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for x in reversed(a):
            d = (x // exp) % 10
            count[d] -= 1
            out[count[d]] = x
        a = list(out)
        exp *= 10
    return a


def sift_down(a: list[int], i: int, n: int) -> None:
    """Просейка вниз элемента i в куче a[0:n] (max-heap), на месте."""
    while True:
        big = i
        l, r = 2 * i + 1, 2 * i + 2
        if l < n and a[l] > a[big]:
            big = l
        if r < n and a[r] > a[big]:
            big = r
        if big == i:
            return
        a[i], a[big] = a[big], a[i]
        i = big


def heap_sort(a: list[int]) -> None:
    """HeapSort in-place: сортирует список a на месте, без вспомогательного массива.

    Схема: построение max-кучи просейками вниз (снизу вверх, O(n)),
    затем обмен вершины с последним элементом и просейка. Сложность: O(n log n).
    """
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)


def quick_sort(a: list[int]) -> list[int]:
    """QuickSort с рандомизацией опорного элемента — перенести из своей ЛР 3."""
    if len(a) <= 1:
        return list(a)
    pivot = a[random.randrange(len(a))]
    left = [x for x in a if x < pivot]
    middle = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


# ---------------------------------------------------------------------------
# 2. Верификация: инварианты + сверка с эталоном sorted()
#    (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def heap_sorted(a: list[int]) -> list[int]:
    """heap_sort сортирует на месте: проверяется тот список, который ему передали."""
    buf = list(a)
    heap_sort(buf)
    return buf


# Каждая функция возвращает отсортированный список; ключи — целые из [0, 999].
SORTS = {
    "merge_sort": lambda a: merge_sort(a),
    "counting_sort": lambda a: counting_sort(a, key_max=999),
    "radix_sort_lsd": lambda a: radix_sort_lsd(a),
    "heap_sort": lambda a: heap_sorted(a),
    "quick_sort": lambda a: quick_sort(a),
}


def check_sort(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [0],                   # нулевой ключ
        [5, 5, 5],             # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
    ]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(res == sorted(a), f"на {a} получено {res}, ожидалось {sorted(a)}")

    # Случайные тесты: инвариант перестановки + сверка с эталоном
    rng = random.Random(0)
    for _ in range(200):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        res = fn(list(a))
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")


def check_merge_stability() -> None:
    """Стабильность слияния на парах (ключ, порядковый номер): эталон sorted() стабилен,
    поэтому при равных ключах порядок номеров обязан совпасть."""
    rng = random.Random(0)
    pairs = [(rng.randint(0, 5), i) for i in range(100)]
    got = merge_sort(pairs, key=lambda p: p[0])
    expect(got == sorted(pairs, key=lambda p: p[0]),
           "merge_sort нестабилен: при равных ключах нарушен исходный порядок номеров")


def check_heap_property() -> None:
    """Проверка свойства кучи после этапа построения в heap_sort."""
    rng = random.Random(0)
    a = [rng.randint(0, 100) for _ in range(50)]
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)
    for i in range(n):
        l, r = 2 * i + 1, 2 * i + 2
        if l < n:
            expect(a[i] >= a[l], "Нарушено свойство max-heap: родитель меньше левого потомка")
        if r < n:
            expect(a[i] >= a[r], "Нарушено свойство max-heap: родитель меньше правого потомка")

def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.append(("стабильность merge_sort", check_merge_stability))
    checks.append(("свойство кучи (собственная проверка)", check_heap_property))
    
    failed = 0
    print("self_check:")
    for name, check in checks:
        try:
            check()
        except NotImplementedError as err:
            where = traceback.extract_tb(err.__traceback__)[-1].name
            print(f"  {name}: не реализовано — {where} (TODO)")
            failed += 1
        except SelfCheckError as err:
            print(f"  {name}: ОШИБКА — {err}")
            failed += 1
        except Exception: 
            print(f"  {name}: ИСКЛЮЧЕНИЕ")
            traceback.print_exc(file=sys.stdout)
            failed += 1
        else:
            print(f"  {name}: OK")
            
    if failed:
        raise SystemExit(f"self_check: не пройдено разделов — {failed}; "
                         f"замеры выполняются только после всех проверок.")
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100        # узкий диапазон: counting уместен
WIDE_KEY_MAX = 10**9       # широкий диапазон: counting теряет смысл (обсудить в отчёте)


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Массивы разных распределений и диапазонов ключей для размера n."""
    almost = list(range(n))
    for _ in range(max(1, n // 100)):  # ~1 % случайных обменов
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn, data: list[int]) -> float:
    """Медиана времени fn на копии data по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_benchmarks(seed: int) -> None:
    rng = random.Random(seed)
    algorithms = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort (ЛР 3)", quick_sort),
        ("sorted() (эталон)", sorted),
    ]
    
    results = {algo[0]: {dist: [] for dist in ["random_wide", "random_small", "sorted", "reversed", "almost_sorted"]} for algo in algorithms}

    for n in SIZES:
        datasets = make_datasets(rng, n)
        print(f"\nn = {n}:")
        for dist_name, data in datasets.items():
            print(f"  распределение {dist_name}:")
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":
                    print(f"    {algo_name:>18}: пропуск (диапазон ключей)")
                    continue
                median_time = bench(fn, data)
                results[algo_name][dist_name].append(median_time)
                print(f"    {algo_name:>18}: t = {median_time:.6f} c")
                
    try:
        import matplotlib.pyplot as plt
        for dist_name in ["random_wide", "random_small", "sorted", "reversed", "almost_sorted"]:
            plt.figure(figsize=(8, 6))
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":
                    continue
                plt.plot(SIZES, results[algo_name][dist_name], marker='o', label=algo_name)
            
            plt.title(f"Время работы от размера n (Вход: {dist_name})")
            plt.xlabel("Размер массива n")
            plt.ylabel("Время, с")
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"benchmark_{dist_name}.png")
            plt.close()
        print("\nГрафики сохранены в формате .png (benchmark_<распределение>.png)")
    except ImportError:
        print("\nБиблиотека matplotlib не установлена, построение графиков пропущено.")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    run_benchmarks(seed)


if __name__ == "__main__":
    main()
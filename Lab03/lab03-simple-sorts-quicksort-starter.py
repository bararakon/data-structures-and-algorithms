#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-03 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Запуск: python lab03-simple-sorts-quicksort-starter.py --variant N
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
import traceback

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# 1. Простые сортировки со счётчиками (сортируют КОПИЮ входа, вход не меняют)
#    Каждая функция возвращает кортеж: (отсортированный список,
#    число сравнений ключей, число обменов/перемещений элементов).
# ---------------------------------------------------------------------------


def bubble_sort(a: list) -> tuple[list, int, int]:
    """Сортировка пузырьком. Ожидаемая сложность: Θ(n^2) в худшем/среднем, Θ(n) в лучшем. (обосновать в отчёте).

    Подсказка: досрочный выход, если за проход не было ни одного обмена, —
    именно он даёт лучший случай на упорядоченном входе.
    """
    result = list(a)
    comparisons = 0
    swaps = 0
    n = len(result)

    for end in range(n - 1, 0, -1):
        swapped = False
        for j in range(end):
            comparisons += 1
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swaps += 1
                swapped = True
        if not swapped:
            break
    return result, comparisons, swaps


def insertion_sort(a: list) -> tuple[list, int, int]:
    """Сортировка вставками. Ожидаемая сложность: Θ(n^2) - худший/средний, Θ(n) - лучший. (лучший/худший случаи)."""
    result = list(a)
    comparisons = 0
    moves = 0

    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        shifted = False
        while j >= 0:
            comparisons += 1
            if result[j] > key:
                result[j + 1] = result[j]
                moves += 1
                shifted = True
                j -= 1
            else:
                break
        if shifted:
            result[j + 1] = key
            moves += 1
    return result, comparisons, moves


def selection_sort(a: list) -> tuple[list, int, int]:
    """Сортировка выбором. Ожидаемая сложность: Θ(n^2) на любом входе. (почему не зависит от входа?)."""
    result = list(a)
    comparisons = 0
    swaps = 0
    n = len(result)

    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            comparisons += 1
            if result[j] < result[min_index]:
                min_index = j
        if min_index != i:
            result[i], result[min_index] = result[min_index], result[i]
            swaps += 1
    return result, comparisons, swaps


# ---------------------------------------------------------------------------
# 2. QuickSort с рандомизированным опорным элементом
# ---------------------------------------------------------------------------


def quick_sort(a: list, rng: random.Random | None = None) -> list:
    """QuickSort с выбором опорного через rng.randrange. Возвращает новый список.

    Ожидаемая сложность: в среднем O(n log n), в худшем случае O(n^2) (обосновать
    в отчёте, объяснить роль рандомизации). rng передаётся снаружи, чтобы
    запуск был воспроизводим по seed варианта.
    """
    if rng is None:
        rng = random.Random()
    result = list(a)

    def sort_range(lo: int, hi: int) -> None:
        while hi - lo > 1:
            pivot = result[rng.randrange(lo, hi)]
            lt, i, gt = lo, lo, hi
            # 3-путевое разбиение: [lo, lt) < pivot, [lt, gt) == pivot, [gt, hi) > pivot
            while i < gt:
                if result[i] < pivot:
                    result[lt], result[i] = result[i], result[lt]
                    lt += 1
                    i += 1
                elif result[i] > pivot:
                    gt -= 1
                    result[i], result[gt] = result[gt], result[i]
                else:
                    i += 1

            left_size = lt - lo
            right_size = hi - gt
            if left_size < right_size:
                if left_size > 1:
                    sort_range(lo, lt)
                lo = gt
            else:
                if right_size > 1:
                    sort_range(gt, hi)
                hi = lt

    sort_range(0, len(result))
    return result


# ---------------------------------------------------------------------------
# 3. Демонстрация стабильности на парах (ключ, метка)
# ---------------------------------------------------------------------------


class _KeyItem:
    """Обёртка над парой (key, label), чтобы сравнение шло строго по ключу p[0]."""

    def __init__(self, key: int, label: str) -> None:
        self.key = key
        self.label = label

    def __lt__(self, other: _KeyItem) -> bool:
        return self.key < other.key

    def __gt__(self, other: _KeyItem) -> bool:
        return self.key > other.key

    def __le__(self, other: _KeyItem) -> bool:
        return self.key <= other.key

    def __ge__(self, other: _KeyItem) -> bool:
        return self.key >= other.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _KeyItem):
            return NotImplemented
        return self.key == other.key

    def __repr__(self) -> str:
        return f"({self.key}, '{self.label}')"


def stability_demo() -> None:
    """Показать, какие из четырёх сортировок стабильны.

    План: взять массив пар (ключ, метка) с повторяющимися ключами, например
    [(2, "a"), (1, "b"), (2, "c"), (1, "d")], отсортировать каждой сортировкой
    по ключу (сравнение только по p[0]!) и напечатать порядок меток при равных
    ключах. Вывод о стабильности каждой сортировки включить в отчёт.
    """
    raw_pairs = [(2, "a"), (1, "b"), (2, "c"), (1, "d"), (2, "e")]
    items = [_KeyItem(k, lbl) for k, lbl in raw_pairs]

    print("\n--- Демонстрация стабильности ---")
    print(f"Исходный массив пар: {raw_pairs}")

    orig_by_key: dict[int, list[str]] = {}
    for k, lbl in raw_pairs:
        orig_by_key.setdefault(k, []).append(lbl)

    for name, fn in ALGORITHMS.items():
        res = fn(list(items))
        by_key: dict[int, list[str]] = {}
        for item in res:
            by_key.setdefault(item.key, []).append(item.label)

        is_stable = by_key == orig_by_key
        stable_str = "СТАБИЛЬНА" if is_stable else "НЕСТАБИЛЬНА"
        print(
            f"  {name:<15}: {res} | key=1: {by_key[1]}, key=2: {by_key[2]} -> {stable_str}"
        )


# ---------------------------------------------------------------------------
# 4. Классы входов (данные по варианту, детерминированно по seed)
# ---------------------------------------------------------------------------


def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    """Три класса входов размера n: упорядоченный, случайный, обратный."""
    rng = random.Random(seed + n)  # свой поток для каждого размера
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ---------------------------------------------------------------------------
# 5. Верификация (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def is_sorted(a: list) -> bool:
    """Инвариант 1: неубывающий порядок элементов."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def check_algorithm(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [5, 5, 5, 5],          # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
    ]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(is_sorted(res), f"нарушен порядок на {a}: получено {res}")
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")

    # Сверка с эталоном из стандартной библиотеки на сотнях случайных входов.
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        res = fn(list(a))
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")


def check_counters() -> None:
    """Счётчики: у сортировки выбором ровно n*(n-1)/2 сравнений на любом входе."""
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    expect(comparisons == 10,
           f"selection_sort: неверный счётчик сравнений — {comparisons}, ожидалось 10")

    # Дополнительные проверки счётчиков пузырька, вставок и стабильности
    _, cmp_b, swp_b = bubble_sort([1, 2, 3, 4, 5])
    expect(cmp_b == 4 and swp_b == 0, f"bubble_sort: ожидалось (4, 0), получено ({cmp_b}, {swp_b})")

    _, cmp_i, mov_i = insertion_sort([1, 2, 3, 4, 5])
    expect(cmp_i == 4 and mov_i == 0, f"insertion_sort: ожидалось (4, 0), получено ({cmp_i}, {mov_i})")

    test_pairs = [_KeyItem(2, "a"), _KeyItem(1, "b"), _KeyItem(2, "c")]
    res_b, _, _ = bubble_sort(test_pairs)
    res_i, _, _ = insertion_sort(test_pairs)
    expect([x.label for x in res_b if x.key == 2] == ["a", "c"], "bubble_sort: нарушена стабильность")
    expect([x.label for x in res_i if x.key == 2] == ["a", "c"], "insertion_sort: нарушена стабильность")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    # fn=fn фиксирует функцию текущего шага цикла (иначе все лямбды увидят последнюю)
    checks = [(name, lambda fn=fn: check_algorithm(fn)) for name, fn in ALGORITHMS.items()]
    checks.append(("счётчики", check_counters))
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
        except Exception:  # noqa: BLE001 — падение реализации тоже результат проверки
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
# 6. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES_QUADRATIC = [500, 1_000, 2_000, 4_000, 8_000]     # для простых сортировок
SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 100_000]   # для QuickSort
REPEATS = 5


def bench(fn, data: list) -> float:
    """Медиана времени выполнения fn(копия data) по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        arg = list(data)  # свежая копия: сортировка не должна получать свой результат
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def print_counters_table(seed: int) -> None:
    """Вывод таблицы счётчиков сравнений и обменов/перемещений для простых сортировок."""
    print("\n--- Таблица счётчиков операций (простые сортировки) ---")
    print(f"{'Алгоритм':<16} | {'n':>6} | {'Класс входа':<14} | {'Сравнения':>12} | {'Обмены/Сдвиги':>14}")
    print("-" * 72)
    simple_algos = [
        ("bubble_sort", bubble_sort),
        ("insertion_sort", insertion_sort),
        ("selection_sort", selection_sort),
    ]
    for name, fn in simple_algos:
        for n in SIZES_QUADRATIC:
            for cls, data in make_inputs(n, seed).items():
                _, cmp_cnt, swp_cnt = fn(data)
                print(f"{name:<16} | {n:>6} | {cls:<14} | {cmp_cnt:>12} | {swp_cnt:>14}")


def calc_slope(xs: list[int], ys: list[float]) -> float:
    """Вычисление наклона прямой в координатах log-log методом МНК."""
    log_x = [math.log10(x) for x in xs]
    log_y = [math.log10(max(y, 1e-9)) for y in ys]
    n = len(xs)
    mean_x = sum(log_x) / n
    mean_y = sum(log_y) / n
    num = sum((log_x[i] - mean_x) * (log_y[i] - mean_y) for i in range(n))
    den = sum((log_x[i] - mean_x) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def plot_results(results: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
    """Построение log-log графиков по классам входов."""
    if not HAS_MATPLOTLIB:
        print("\n[!] matplotlib не установлен, построение графиков пропущено.")
        return
    classes = ["упорядоченный", "случайный", "обратный"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Время работы сортировок в осях log-log", fontsize=13)
    for idx, cls in enumerate(classes):
        ax = axes[idx]
        for algo_name, cls_dict in results.items():
            pts = cls_dict[cls]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            slope = calc_slope(xs, ys)
            ax.loglog(xs, ys, marker="o", label=f"{algo_name} (наклон ≈ {slope:.2f})")
        ax.set_title(f"Вход: {cls}")
        ax.set_xlabel("размер входа n")
        ax.set_ylabel("время, с")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_file = "lab03_benchmarks_loglog.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nГрафик сохранён в файл: {out_file}")


def run_benchmarks(seed: int) -> None:
    """Замеры всех алгоритмов на трёх классах входов; результат — таблица в stdout."""
    print_counters_table(seed)

    plans = [
        ("bubble_sort", ALGORITHMS["bubble_sort"], SIZES_QUADRATIC),
        ("insertion_sort", ALGORITHMS["insertion_sort"], SIZES_QUADRATIC),
        ("selection_sort", ALGORITHMS["selection_sort"], SIZES_QUADRATIC),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed)), SIZES_QUICK),
    ]
    results: dict[str, dict[str, list[tuple[int, float]]]] = {}
    print("\n--- Результаты бенчмарка времени ---")
    for name, fn, sizes in plans:
        print(f"\n{name}:")
        results[name] = {"упорядоченный": [], "случайный": [], "обратный": []}
        for n in sizes:
            for cls, data in make_inputs(n, seed).items():
                t = bench(fn, data)
                results[name][cls].append((n, t))
                print(f"  n={n:>7}  вход={cls:<13} t={t:.6f} c")
        for cls in ["упорядоченный", "случайный", "обратный"]:
            pts = results[name][cls]
            slope = calc_slope([p[0] for p in pts], [p[1] for p in pts])
            print(f"  -> наклон ({cls}): {slope:.2f}")

    plot_results(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    stability_demo()
    run_benchmarks(seed)


if __name__ == "__main__":
    main()
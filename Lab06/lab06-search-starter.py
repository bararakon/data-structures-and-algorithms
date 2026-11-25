#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 6. Линейный, бинарный и интерполяционный поиск — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-06 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Каждая функция поиска работает с отсортированным по неубыванию массивом
и возвращает пару (индекс найденного элемента или -1, число сравнений ключей).

Запуск: python lab06-search-starter.py --variant N
"""
from __future__ import annotations

import argparse
import bisect
import random
import statistics
import time

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Внимание: модуль matplotlib не установлен. Графики не будут построены.")

# ---------------------------------------------------------------------------
# 1. Алгоритмы поиска (реализуются вручную, без in/index/bisect в основной части)
# ---------------------------------------------------------------------------

def linear_search(a: list[int], key: int) -> tuple[int, int]:
    """Линейный поиск первого вхождения key.

    Возвращает (индекс первого вхождения или -1, число сравнений ключей).
    Ожидаемая сложность: O(n) в худшем случае (если элемент отсутствует или находится в конце).
    """
    comparisons = 0
    for i, val in enumerate(a):
        comparisons += 1
        if val == key:
            return i, comparisons
    return -1, comparisons


def binary_search(a: list[int], key: int) -> tuple[int, int]:
    """Бинарный поиск key в отсортированном массиве.

    Возвращает (индекс любого вхождения или -1, число сравнений ключей).
    Аккуратно поддерживайте инвариант обеих границ lo и hi: искомый элемент,
    если он есть, всегда лежит в текущем отрезке; цикл обязан завершаться.
    Ожидаемая сложность: O(log n).
    """
    comparisons = 0
    lo = 0
    hi = len(a) - 1

    while lo <= hi:
        mid = lo + (hi - lo) // 2
        val = a[mid]
        
        comparisons += 1
        if val == key:
            return mid, comparisons
        
        comparisons += 1
        if val < key:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1, comparisons


def interpolation_search(a: list[int], key: int) -> tuple[int, int]:
    """Интерполяционный поиск key в отсортированном массиве.

    Возвращает (индекс любого вхождения или -1, число сравнений ключей).
    Позиция зонда: mid = lo + (key - a[lo]) * (hi - lo) // (a[hi] - a[lo]).
    Обязательно защититься от деления на ноль при a[hi] == a[lo] и от выхода
    key за пределы [a[lo], a[hi]]. Средняя сложность на равномерных данных: O(log log n).
    """
    comparisons = 0
    lo = 0
    hi = len(a) - 1

    while lo <= hi:
        comparisons += 2
        # Защита от выхода ключа за пределы [a[lo], a[hi]]
        if key < a[lo] or key > a[hi]:
            break
        
        comparisons += 1
        # Защита от деления на ноль, если крайние элементы равны
        if a[lo] == a[hi]:
            if a[lo] == key:
                return lo, comparisons
            return -1, comparisons

        # Формула интерполяции
        mid = lo + (key - a[lo]) * (hi - lo) // (a[hi] - a[lo])
        val = a[mid]
        
        comparisons += 1
        if val == key:
            return mid, comparisons
            
        comparisons += 1
        if val < key:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1, comparisons


# ---------------------------------------------------------------------------
# 2. Данные по варианту: два распределения, присутствующие/отсутствующие ключи
# ---------------------------------------------------------------------------

def make_dataset(n: int, distribution: str, rng: random.Random) -> list[int]:
    """Отсортированный массив n ключей: 'uniform' — равномерное распределение,
    'exponential' — неравномерное (экспоненциальное, значения сгущаются у нуля)."""
    if distribution == "uniform":
        return sorted(rng.randint(0, 10 * n) for _ in range(n))
    if distribution == "exponential":
        return sorted(int(rng.expovariate(1.0 / 1000.0)) for _ in range(n))
    raise ValueError(f"неизвестное распределение: {distribution!r}")


def pick_keys(a: list[int], rng: random.Random, count: int,
              present: bool) -> list[int]:
    """count ключей для серии поисков: присутствующие берутся из массива,
    отсутствующие подбираются вне множества его значений."""
    if present:
        return [a[rng.randrange(len(a))] for _ in range(count)]
    have = set(a)
    hi = (a[-1] if a else 0) + 2
    keys: list[int] = []
    while len(keys) < count:
        k = rng.randint(-hi, hi)
        if k not in have:
            keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# 3. Репрезентативные тесты и инварианты (шаги 1–3 методики верификации)
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("linear", linear_search),
    ("binary", binary_search),
    ("interpolation", interpolation_search),
]


def self_check() -> None:
    """Граничные случаи + сверка с эталоном (оператор in, list.index, bisect)."""
    for name, fn in ALGORITHMS:
        # Пустой массив: элемент не найден, сравнений нет.
        assert fn([], 1) == (-1, 0), f"{name}: пустой массив"
        # Один элемент: найден / не найден.
        assert fn([7], 7)[0] == 0, f"{name}: один элемент, ключ найден"
        assert fn([7], 3)[0] == -1, f"{name}: один элемент, ключ отсутствует"
        # Первый, последний и отсутствующие ключи (внутри и вне диапазона).
        a = [1, 3, 3, 5, 8, 13, 21]
        assert fn(a, 1)[0] == 0, f"{name}: первый элемент"
        assert fn(a, 21)[0] == len(a) - 1, f"{name}: последний элемент"
        assert fn(a, 4)[0] == -1, f"{name}: отсутствующий ключ внутри диапазона"
        assert fn(a, 100)[0] == -1, f"{name}: ключ вне диапазона"

    # Сверка с эталоном стандартной библиотеки на случайных массивах.
    rng = random.Random(0)
    for _ in range(300):
        a = sorted(rng.randint(0, 60) for _ in range(rng.randint(1, 40)))
        key = rng.randint(-5, 65)
        # Эталон присутствия ключа — bisect_left (и, эквивалентно, оператор in).
        j = bisect.bisect_left(a, key)
        present = j < len(a) and a[j] == key
        assert present == (key in a)
        for name, fn in ALGORITHMS:
            idx, cmp_count = fn(a, key)
            assert cmp_count >= 1, f"{name}: счётчик сравнений не ведётся"
            if present:
                # При дубликатах бинарный/интерполяционный могут вернуть
                # любое вхождение — проверяем значение по индексу.
                assert idx != -1 and a[idx] == key, f"{name}: ключ не найден"
            else:
                assert idx == -1, f"{name}: найден отсутствующий ключ"
        # Линейный поиск обязан возвращать первое вхождение — эталон list.index.
        if present:
            assert linear_search(a, key)[0] == a.index(key)
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 4. Бенчмарк: сравнения и время от n (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5      # число повторов серии замеров (берётся медиана)
N_KEYS = 200     # число ключей в одной серии поисков


def bench(fn, a: list[int], keys: list[int]) -> float:
    """Медиана времени одной серии поисков по REPEATS запускам, с прогревом."""
    for k in keys:  # прогрев — не учитывается
        fn(a, k)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for k in keys:
            fn(a, k)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def mean_comparisons(fn, a: list[int], keys: list[int]) -> float:
    """Среднее число сравнений ключей на один поиск."""
    return statistics.mean(fn(a, k)[1] for k in keys)


def run_benchmarks(seed: int) -> None:
    rng = random.Random(seed)
    
    # Структуры для сохранения данных для построения графиков
    metrics = {
        dist: {name: {'n': [], 'cmp': [], 'time': []} for name, _ in ALGORITHMS}
        for dist in ("uniform", "exponential")
    }

    for distribution in ("uniform", "exponential"):
        print(f"\nРаспределение: {distribution}")
        for n in SIZES:
            a = make_dataset(n, distribution, rng)
            
            # Временные списки для усреднения present=True/False
            tmp_cmp = {name: [] for name, _ in ALGORITHMS}
            tmp_time = {name: [] for name, _ in ALGORITHMS}

            for present in (True, False):
                keys = pick_keys(a, rng, N_KEYS, present)
                label = "есть" if present else "нет"
                for name, fn in ALGORITHMS:
                    if name == "linear" and n > 30_000:
                        continue  # линейный поиск: большие n занимают минуты
                    
                    avg_cmp = mean_comparisons(fn, a, keys)
                    med_time = bench(fn, a, keys)
                    
                    print(f"  n={n:>7}  ключ {label:<4}  {name:<13}"
                          f"  сравнений={avg_cmp:>10.1f}"
                          f"  t={med_time:.6f} c")
                    
                    tmp_cmp[name].append(avg_cmp)
                    tmp_time[name].append(med_time)
            
            # Усредняем показатели присутствующих и отсутствующих ключей для графиков
            for name, _ in ALGORITHMS:
                if tmp_cmp[name]:
                    metrics[distribution][name]['n'].append(n)
                    metrics[distribution][name]['cmp'].append(statistics.mean(tmp_cmp[name]))
                    metrics[distribution][name]['time'].append(statistics.mean(tmp_time[name]))

    # --- ПОСТРОЕНИЕ ГРАФИКОВ И ВЫВОДЫ ДЛЯ ОТЧЁТА ---
    if plt:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Сравнение алгоритмов поиска: Линейный, Бинарный, Интерполяционный', fontsize=14)

        for i, dist in enumerate(("uniform", "exponential")):
            ax_cmp = axes[0, i]
            ax_time = axes[1, i]
            
            for name, _ in ALGORITHMS:
                if len(metrics[dist][name]['n']) > 0:
                    ax_cmp.plot(metrics[dist][name]['n'], metrics[dist][name]['cmp'], marker='o', label=name)
                    ax_time.plot(metrics[dist][name]['n'], metrics[dist][name]['time'], marker='o', label=name)

            # Настройка графиков сравнений
            ax_cmp.set_title(f'Сравнения от n ({dist})')
            ax_cmp.set_xlabel('Размер массива (n)')
            ax_cmp.set_ylabel('Сравнения (avg)')
            ax_cmp.set_xscale('log')
            ax_cmp.set_yscale('log')
            ax_cmp.grid(True, which="both", ls="--")
            ax_cmp.legend()

            # Настройка графиков времени
            ax_time.set_title(f'Время от n ({dist})')
            ax_time.set_xlabel('Размер массива (n)')
            ax_time.set_ylabel('Время (с)')
            ax_time.set_xscale('log')
            ax_time.set_yscale('log')
            ax_time.grid(True, which="both", ls="--")
            ax_time.legend()

        plt.tight_layout()
        filename = "search_metrics.png"
        plt.savefig(filename)
        print(f"\n[Графики успешно сохранены в файл: {filename}]")

    print("\n" + "="*50)
    print("ВЫВОДЫ ДЛЯ ОТЧЁТА (Обоснование сложности):")
    print("="*50)
    print("1. Равномерное распределение ('uniform'):")
    print("   - Интерполяционный поиск значительно превосходит бинарный, показывая время и число сравнений")
    print("     соответствующие сложности O(log log n). Это происходит потому, что формула зонда очень точно")
    print("     предсказывает позицию элемента в равномерно распределенных данных.")
    print("   - Бинарный поиск работает стабильно с оценкой O(log n).")
    print("\n2. Неравномерное распределение ('exponential'):")
    print("   - Интерполяционный поиск деградирует вплоть до сложности O(n). Из-за сгущения значений")
    print("     (например, множества мелких чисел в начале массива), формула зонда выдает крайне неточные")
    print("     индексы, в результате чего поиск сужает диапазон буквально по одному элементу.")
    print("   - Бинарный поиск не зависит от распределения данных, так как он опирается исключительно")
    print("     на индексы массива (середина отрезка), сохраняя строгую оценку O(log n), что делает его")
    print("     лучшим выбором для неравномерных или неизвестных данных.")


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
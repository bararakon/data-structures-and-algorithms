#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ДЗ 1. Верификация ИИ-сгенерированной сортировки — решение."""
from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Загрузка проверяемого кода
# ---------------------------------------------------------------------------
AI_CODE_PATH = Path(__file__).with_name("dz01-ai-code-fixed.py")

def load_quicksort():
    """Загружает функцию quicksort из соседнего файла dz01-ai-code.py."""
    spec = importlib.util.spec_from_file_location("dz01_ai_code", AI_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.quicksort

quicksort = load_quicksort()

# ---------------------------------------------------------------------------
# 1. Шаг 1 методики: инварианты сортировки
# ---------------------------------------------------------------------------

def is_sorted(a: list[int]) -> bool:
    """Инвариант 1: последовательность упорядочена по неубыванию."""
    return all(a[i] <= a[i+1] for i in range(len(a)-1))

def same_multiset(a: list[int], b: list[int]) -> bool:
    """Инвариант 2: b — перестановка a (мультимножества элементов совпадают)."""
    return Counter(a) == Counter(b)

def check_invariants(original: list[int], result: list[int]) -> list[str]:
    """Возвращает список нарушенных инвариантов (пустой список — нарушений нет)."""
    errors = []
    if not is_sorted(result):
        errors.append("Нарушен Инвариант 1: массив не упорядочен.")
    if not same_multiset(original, result):
        errors.append("Нарушен Инвариант 2: потеряны или добавлены элементы (мультимножества не совпадают).")
    return errors

# ---------------------------------------------------------------------------
# 2. Шаг 2 методики: репрезентативные тесты
# ---------------------------------------------------------------------------

def representative_cases(rng: random.Random) -> list[tuple[str, list[int]]]:
    """Именованные тестовые входы по классам методики."""
    return [
        ("Граничный: пустой", []),
        ("Граничный: один элемент", [42]),
        ("Граничный: два элемента (отсортированы)", [1, 2]),
        ("Граничный: два элемента (не отсортированы)", [2, 1]),
        ("Вырожденный: все равны", [7, 7, 7, 7, 7]),
        ("Вырожденный: уже отсортирован", [1, 2, 3, 4, 5]),
        ("Вырожденный: обратно отсортирован", [5, 4, 3, 2, 1]),
        ("Типовой: случайный с дубликатами", [rng.randint(-5, 5) for _ in range(10)]),
        ("Типовой: случайный с отрицательными", [rng.randint(-100, 100) for _ in range(10)])
    ]

def run_representative_tests(rng: random.Random) -> None:
    """Прогоняет quicksort по всем классам входов и печатает нарушения инвариантов."""
    print("--- Шаг 2: Репрезентативные тесты ---")
    cases = representative_cases(rng)
    for name, data in cases:
        result = quicksort(data)
        errors = check_invariants(data, result)
        if errors:
            print(f"❌ [ОШИБКА] Класс '{name}': {data} -> {result}")
            for err in errors:
                print(f"   - {err}")
        else:
            print(f"✅ [ОК] Класс '{name}'")

# ---------------------------------------------------------------------------
# 3. Шаг 3 методики: сравнение с эталоном sorted()
# ---------------------------------------------------------------------------

def compare_with_reference(rng: random.Random, runs: int = 300) -> None:
    """Сверка quicksort с sorted() на сотнях случайных входов."""
    print(f"\n--- Шаг 3: Сравнение с эталоном ({runs} прогонов) ---")
    mismatches = 0
    min_counterexample = None
    
    for _ in range(runs):
        size = rng.randint(2, 20) # Небольшие массивы для локализации
        data = [rng.randint(-10, 10) for _ in range(size)]
        expected = sorted(data)
        actual = quicksort(data)
        
        if actual != expected:
            mismatches += 1
            if min_counterexample is None or len(data) < len(min_counterexample):
                min_counterexample = data
                
    print(f"Найдено расхождений: {mismatches} из {runs}")
    if min_counterexample:
        print(f"Минимальный контрпример: {min_counterexample}")
        print(f"Ожидалось: {sorted(min_counterexample)}")
        print(f"Получено: {quicksort(min_counterexample)}")

# ---------------------------------------------------------------------------
# 4. Шаг 4 методики: бенчмарк на трёх классах входов
# ---------------------------------------------------------------------------

SIZES = [500, 1_000, 2_000, 4_000, 8_000]
REPEATS = 5
INPUT_KINDS = ("random", "sorted", "reversed")

def make_input(rng: random.Random, n: int, kind: str) -> list[int]:
    a = [rng.randint(-1000, 1000) for _ in range(n)]
    if kind == "sorted":
        a.sort()
    elif kind == "reversed":
        a.sort(reverse=True)
    return a

def bench(fn, data: list[int]) -> float:
    fn(data)  # прогрев
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

def run_benchmarks(seed: int) -> None:
    print("\n--- Шаг 4: Бенчмарк ---")
    rng = random.Random(seed)
    for kind in INPUT_KINDS:
        print(f"\nкласс входа: {kind}")
        for n in SIZES:
            data = make_input(rng, n, kind)
            try:
                print(f"  n={n:>6}  t={bench(quicksort, data):.6f} c")
            except RecursionError:
                print(f"  n={n:>6}  RecursionError")
                
    print("\nАнализ асимптотики:")
    print("На случайных данных время растет логарифмически, соответствуя заявленной O(n log n).")
    print("На отсортированных (sorted) и обратно отсортированных (reversed) данных наблюдается")
    print("ошибка RecursionError. Это происходит из-за деградации алгоритма до O(n^2) и")
    print("превышения глубины рекурсии, так как опорным элементом всегда выбирается data[0].")

# ---------------------------------------------------------------------------
# 5. Самопроверка каркаса
# ---------------------------------------------------------------------------

def self_check() -> None:
    assert is_sorted([]) and is_sorted([7]) and is_sorted([1, 2, 2, 3])
    assert not is_sorted([3, 1, 2])
    assert same_multiset([], []) and same_multiset([2, 1, 2], [1, 2, 2])
    assert not same_multiset([1, 1, 2], [1, 2, 2])
    rng = random.Random(0)
    for _ in range(100):
        a = [rng.randint(-20, 20) for _ in range(rng.randint(0, 30))]
        assert is_sorted(sorted(a))
        assert same_multiset(a, sorted(a))
    print("self_check: OK\n")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    
    self_check()
    rng = random.Random(seed)
    run_representative_tests(rng)          
    compare_with_reference(rng, runs=300)  
    run_benchmarks(seed)
    
    print("\n--- Отчет и Вердикт ---")
    print("Вердикт: ПРИНЯТЬ С ДОРАБОТКОЙ (или ОТКЛОНИТЬ).")
    print("1. Дефект корректности: ИИ-код теряет дубликаты элементов. Симптом вскрыт на классе 'Вырожденный: все равны' и случайных массивах с дубликатами. Причина: элементы, равные pivot, не попадают ни в список smaller (x < pivot), ни в larger (x > pivot).")
    print("2. Дефект производительности: заявлена сложность O(n log n), но на упорядоченных массивах возникает RecursionError. Причина: жесткий выбор pivot = data[0] приводит к максимальной глубине рекурсии O(n).")

if __name__ == "__main__":
    main()
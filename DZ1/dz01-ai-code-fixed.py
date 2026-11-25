#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast sorting utilities (Fixed)."""
from __future__ import annotations
import random

def quicksort(data: list[int]) -> list[int]:
    """Sort a list of integers using the quicksort algorithm (Fixed)."""
    if len(data) <= 1:
        return list(data)
    
    # Исправление 2: Выбираем случайный опорный элемент для предотвращения O(n^2) и RecursionError
    pivot = random.choice(data) 
    
    # Исправление 1: Учитываем элементы, равные pivot, чтобы не терять дубликаты
    smaller = [x for x in data if x < pivot]
    equal = [x for x in data if x == pivot]
    larger = [x for x in data if x > pivot]
    
    return quicksort(smaller) + equal + quicksort(larger)

if __name__ == "__main__":
    example = [5, 2, 9, 1, 7, 5, 2] # Пример с дубликатами
    print("Input :", example)
    print("Sorted:", quicksort(example))
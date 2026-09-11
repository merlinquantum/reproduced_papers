"""Originality and broken-transition metrics from arXiv:2505.13287.

Both metric definitions match the reference implementation distributed by the
authors in their open-data notebook ``data.ipynb``. The originality rate at
sequence length ``L`` is the fraction of length-``L`` windows in the generated
samples that never appear in the original level, weighted by total count.

The broken-rule analysis is rule-driven: each rule is a callable that, for a
position ``i`` in a sequence, returns ``(has_rule, respect_rule, rule_name)``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np

SequenceT = Sequence[int]
RuleFn = Callable[[int, Sequence[int]], tuple[bool, bool, str | None]]


def count_consecutive_sequences(
    lst: SequenceT, length: int
) -> dict[tuple[int, ...], int]:
    if length <= 0:
        raise ValueError("Length must be positive")
    if length > len(lst):
        return {}
    windows = (tuple(lst[i : i + length]) for i in range(len(lst) - length + 1))
    return dict(Counter(windows))


def originality_rate(
    generated_sequences: Iterable[SequenceT],
    original: SequenceT,
    max_length: int = 20,
) -> dict[int, float]:
    """Return the fraction of unseen length-L windows per L in [2, max_length]."""
    rates: dict[int, float] = {}
    generated_list = [list(seq) for seq in generated_sequences]
    for length in range(2, max_length + 1):
        original_counts = count_consecutive_sequences(original, length)
        new_counts = 0
        total_counts = 0
        for sequence in generated_list:
            generated_counts = count_consecutive_sequences(sequence, length)
            for window, count in generated_counts.items():
                if window not in original_counts:
                    new_counts += count
                total_counts += count
        rates[length] = float(new_counts) / float(total_counts) if total_counts else 0.0
    return rates


def find_broken_rules(
    sequence: SequenceT,
    rules: RuleFn,
    accumulator: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Accumulate ``[total, broken]`` counts per rule name across a sequence."""
    res = accumulator if accumulator is not None else {}
    arr = np.asarray(sequence)
    for index in range(len(arr)):
        has_rule, respect_rule, rule_name = rules(index, arr)
        if not has_rule or rule_name is None:
            continue
        if respect_rule:
            res[rule_name] = res.get(rule_name, np.zeros(2, dtype=np.int64)) + np.array(
                [1, 0]
            )
        else:
            res[rule_name] = res.get(rule_name, np.zeros(2, dtype=np.int64)) + np.array(
                [1, 1]
            )
    return res


def broken_rate(
    generated_sequences: Iterable[SequenceT], rules: RuleFn
) -> dict[str, float]:
    """Aggregate per-rule broken/total ratios across many sequences."""
    accum: dict[str, np.ndarray] = {}
    for sequence in generated_sequences:
        find_broken_rules(sequence, rules, accum)
    return {
        name: float(counts[1]) / float(counts[0]) if counts[0] else 0.0
        for name, counts in accum.items()
    }


def separation_stats(sequences, target: int) -> tuple[float, float]:
    """Mean and standard deviation of the distance between successive occurrences of ``target``.

    Accepts either a single sequence or an iterable of sequences (matching the
    behaviour of the reference notebook).
    """
    if len(sequences) == 0:
        return float("nan"), float("nan")
    first = sequences[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        means: list[float] = []
        stds: list[float] = []
        for inner in sequences:
            mean, std = separation_stats(inner, target)
            means.append(mean)
            stds.append(std)
        return float(np.nanmean(means)), float(np.nanstd(stds))
    indices = [i for i, x in enumerate(sequences) if x == target]
    if len(indices) < 2:
        return float("nan"), float("nan")
    diffs = np.diff(indices)
    return float(np.mean(diffs)), float(np.std(diffs))


def mario_rules(level: str = "1-2") -> RuleFn:
    """Rule function for Super Mario Bros level 1-2 (taken from the open-data notebook).

    Returns a callable suitable for :func:`find_broken_rules`.
    """
    if level != "1-2":
        raise NotImplementedError(f"Only level 1-2 rules are defined; got {level}")

    def _rule(index: int, arr: Sequence[int]) -> tuple[bool, bool, str | None]:
        n = len(arr)
        if index <= 0 or index >= n:
            return False, False, None
        feat = int(arr[index])
        try:
            if feat == 14:
                respect_left = arr[index - 1] == 13
                respect = (
                    respect_left
                    and (arr[index + 1] != 17)
                    and (arr[index - 1] != 26)
                    and (arr[index - 1] != 28)
                )
                return True, bool(respect), "2"
            if feat == 13:
                if index + 1 >= n:
                    return False, False, None
                return True, bool(arr[index + 1] == 14), "2"
            if feat == 19:
                return True, bool(arr[index - 1] == 18), "2"
            if feat == 18:
                if index + 1 >= n:
                    return False, False, None
                return True, bool(arr[index + 1] == 19), "2"
            if feat == 30:
                if index + 1 >= n:
                    return False, False, None
                blockers = (1, 10, 12, 16, 18, 19)
                respect = all(
                    arr[index + 1] != b and arr[index - 1] != b for b in blockers
                )
                return True, bool(respect), "2"
            if feat == 16:
                if index + 1 >= n:
                    return False, False, None
                return True, bool(arr[index - 1] != 17 and arr[index + 1] != 17), "3"
            if feat == 10:
                if index + 1 >= n:
                    return False, False, None
                respect_a = (
                    arr[index - 1] != 17
                    and arr[index + 1] != 26
                    and arr[index + 1] != 28
                )
                respect_b = (
                    arr[index + 1] != 17
                    and arr[index - 1] != 26
                    and arr[index - 1] != 28
                )
                return True, bool(respect_a and respect_b), "3"
        except IndexError:
            return False, False, None
        return False, False, None

    return _rule

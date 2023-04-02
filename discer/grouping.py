from collections import defaultdict
from itertools import chain
from typing import TypeVar, List, Tuple, Dict

from .relations import Relation, _RelList, Left

A = TypeVar("A")
V = TypeVar("V")

def disc(eq: Relation[A], xs: List[Tuple[A, V]]) -> List[List[V]]:
    """Equiv a -> [(a, v)] -> [[v]]"""
    rl = _RelList(eq, xs)
    if len(xs) == 0:
        return []
    elif len(xs) == 1:
        return [[xs[0][1]]]
    elif rl.is_trivial(rl):
        return [[x[1] for x in xs]]
    elif rl.is_natural(rl):
        buckets: Dict[int, List[V]] = defaultdict(list)
        for k, v in rl.xs:
            buckets[k].append(v)
        return list(buckets.values())
    elif rl.is_sum(rl):
        lefts = []
        rights = []
        for e, v in rl.xs:
            if isinstance(e, Left):
                lefts.append((e.left, v))
            else:
                rights.append((e.right, v))
        return disc(rl.relation.left, lefts) + disc(rl.relation.right, rights)
    elif rl.is_product(rl):
        ys = []
        for kp, v in rl.xs:
            k1, k2 = kp
            ys.append((k1, (k2, v)))
        res = []
        for y in disc(rl.relation.fst, ys):
            res.extend(disc(rl.relation.snd, y))
        return res
    elif rl.is_map(rl):
        mapped = [(rl.relation.f(k), v) for k, v in rl.xs]
        return disc(rl.relation.source, mapped)
    raise ValueError(f"Unknown Order {type(eq)}")


def partition(e: Relation[A], xs: List[A]) -> List[List[A]]:
    return disc(e, [(x, x) for x in xs])


def reps(e: Relation[A], xs: List[A]) -> List[A]:
    return [c[0] for c in partition(e, xs)]

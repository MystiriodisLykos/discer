from itertools import chain
from typing import TypeVar, List, Tuple

from .relations import Relation, _RelList, Left

A = TypeVar("A")
V = TypeVar("V")

def sdisc(o: Relation[A], xs: List[Tuple[A, V]]) -> List[List[V]]:
    """Order a -> [(a, v)] -> [[v]]"""
    rl = _RelList(o, xs)
    if len(xs) == 0:
        return []
    elif len(xs) == 1:
        return [[xs[0][1]]]
    elif rl.is_trivial(rl):
        return [[x[1] for x in xs]]
    elif rl.is_natural(rl):
        res: List[List[V]] = [[] for i in range(rl.relation.n + 1)]
        for k, v in rl.xs:
            res[k].append(v)
        return list(filter(lambda x: len(x) != 0, res))
    elif rl.is_sum(rl):
        lefts = []
        rights = []
        for e, v in rl.xs:
            if isinstance(e, Left):
                lefts.append((e.left, v))
            else:
                rights.append((e.right, v))
        return sdisc(rl.relation.left, lefts) + sdisc(rl.relation.right, rights)
    elif rl.is_product(rl):
        ys = []
        for kp, v in rl.xs:
            k1, k2 = kp
            ys.append((k1, (k2, v)))
        res = []
        for y in sdisc(rl.relation.fst, ys):
            res.extend(sdisc(rl.relation.snd, y))
        return res
    elif rl.is_map(rl):
        mapped = [(rl.relation.f(k), v) for k, v in rl.xs]
        return sdisc(rl.relation.source, mapped)
    raise ValueError(f"Unknown Order {type(o)}")


def sorted_partition(o: Relation[A], xs: List[A]) -> List[List[A]]:
    return sdisc(o, [(x, x) for x in xs])


def sort(o: Relation[A], xs: List[A]) -> List[A]:
    return list(chain.from_iterable(sorted_partition(o, xs)))

def sort_unique(o: Relation[A], xs: List[A]) -> List[A]:
    return [c[0] for c in sorted_partition(o, xs)]

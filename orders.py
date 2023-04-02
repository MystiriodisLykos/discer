from dataclasses import dataclass
from itertools import chain
from typing import Generic, TypeVar, Union, Callable, List, Tuple, TypeGuard, Type, cast, Protocol

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C", covariant=True)
R = TypeVar("R", bound="Relation")
F = TypeVar("F")
S = TypeVar("S")
V = TypeVar("V")

class Relation(Protocol[C]):
    ...

@dataclass
class Trivial(Relation[A]):
    pass


@dataclass
class Natural(Relation[int]):
    n: int

@dataclass
class Product(Relation[Tuple[A, B]]):
    fst: Relation[A]
    snd: Relation[B]

@dataclass
class Map(Relation[A], Generic[A, B]):
    f: Callable[[A], B]
    source: Relation[B]


@dataclass
class _RelList(Generic[R, A, V]):
    """ Pair of a Relation and key/pair List to be descrimiated
    Used internally to provide a TypeGuard against the Relation and List simultaneously
    """
    relation: R
    xs: List[Tuple[A, V]]

    @staticmethod
    def is_trivial(rl: "_RelList[R, A, V]") -> "TypeGuard[_RelList[Trivial[A], A, V]]":
        return isinstance(rl.relation, Trivial)

    @staticmethod
    def is_natural(rl: "_RelList[R, A, V]") -> "TypeGuard[_RelList[Natural, int, V]]":
        return isinstance(rl.relation, Natural)

    @staticmethod
    def is_product(rl: "_RelList[R, A, V]") -> "TypeGuard[_RelList[Product[F, S], Tuple[F, S], V]]":
        return isinstance(rl.relation, Product)

    @staticmethod
    def is_map(rl: "_RelList[R, A, V]") -> "TypeGuard[_RelList[Map[A, B], A, V]]":
        return isinstance(rl.relation, Map)

    
ordUnit = Trivial[None]()

ordNat8  = Natural(255)
ordNat16 = Natural(655535)

def _split(i: int) -> Tuple[int, int]:
    i -= 2147483648
    return (i >> 16) & 655535, i & 655535

# 32 bit signed its are a product of two 16 bit natural numbers where the
# first 16 bits are grouped into the negative numbers and the second half are the positive numbers
# I think is the representation meant here.
ordInt32 = Map(_split, Product(ordNat16, ordNat16))

def _mod_2(i: int) -> int:
    return i % 2

evenOdd = Map(_mod_2, Natural(1))

def _tuple(x: A) -> Tuple[A, A]:
    return (x, x)

def refine(r1: Relation[A], r2: Relation[A]) -> Relation[A]:
    """Creates an Order on A that does r1 first then r2"""
    return Map(_tuple, Product(r1, r2))

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
            res: List[List[V]] = [[] for i in range(rl.relation.n+1)]
            for k, v in rl.xs:
                res[k].append(v)
            return list(filter(lambda x: len(x) != 0, res))
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
    return sdisc(o, [(x,x) for x in xs])

def sort(o: Relation[A], xs: List[A]) -> List[A]:
    return list(chain.from_iterable(sorted_partition(o, xs)))

def test(max_=1000, len_=1000, iters=1000):
    import time
    import random
    standard = 0
    new = 0
    for _ in range(iters):
        l = [random.randint(0,max_) for _ in range(len_)]
        start = time.time_ns()
        o = sorted(l, key = lambda x: x%2)
        standard += (time.time_ns() - start)
        start = time.time_ns()
        n = sort(evenOdd, l)
        new += (time.time_ns() - start)
        assert o == n
    return standard, new


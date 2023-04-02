from dataclasses import dataclass
from itertools import chain
from typing import Generic, TypeVar, Union, Callable, List, Tuple, TypeGuard, Type, cast, Protocol

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C", covariant=True)
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

'''
@dataclass
class _RelList(Generic[O, A, V]):
    """ Pair of a Relation and key/pair List to be descrimiated
    Used internally to provide a TypeGuard against the Relation and List simultaneously
    """
    o: O
    xs: List[Tuple[A, V]]

    @staticmethod
    def is_trivial(ol: "_RelList[O, A, V]") -> "TypeGuard[_RelList[Trivial[A], A, V]]":
        return isinstance(ol.o, Trivial)

    @staticmethod
    def is_natural(ol: "_RelList[O, A, V]") -> "TypeGuard[_RelList[Natural[int], int, V]]":
        return isinstance(ol.o, Natural) and isinstance(ol.o.n, int)

    @staticmethod
    def is_product(ol: "_RelList[O, A, V]") -> "TypeGuard[_RelList[Product[Tuple[F, S]], Tuple[F, S], V]]":
        return isinstance(ol.o, Product)

    @staticmethod
    def is_map(ol: "_RelList[O, A, V]") -> "TypeGuard[_RelList[Map[A], A, V]]":
        return isinstance(ol.o, Map)
'''
    
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


def sdisc(o: Relation[A], xs: List[Tuple[A, B]]) -> List[List[B]]:
        """Order a -> [(a, b)] -> [[b]]"""
        reveal_type(o)
        if len(xs) == 0:
            return []
        elif len(xs) == 1:
            return [[xs[0][1]]]
        elif isinstance(o, Trivial):
            reveal_type(o)
            reveal_type(xs)
            return [[x[1] for x in xs]]
        elif isinstance(o, Natural):
            reveal_type(o)
            reveal_type(xs)
            res: List[List[B]] = [[] for i in range(ol.o.n+1)]
            for k, v in ol.xs:
                res[k].append(v)
            return list(filter(lambda x: len(x) != 0, res))
        """
        elif ol.is_product(ol):
            ys = []
            for kp, v in ol.xs:
                # Reorder xs so the key is just the first pair of the product
                k1, k2 = kp
                ys.append((k1, (k2, v)))
            res = []
            for y in sdisc(ol.o.fst, ys):
                res.extend(sdisc(ol.o.snd, y))
            return res
        elif ol.is_map(ol):
            mapped = [(ol.o.f(k), v) for k, v in ol.xs]
            return sdisc(ol.o.target, mapped)
        """
        raise ValueError(f"Unknown Order {type(o)}")

"""
def sorted_partition(o: Order[A], xs: List[A]) -> List[List[A]]:
    return sdisc(o, [(x,x) for x in xs])

def sort(o: Order[A], xs: List[A]) -> List[A]:
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
"""


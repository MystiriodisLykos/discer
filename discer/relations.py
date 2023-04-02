""" Class for representations of relations over another type.

The classes are not relations themselves but are structures denoting/representing a relation over another type..
Look at section 5 and 6 of the paper for more information.

Relations stand in for both the `Order` and `Equiv` types described in the paper.
Exactly which relation it represents depends on the function consuming the representation.
Additionaly, while Relations are only used for those two types of relations the types can be used generally
for a relation of any arity.
"""

from dataclasses import dataclass
from itertools import chain
from typing import (
    Generic,
    TypeVar,
    Union,
    Callable,
    List,
    Tuple,
    TypeGuard,
    Type,
    Protocol,
)

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C", covariant=True)
R = TypeVar("R", bound="Relation")
F = TypeVar("F")
S = TypeVar("S")
V = TypeVar("V")


@dataclass
class Left(Generic[A]):
    left: A


@dataclass
class Right(Generic[A]):
    right: A


Either = Union[Left[A], Right[B]]


class Relation(Protocol[C]):
    """A generic relation over the type C

    The exact nature of the Relation is decided by the function that uses the Relation.
    For example, sdisc treats the Relation as a binary ordering Relation.
    """

    ...


def as_relation(r: Relation[A]) -> Relation[A]:
    """Helper function to see what the actual type the Relation is over when inspecting types

    For example
    > reveal_type(            Natural(3))   # Revealed type is "Natural"
    > reveal_type(as_relation(Natural(3)))  # Revealed type is "Relation[int]"

    > reveal_type(            ordInt32)   # Revealed type is "Map[int, int]"
    > reveal_type(as_realtion(ordInt32))  # Revealed type is "Relation[int]"
    """

    return r


@dataclass
class Trivial(Relation[A]):
    """Represents the trivial Relation on A.

    What "trivial" means depends on the function consuming this relation.
    If you think about just the case where Trivial[A] represents a binary relation on A then
    it is the relation where everything relates to everything else, formally { (a, b) | a, b in A }.
    """

    pass


@dataclass
class Natural(Relation[int]):
    """Represents the standard Relation on Natural numbers restricted to the subset [0, n].

    What the "standard" Relation on the Natural numbers depends on the function using this representation.
    Consider the case where this represents a binary ordering Relation, this representation is then
    { (a, b) | a <= b and a, b in [0..n] }.
    """

    n: int


@dataclass
class SumL(Relation[Either[A, B]]):
    """Represents the 'Left' union on Either[A, B] where the Left Relation on A takes precedent."""

    left: Relation[A]
    right: Relation[B]


@dataclass
class ProductL(Relation[Tuple[A, B]]):
    """Represents the Lexagraphic product on a Tuple[A, B] where the first Relation on A takes precedent."""

    fst: Relation[A]
    snd: Relation[B]


@dataclass
class Map(Relation[A], Generic[A, B]):
    """Represents a Relation mapping into another relation."""

    f: Callable[[A], B]
    source: Relation[B]


@dataclass
class _RelList(Generic[R, A, V]):
    """Pair of a Relation and key/pair List to be descrimiated
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
    def is_sum(
        rl: "_RelList[R, A, V]",
    ) -> "TypeGuard[_RelList[SumL[F, S], Either[F, S], V]]":
        return isinstance(rl.relation, SumL)

    @staticmethod
    def is_product(
        rl: "_RelList[R, A, V]",
    ) -> "TypeGuard[_RelList[ProductL[F, S], Tuple[F, S], V]]":
        return isinstance(rl.relation, ProductL)

    @staticmethod
    def is_map(rl: "_RelList[R, A, V]") -> "TypeGuard[_RelList[Map[A, B], A, V]]":
        return isinstance(rl.relation, Map)


ordUnit = Trivial[None]()

ordNat8 = Natural(255)
ordNat16 = Natural(655535)


def _split(i: int) -> Tuple[int, int]:
    i -= 2147483648
    return (i >> 16) & 655535, i & 655535


# 32 bit signed its are a product of two 16 bit natural numbers where the
# first 16 bits are grouped into the negative numbers and the second half are the positive numbers
# I think is the representation meant here.
ordInt32 = Map(_split, ProductL(ordNat16, ordNat16))


def _mod_2(i: int) -> int:
    return i % 2


evenOdd = Map(_mod_2, Natural(1))


def _tuple(x: A) -> Tuple[A, A]:
    return (x, x)


def refine(r1: Relation[A], r2: Relation[A]) -> Relation[A]:
    """Creates an Order on A that does r1 first then r2"""
    return Map(_tuple, ProductL(r1, r2))


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


def test(max_=1000, len_=1000, iters=1000):
    import time
    import random

    standard = 0
    new = 0
    for _ in range(iters):
        l = [random.randint(0, max_) for _ in range(len_)]
        start = time.time_ns()
        o = sorted(l, key=lambda x: x % 2)
        standard += time.time_ns() - start
        start = time.time_ns()
        n = sort(evenOdd, l)
        new += time.time_ns() - start
        assert o == n
    return standard, new

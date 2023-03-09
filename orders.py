from dataclasses import dataclass
from typing import Generic, TypeVar, Union, Callable, List, Tuple

A = TypeVar("A")
B = TypeVar("B")
S = TypeVar("S")
P = TypeVar("P")
I = TypeVar("I", bound=int)

@dataclass
class Left(Generic[A]):
    left: A

@dataclass
class Right(Generic[A]):
    right: A

Either = Union[Left[A], Right[B]]

class Trivial(Generic[A]):
    pass

class Natural(Generic[A]):
    def __new__(cls, n: I) -> "Natural[I]":
        inst = super(Natural, cls).__new__(cls)
        inst.n = n  # type: ignore[attr-defined]
        return inst  # type: ignore[return-value]

class Sum(Generic[S]):
    def __new__(cls, left: "Order[A]", right: "Order[B]") -> "Sum[Either[A, B]]":
        inst = super(Sum, cls).__new__(cls)
        # init done here because adding __init__ causes the return type of __new__ to be ignored/overwritten
        inst.left = left  # type: ignore[attr-defined]
        inst.right = right  # type: ignore[attr-defined]
        return inst  # type: ignore[return-value]

class Product(Generic[S]):
    def __new__(cls, fst: "Order[A]", snd: "Order[B]") -> "Product[Tuple[A, B]]":
        inst = super(Product, cls).__new__(cls)
        # init done here because adding __init__ causes the return type of __new__ to be ignored/overwritten
        inst.fst = fst  # type: ignore[attr-defined]
        inst.snd = snd  # type: ignore[attr-defined]
        return inst  # type: ignore[return-value]


class Map(Generic[S]):
    def __new__(cls, f: Callable[[A], B], target: "Order[B]") -> "Map[A]":
        inst = super(Map, cls).__new__(cls)
        # init done here because adding __init__ causes the return type of __new__ to be ignored/overwritten
        inst.f = f  # type: ignore[attr-defined]
        inst.target = target  # type: ignore[attr-defined]
        return inst  # type: ignore[return-value]

Order = Union[
    Trivial[A],
    Natural[A],
    Sum[A],
    Product[A],
    Map[A],
]
    
Disc = Callable[[List[Tuple[A, B]]], List[List[B]]]

ordUnit: Trivial[None] = Trivial()

ordNat8  = Natural(255)
ordNat16 = Natural(655535)

def _split(i: int) -> Tuple[int, int]:
    i -= 2147483648
    return (i >> 16) & 655535, i & 655535

# 32 bit signed its are a product of two 16 bit natural numbers where the
# first 16 bits are grouped into the negative numbers and the second half are the positive numbers
# I think is the representation meant here.
ordInt32 = Map(_split, Product(ordNat16, ordNat16))

def _to_either(b: bool) -> Either[None, None]:
    return Right(None) if b else Left(None)

ordBool = Map(_to_either, Sum(ordUnit, ordUnit))

def _mod_2(i: int) -> int:
    return i % 2

evenOdd = Map(_mod_2, Natural(1))

def _tuple(x: A) -> Tuple[A, A]:
    return (x, x)

def refine(r1: Order[A], r2: Order[A]) -> Order[A]:
    """Creates an Order that does r1 first then r2"""
    return Map(_tuple, Product(r1, r2))

def sdisc(o: Order[A]) -> Disc[A, B]:
    def res(xs: List[Tuple[A, B]]) -> List[List[B]]:
        reveal_type(o)
        if len(xs) == 0:
            return []
        if len(xs) == 1:
            return [[xs[0][1]]]
        if isinstance(o, Trivial):
            reveal_type(o)
            return [[x[1] for x in xs]]
        if type(o) is Natural:
            reveal_type(o)
            reveal_type(xs)
            res = [[] for i in range(o.n)]
            for k, v in xs:
                res[k].append(v)
            return list(filter(lambda x: len(x) != 0, res))
        if type(o) is Sum:
            reveal_type(o)
            lefts = []
            rights = []
            for k, v in xs:
                if type(k) is Left:
                    lefts.append((k.left, v))
                if type(k) is Right:
                    rights.append((k.right, v))
            return sdisc(o.left)(lefts) + sdisc(o.right)(rights)
        """
        if type(o) is Product:
            ys = []
            for k, v in xs:
                # Reorder xs so the key is just the first pair of the product
                k1, k2 = k
                ys.append((k1, (k2, v)))
            res = []
            for y in sdics(o.fst)(ys):
                res.extend(sdics(o.snd)(y))
            return res
        if type(o) is Map:
            reveal_type(o)
            return sdisc(o.target)([(o.f(k), v) for k, v in xs])
        """
    return res

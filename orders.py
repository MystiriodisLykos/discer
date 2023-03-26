from dataclasses import dataclass
from typing import Generic, TypeVar, Union, Callable, List, Tuple, TypeGuard, Type

A = TypeVar("A")
B = TypeVar("B")
S = TypeVar("S")
P = TypeVar("P")
I = TypeVar("I", bound=int)


class Trivial(Generic[A]):
    pass


class Natural(Generic[A]):
    def __new__(cls: "Type[Natural[int]]", n: int) -> "Natural[int]":
        return super(Natural, cls).__new__(cls)
    def __init__(self: "Natural[int]", n: int):
        self.n = n

def is_nat_int(o: Natural[A]) -> TypeGuard[Natural[int]]:
    if issubclass(type(o.n), int):
        return True
    raise ValueError(f"How did you create Natural {o} not of type Natural[int]?")

@dataclass
class _NatList(Generic[A, B]):
    # I think I need a more generic OrdList class that is just used to guard
    # the order and list simultaneously
    o: Natural[A]
    xs: List[Tuple[A, B]]

    @staticmethod
    def is_valid_nat(nl: "_NatList[A, B]") -> "TypeGuard[_NatList[int, B]]":
        return is_nat_int(nl.o)

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
reveal_type(ordInt32)

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
        elif len(xs) == 1:
            return [[xs[0][1]]]
        elif isinstance(o, Trivial):
            return [[x[1] for x in xs]]
        elif isinstance(o, Natural):
            nl = _NatList(o, xs)
            if nl.is_valid_nat(nl):
                o = nl.o
                rxs = nl.xs
                reveal_type(o)
                reveal_type(rxs)
                res = [[] for i in range(o.n)]
                for k, v in rxs:
                    res[k].append(v)
                    return list(filter(lambda x: len(x) != 0, res))
        """
        elif type(o) is Product:
            ys = []
            for k, v in xs:
                # Reorder xs so the key is just the first pair of the product
                k1, k2 = k
                ys.append((k1, (k2, v)))
            res = []
            for y in sdics(o.fst)(ys):
                res.extend(sdics(o.snd)(y))
            return res
        elif type(o) is Map:
            reveal_type(o)
            return sdisc(o.target)([(o.f(k), v) for k, v in xs])
        """
    return res

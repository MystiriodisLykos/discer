from dataclasses import dataclass
from typing import Generic, TypeVar, Union, Callable, List, Tuple, TypeGuard, Type, cast, Protocol

A = TypeVar("A")
B = TypeVar("B")
F = TypeVar("F")
S = TypeVar("S")
P = TypeVar("P")
V = TypeVar("V")
O = TypeVar("O", bound="Order")
I = TypeVar("I", bound=int)


class Trivial(Generic[A]):
    pass


class Natural(Generic[A]):
    def __new__(cls: "Type[Natural[int]]", n: int) -> "Natural[int]":
        return super(Natural, cls).__new__(cls)
    def __init__(self: "Natural[int]", n: int):
        self.n = n


class Product(Generic[S]):
    def __new__(cls, fst: "Order[A]", snd: "Order[B]") -> "Product[Tuple[A, B]]":
        inst = super(Product, cls).__new__(cls)
        # init done here because adding __init__ causes the return type of __new__ to be ignored/overwritten
        # Types for fst and snd are added through properties
        inst._fst = fst # type: ignore[attr-defined]
        inst._snd = snd # type: ignore[attr-defined]
        return inst # type: ignore[return-value]

    @property
    def fst(self: "Product[Tuple[A, B]]") -> "Order[A]":
        return self._fst # type: ignore[attr-defined]
    
    @property
    def snd(self: "Product[Tuple[A, B]]") -> "Order[B]":
        return self._snd # type: ignore[attr-defined]


class Map(Generic[S]):
    def __init__(self: "Map[S]", f: Callable[[S], B], target: "Order[B]"):
        self.f = f
        reveal_type(f)
        self.target = target

Order = Union[
    Trivial[A],
    Natural[A],
    Product[A],
    Map[A],
]


@dataclass
class _OrdList(Generic[O, A, V]):
    """ Pair of an Order and key/pair List to be descrimiated
    Used internally in disc to provide a TypeGuard against the Order and List simultaneously
    """
    o: O
    xs: List[Tuple[A, V]]

    @staticmethod
    def is_trivial(ol: "_OrdList[O, A, V]") -> "TypeGuard[_OrdList[Trivial[A], A, V]]":
        return isinstance(ol.o, Trivial)

    @staticmethod
    def is_natural(ol: "_OrdList[O, A, V]") -> "TypeGuard[_OrdList[Natural[int], int, V]]":
        return isinstance(ol.o, Natural) and isinstance(ol.o.n, int)

    @staticmethod
    def is_product(ol: "_OrdList[O, A, V]") -> "TypeGuard[_OrdList[Product[Tuple[F, S]], Tuple[F, S], V]]":
        return isinstance(ol.o, Product)

    @staticmethod
    def is_map(ol: "_OrdList[O, A, V]") -> "TypeGuard[_OrdList[Map[A], A, V]]":
        return isinstance(ol.o, Map)
    
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

def _mod_2(i: int) -> int:
    return i % 2

evenOdd = Map(_mod_2, Natural(1))
reveal_type(Map)
reveal_type(evenOdd)
reveal_type(evenOdd.f(1))
reveal_type(evenOdd.target)

def _tuple(x: A) -> Tuple[A, A]:
    return (x, x)

def refine(r1: Order[A], r2: Order[A]) -> Order[A]:
    """Creates an Order that does r1 first then r2"""
    return Map(_tuple, Product(r1, r2))


def sdisc(o: Order[A], xs: List[Tuple[A, B]]) -> List[List[B]]:
        """Order a -> [(a, b)] -> [[b]]"""
        ol = _OrdList(o, xs)
        if len(xs) == 0:
            return []
        elif len(xs) == 1:
            return [[xs[0][1]]]
        elif ol.is_trivial(ol):
            return [[x[1] for x in ol.xs]]
        elif ol.is_natural(ol):
            res: List[List[B]] = [[] for i in range(ol.o.n+1)]
            print(res)
            for k, v in ol.xs:
                print(k, v)
                res[k].append(v)
            return list(filter(lambda x: len(x) != 0, res))
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
        """
        elif ol.is_map(ol):
            reveal_type(ol.o.f)
            reveal_type(ol.xs)
            mapped = [(ol.o.f(k), v) for k, v in ol.xs]
            return sdisc(ol.o.target, mapped)
        """
        raise ValueError(f"Unknown Order {type(o)}")



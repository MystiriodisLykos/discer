# discer
Generic linear-time sorting and partitioning

Based on [this paper](http://hjemmesider.diku.dk/~henglein/papers/henglein2011a.pdf) by Fritz Henglein.

**Warning before you start using this library**

The standard libraries `sorted` function has linear-time sorting in the average case ([Timsort](https://en.wikipedia.org/wiki/Timsort)) and is way more optimized than this library and will almost certainly be faster than any sorting done with this library.

`discer.relations` contains a `Relation` type that represents the `Order` and `Equiv` types from the paper along with a few of standard relations from the paper.
`discer.grouping` and `discer.sorting` contain the respective `Equiv` and `Order` based from functions from the paper.

## Future work

Not in any particular order:
- Better README and docs
- Add some tests
- Optimize
    - Lazy evaluation over the key/pair list to reduces passes on it
    - Defer to `sorted` when it will likely (or definitely) run in linear time
    - The paper includes some optimizations not included
    - [Generic Multiset Programming](http://hjemmesider.diku.dk/~henglein/papers/henglein2011c.pdf) may also contain further optimizations
- Setup `SumL` and `ProductL` to use `TypeVarTuple`
- Functions to extend Relations
    - `list :: Relation a -> Relation [a]`
    - `tuple :: Relation a -> Relation Tuple[a, ...]`
    - `dict :: Relation a -> Relation Dict[k, a]`
- Refactor `sdisc` and `disc`, the functions are basically exactly the same except how they deal with `Natural`
- Add compatibility for at least Python 3.8+

## References
> Henglein, Fritz. "Generic top-down discrimination for sorting and partitioning in linear time." _Journal of Functional Programming_ 22.3 (2012): 300-374.
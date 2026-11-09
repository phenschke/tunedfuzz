from collections.abc import Callable, Iterable
from os import PathLike

import numpy as np
import numpy.typing as npt

Hit = tuple[str, float, int]
Processor = Callable[[str], str] | None

class Costs:
    def __init__(
        self, rows: Iterable[tuple[str, str, float]] | None = None, default: float = 1.0, *, symmetric: bool = True
    ) -> None: ...
    @staticmethod
    def read_csv(path: str | PathLike[str], default: float = 1.0, *, symmetric: bool = True) -> Costs: ...
    @staticmethod
    def pretrained(name: str = "general") -> Costs: ...
    def to_csv(self, path: str | PathLike[str]) -> None: ...
    def reversed(self) -> Costs: ...
    def rows(self) -> list[tuple[str, str, float]]: ...
    @property
    def default(self) -> float: ...
    @property
    def symmetric(self) -> bool: ...
    def __len__(self) -> int: ...

class Index:
    def __init__(self, names: Iterable[str], costs: Costs, *, processor: Processor = None) -> None: ...
    def __len__(self) -> int: ...
    def extract_one(self, query: str, *, score_cutoff: float | None = None, normalize: bool = False) -> Hit | None: ...
    def extract(
        self, query: str, *, limit: int | None = 5, score_cutoff: float | None = None, normalize: bool = False
    ) -> list[Hit]: ...
    def extract_one_many(
        self, queries: Iterable[str], *, score_cutoff: float | None = None, normalize: bool = False, workers: int = 1
    ) -> list[Hit | None]: ...
    def extract_many(
        self,
        queries: Iterable[str],
        *,
        limit: int | None = 5,
        score_cutoff: float | None = None,
        normalize: bool = False,
        workers: int = 1,
    ) -> list[list[Hit]]: ...

def distance(
    a: str,
    b: str,
    costs: Costs,
    *,
    score_cutoff: float | None = None,
    normalize: bool = False,
    processor: Processor = None,
) -> float: ...
def align(a: str, b: str, costs: Costs, *, processor: Processor = None) -> list[tuple[str, str, float]]: ...
def cdist(
    queries: Iterable[str],
    names: Iterable[str],
    costs: Costs,
    *,
    score_cutoff: float | None = None,
    normalize: bool = False,
    processor: Processor = None,
    workers: int = 1,
) -> npt.NDArray[np.float64]: ...
def cpdist(
    queries: Iterable[str],
    choices: Iterable[str],
    costs: Costs,
    *,
    score_cutoff: float | None = None,
    normalize: bool = False,
    processor: Processor = None,
    workers: int = 1,
) -> npt.NDArray[np.float64]: ...
def learn_costs(
    pairs: Iterable[tuple[str, str, int]],
    *,
    seeds: Iterable[tuple[str, str]] | None = None,
    fixed: Iterable[tuple[str, str, float]] | None = None,
    rounds: int = 4,
    p_floor: float = 1e-5,
    min_multi: int = 20,
    min_ratio: float = 0.5,
    shrink: float = 2.0,
    symmetric: bool = True,
    prior: Costs | None = None,
    prior_weight: float = 30.0,
) -> dict[str, list]: ...

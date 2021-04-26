from dataclasses import dataclass
from typing import Optional


@dataclass
class Bound:
    lower: list[int]
    upper: list[int]


@dataclass
class Color:
    name: str
    bounds: list[Bound]
    default_color: list[int]

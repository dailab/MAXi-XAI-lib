from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union, OrderedDict
from typing_extensions import TypedDict

from numpy import ndarray
from PIL.Image import Image


class Vector3(TypedDict):
    x: int
    y: int
    z: int


class Level(TypedDict):
    extent: Vector3
    downsample_factor: int
    generated: bool


class WSI(TypedDict):
    id: str
    extent: Vector3
    num_levels: int
    pixel_size_nm: Vector3
    tile_extent: Vector3
    levels: List[Level]


class Rectangle(TypedDict):
    upper_left: List[int]
    width: int
    height: int
    level: int


class Tile(TypedDict):
    image: ndarray
    x: int
    y: int


class EntityRect(TypedDict):
    x: int  # upper-
    y: int  # left
    w: int
    h: int


TileRequest = Callable[[Rectangle], Image]

InferenceCall = Callable[[Union[List[ndarray], ndarray, str]], ndarray]

MetaData = Dict[str, Any]

Processor = Callable[[ndarray], ndarray]

PresenterCallback = Callable[[Union[ndarray, Tuple[ndarray, ndarray]]], Any]

LogCallback = Callable[[Any], None]

SaveResCallback = Callable[[Tuple[OrderedDict[str, ndarray], MetaData]], None]

X0_Generator = Callable[[ndarray], ndarray]


@dataclass
class ExtractedEntities:
    contours: List
    confidences: List

    def count(self):
        return len(self.contours)

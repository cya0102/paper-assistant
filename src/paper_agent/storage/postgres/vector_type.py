"""Small SQLAlchemy pgvector type without a runtime pgvector-python dependency."""

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any, cast

from sqlalchemy.types import UserDefinedType


class VectorType(UserDefinedType[tuple[float, ...]]):
    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **kw: Any) -> str:
        del kw
        return f"VECTOR({self.dimension})"

    def bind_processor(self, dialect: object) -> Callable[[object], str | None]:
        del dialect

        def process(value: object) -> str | None:
            if value is None:
                return None
            values = tuple(float(item) for item in cast(Iterable[float | int | str], value))
            if len(values) != self.dimension:
                raise ValueError(f"Expected vector dimension {self.dimension}, got {len(values)}")
            return "[" + ",".join(format(item, ".12g") for item in values) + "]"

        return process

    def result_processor(
        self, dialect: object, coltype: object
    ) -> Callable[[object], tuple[float, ...] | None]:
        del dialect, coltype

        def process(value: object) -> tuple[float, ...] | None:
            if value is None:
                return None
            if isinstance(value, str):
                return tuple(float(item) for item in value.strip("[]").split(",") if item)
            return tuple(
                float(item) for item in cast(Iterable[float | int | str], value)
            )

        return process

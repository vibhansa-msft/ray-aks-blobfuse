import asyncio
from collections import deque
from dataclasses import asdict, fields
from enum import Enum, EnumMeta
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml


# In Python 3.11 there is a 'Self' introduced, but this is the
# workaround before that: https://peps.python.org/pep-0673/.
T = TypeVar("T")
TModelBase = TypeVar("TModelBase", bound="ModelBase")


class ModelEnumType(EnumMeta):
    def __new__(cls, subcls, bases, classdict):
        new_cls = super().__new__(cls, subcls, bases, classdict)

        # Assert that all enum values have docstrings.
        if not isinstance(new_cls.__docstrings__, dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in new_cls.__docstrings__.items()
        ):
            raise TypeError(
                f"ModelEnum '{new_cls.__name__}.__docstrings__' is the wrong type. "
                "Must be a Dict[ModelEnum, str] documenting all enum values"
            )

        missing = {v.value for v in new_cls if not v.name.startswith("_")} - set(
            new_cls.__docstrings__.keys()
        )
        if missing:
            raise ValueError(
                f"ModelEnum '{new_cls.__name__}.__docstrings__' is missing docstrings for values: {sorted(missing)}"
            )

        return new_cls


M = TypeVar("M", bound="ModelEnum")


class ModelEnum(str, Enum, metaclass=ModelEnumType):
    # Must be populated by subclasses to document every value.
    __docstrings__: ClassVar[Dict[str, str]] = {}

    def __str__(self):
        return self.name

    @classmethod
    def validate(cls: Type[M], value: Union[str, M]) -> M:
        allowed_values = [str(s) for s in cls.__members__ if not str(s).startswith("_")]
        if value.upper() in allowed_values:
            return cls(value.upper())
        else:
            raise ValueError(
                f"'{value.upper()}' is not a valid {cls.__name__}. Allowed values are {allowed_values}"
            )


class ModelBaseType(type):
    pass


class ModelBase(metaclass=ModelBaseType):
    def __new__(cls, *_args, **_kwargs):
        """Validate that the subclass is a conforming dataclass."""
        if (
            not hasattr(cls, "__dataclass_params__")
            or not cls.__dataclass_params__.frozen
        ):
            raise TypeError(
                "Subclasses of `ModelBase` must be dataclasses with `frozen=True`."
            )

        return super().__new__(cls)

    def _run_validators(self):
        """Run magic validator methods.

        Looks for methods with the name: `_validate_{field}`. The method will be called
        with the field value as its sole argument.

        If the validator has a return type annotation, the field will be updated to the
        returned value.
        """
        if getattr(self, "__ignore_validation__", False):
            return

        for field in fields(self):
            if not field.metadata.get("docstring", None):
                raise ValueError(
                    f"Model '{type(self).__name__}' is missing docstring for field '{field.name}'."
                )

            validator = getattr(self, f"_validate_{field.name}", None)
            if validator is None:
                raise RuntimeError(
                    f"Model '{type(self).__name__}' is missing validator method for field '{field.name}'."
                )

            signature: inspect.Signature = inspect.signature(validator)
            if len(signature.parameters) != 1:
                raise TypeError(
                    f"Validator for field '{field.name}' must take the field value as its only argument."
                )
            elif field.name not in signature.parameters:
                raise TypeError(
                    f"Validator for field '{field.name}' has invalid parameter name (must match the field name)."
                )
            elif signature.parameters[field.name].annotation == inspect.Parameter.empty:
                raise TypeError(
                    f"Validator for field '{field.name}' is missing a type annotation (should be '{field.type}')."
                )
            elif signature.parameters[field.name].annotation != field.type:
                raise TypeError(
                    f"Validator for field '{field.name}' has the wrong type annotation (should be '{field.type}')."
                )

            maybe_updated_field_value = validator(getattr(self, field.name))
            if signature.return_annotation != inspect.Signature.empty:
                # Only update the value if the validator has a return type annotation.
                object.__setattr__(self, field.name, maybe_updated_field_value)

    def __post_init__(self):
        self._run_validators()

    @classmethod
    def from_dict(cls: Type[TModelBase], d: Dict[str, Any]) -> TModelBase:
        return cls(**d)

    @classmethod
    def from_yaml(cls: Type[TModelBase], path: str, **kwargs) -> TModelBase:
        with open(path) as f:
            args = yaml.safe_load(f) or {}
            if kwargs:
                args.update(kwargs)
            return cls.from_dict(args)

    @classmethod
    def parse_from_internal_model(
        cls: Type[TModelBase], _internal_model: Any
    ) -> TModelBase:
        raise NotImplementedError(
            f"{cls.__name__}.parse_from_internal_model must be implemented"
        )

    def to_dict(self, *, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert the model to a dictionary representation.

        If `exclude_none` is `True`, keys whose values are `None` will be excluded.
        """
        # First, manually convert any nested ModelBase objects to dicts
        # because asdict() would convert them to dicts using raw field values
        converted_fields: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)

            if isinstance(value, ModelBase):
                # Convert nested ModelBase objects using their to_dict method
                converted_fields[field.name] = value.to_dict(exclude_none=exclude_none)
            elif isinstance(value, list):
                # Handle lists of ModelBase objects
                converted_list = []
                for item in value:
                    if isinstance(item, ModelBase):
                        converted_list.append(item.to_dict(exclude_none=exclude_none))
                    else:
                        converted_list.append(item)
                converted_fields[field.name] = converted_list
            else:
                converted_fields[field.name] = value

        def maybe_exclude_nones(i: Iterable[Tuple[str, Any]]):
            d = {}
            for k, v in i:
                # Exclude none values if specified.
                if exclude_none and v is None:
                    continue

                # Convert enums to their string representation.
                final_v = v
                if isinstance(v, ModelEnum):
                    final_v = v.value
                elif k in converted_fields and isinstance(
                    converted_fields[k], (dict, list)
                ):
                    final_v = converted_fields[k]
                d[k] = final_v

            return d

        return asdict(self, dict_factory=maybe_exclude_nones)

    def options(self: TModelBase, **kwargs) -> TModelBase:
        """Return a copy of the model with the provided fields updated.

        All fields in the constructor are supported. Those not provided will be unchanged.
        """
        new_instance_kwargs = {
            field.name: kwargs.pop(field.name)
            if field.name in kwargs
            else getattr(self, field.name)
            for field in fields(self)
        }
        if len(kwargs) > 0:
            raise ValueError(
                f"Unexpected values passed to '.options': {list(kwargs.keys())}."
            )

        return type(self)(**new_instance_kwargs)


class InternalListResponse(Generic[T]):
    results: List[T]
    has_more: bool


class ListResponse(List[TModelBase]):
    """
    List, but with `has_more` attribute.

    Attributes:
        has_more (bool): Whether there are more results to fetch.

    To use:
    ```
    entities: ListResponse[X] = anyscale.<entity>.list(limit=10)
    for x in entities:
        print(x)
    ```
    """

    def __init__(
        self,
        after: Optional[str],
        limit: Optional[int],
        get_next_page: Callable[[Optional[str]], InternalListResponse[T]],
        cls: Type[TModelBase],
    ):
        super().__init__()
        self.has_more = True

        self._after = after
        self._limit = limit or 1000
        self._get_next_page = get_next_page
        self._cls = cls

        # Fetch the first page
        self._fetch_next_page()

    def _fetch_next_page(self):
        if not self.has_more:
            return
        next_page = self._get_next_page(self._after)
        self.has_more = next_page.has_more
        self.extend([self._cls.parse_from_internal_model(r) for r in next_page.results])
        del self[self._limit :]
        self._after = self[-1].id if self else None  # type: ignore

    def __iter__(self) -> Iterator[TModelBase]:
        index = 0
        while index < len(self):
            yield self[index]
            index += 1
            if index >= self._limit:
                return
            if index >= len(self) and self.has_more:
                self._fetch_next_page()


RT = TypeVar("RT")


class ResultIterator(Generic[RT]):
    """
    Lazily fetch and parse pages from a paged-list API that returns
    Pydantic models with `.results` and `.metadata.next_paging_token`.
    """

    def __init__(
        self,
        *,
        page_token: Optional[str],
        max_items: Optional[int],
        fetch_page: Callable[[Optional[str]], Any],
        parse_fn: Optional[Callable[[Any], RT]] = None,
        async_parse_fn: Optional[Callable[[Any], Awaitable[RT]]] = None,
    ):
        if parse_fn and async_parse_fn:
            raise ValueError("Only one of parse_fn or async_parse_fn may be provided")

        self._token = page_token
        self._max = max_items
        self._fetch = fetch_page
        self._parse = parse_fn
        self._aparse = async_parse_fn
        self._buffer: Deque[RT] = deque()
        self._count = 0
        self._finished = False

    def __iter__(self) -> Iterator[RT]:
        while True:
            # 1) Drain the buffer
            while self._buffer:
                if self._max is not None and self._count >= self._max:
                    return
                self._count += 1
                yield self._buffer.popleft()

            # 2) Done?
            if self._finished or (self._max is not None and self._count >= self._max):
                return

            # 3) Fetch the next page (Pydantic model)
            page = self._fetch(self._token)
            raw_results = page.results
            self._token = page.metadata.next_paging_token

            # 4) No more data?
            if not raw_results:
                self._finished = True
                return

            # 5) Parse—sync or async
            if self._aparse:
                processed = asyncio.run(
                    ResultIterator._process_items_async(raw_results, self._aparse)
                )
                self._buffer.extend(processed)

            elif self._parse:
                try:
                    for raw in raw_results:
                        self._buffer.append(self._parse(raw))
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"sync parse error: {e}") from e

            else:
                # No parser: assume items are already RT
                self._buffer.extend(raw_results)  # type: ignore

            # 6) If no next token, finish on next loop
            if self._token is None:
                self._finished = True

    @staticmethod
    async def _process_items_async(
        items: List[Any], parser: Callable[[Any], Awaitable[RT]],
    ) -> List[RT]:
        if not items:
            return []
        tasks = [parser(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed: List[RT] = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                raise RuntimeError(f"async parse failed on item {idx}: {res}") from res
            processed.append(res)
        return processed

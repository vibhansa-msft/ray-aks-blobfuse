from enum import Enum
from typing import Optional, Union

import click
from typing_extensions import Literal

from anyscale.anyscale_pydantic import BaseModel


class EntityType(str, Enum):
    ID = "ID"
    NAME = "NAME"


class Entity(BaseModel):
    type: EntityType


class IdBasedEntity(Entity):
    type: Literal[EntityType.ID] = EntityType.ID
    id: str


class NameBasedEntity(Entity):
    type: Literal[EntityType.NAME] = EntityType.NAME
    name: str
    version: Optional[int] = None


def format_inputs_to_entity(
    name: Optional[str], entity_id: Optional[str]
) -> Union[IdBasedEntity, NameBasedEntity]:
    """
    Share method for CLI commands to accept either the name of the id of an entity.
    """
    if int(bool(name)) + int(bool(entity_id)) != 1:
        raise click.ClickException("Please provide exactly one of: name, id.")
    elif name:
        return NameBasedEntity(name=name)
    elif entity_id:
        return IdBasedEntity(id=entity_id)
    else:
        raise click.ClickException("Please provide exactly one of: name, id.")

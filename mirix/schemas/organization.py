from datetime import datetime
from typing import Optional
import uuid

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase
from mirix.utils import create_random_username, get_utc_time


class OrganizationBase(MirixBase):
    __id_prefix__ = "org"


def _generate_org_id() -> str:
    """Generate a random organization ID."""
    return f"org-{uuid.uuid4().hex[:8]}"


class Organization(OrganizationBase):
    id: str = Field(
        default_factory=_generate_org_id,
        description="The unique identifier of the organization.",
    )
    name: str = Field(
        create_random_username(),
        description="The name of the organization.",
        json_schema_extra={"default": "SincereYogurt"},
    )
    created_at: Optional[datetime] = Field(
        default_factory=get_utc_time,
        description="The creation date of the organization.",
    )


class OrganizationCreate(OrganizationBase):
    id: Optional[str] = Field(None, description="The unique identifier of the organization.")
    name: Optional[str] = Field(None, description="The name of the organization.")

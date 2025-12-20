"""
Provider schemas for storing API keys.

Supported providers:
- openai: OpenAI API
- anthropic: Anthropic Claude API
- google_ai: Google AI (Gemini) API
- azure_openai: Azure OpenAI API
"""

from datetime import datetime
from typing import Optional

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase


class ProviderBase(MirixBase):
    __id_prefix__ = "provider"


class Provider(ProviderBase):
    """Provider class for storing API keys."""

    id: Optional[str] = Field(
        None,
        description="The id of the provider, lazily created by the database manager.",
    )
    name: str = Field(..., description="The name of the provider")
    api_key: Optional[str] = Field(
        None, description="API key used for requests to the provider."
    )
    organization_id: Optional[str] = Field(
        None, description="The organization id of the user"
    )
    updated_at: Optional[datetime] = Field(
        None, description="The last update timestamp of the provider."
    )

    def resolve_identifier(self):
        if not self.id:
            self.id = ProviderBase._generate_id(prefix=ProviderBase.__id_prefix__)


class ProviderCreate(ProviderBase):
    """Schema for creating a new provider."""

    name: str = Field(..., description="The name of the provider.")
    api_key: str = Field(..., description="API key used for requests to the provider.")


class ProviderUpdate(ProviderBase):
    """Schema for updating an existing provider."""

    id: str = Field(..., description="The id of the provider to update.")
    api_key: str = Field(..., description="API key used for requests to the provider.")

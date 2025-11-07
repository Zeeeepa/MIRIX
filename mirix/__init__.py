__version__ = "0.2.6"


# import clients
from mirix.client.client import LocalClient as LocalClient
from mirix.client.client import create_client as create_client
from mirix.client.remote_client import MirixClient as MirixClient

# # imports for easier access
from mirix.schemas.agent import AgentState as AgentState
from mirix.schemas.block import Block as Block
from mirix.schemas.embedding_config import EmbeddingConfig as EmbeddingConfig
from mirix.schemas.enums import JobStatus as JobStatus
from mirix.schemas.llm_config import LLMConfig as LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary as ArchivalMemorySummary
from mirix.schemas.memory import BasicBlockMemory as BasicBlockMemory
from mirix.schemas.memory import ChatMemory as ChatMemory
from mirix.schemas.memory import Memory as Memory
from mirix.schemas.memory import RecallMemorySummary as RecallMemorySummary
from mirix.schemas.message import Message as Message
from mirix.schemas.mirix_message import MirixMessage as MirixMessage
from mirix.schemas.openai.chat_completion_response import (
    UsageStatistics as UsageStatistics,
)
from mirix.schemas.organization import Organization as Organization
from mirix.schemas.tool import Tool as Tool
from mirix.schemas.usage import MirixUsageStatistics as MirixUsageStatistics
from mirix.schemas.user import User as User

# Import the new SDK interface
from mirix.sdk import Mirix as Mirix
from mirix.sdk import load_config as load_config

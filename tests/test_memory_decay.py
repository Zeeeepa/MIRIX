"""
Server-side tests for memory decay (fade + expire).

These tests insert backdated episodic events directly via managers and verify:
1) faded memories are excluded by default
2) expired memories are deleted by cleanup
"""

import configparser
import os
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="speech_recognition",
)

from mirix.schemas.agent import (
    AgentType,
    CreateMetaAgent,
    MemoryConfig,
    MemoryDecayConfig,
    UpdateAgent,
)
from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.organization import Organization
from mirix.schemas.user import User as PydanticUser


TEST_RUN_ID = uuid.uuid4().hex[:8]
TEST_ORG_ID = f"test-decay-org-{TEST_RUN_ID}"
TEST_CLIENT_ID = f"test-decay-client-{TEST_RUN_ID}"
TEST_USER_ID = f"test-decay-user-{TEST_RUN_ID}"

OLD_TAGS = {"test": "memory-decay", "batch": "old"}
RECENT_TAGS = {"test": "memory-decay", "batch": "recent"}

DECAY_CONFIG = {"fade_after_days": 1, "expire_after_days": 2}


@pytest.fixture(scope="module")
def server_context():
    original_config_path = os.environ.get("MEMGPT_CONFIG_PATH")
    temp_root = Path(__file__).parent.parent / ".test-data" / f"memory-decay-{TEST_RUN_ID}"
    temp_root.mkdir(parents=True, exist_ok=True)
    config_path = temp_root / "config.ini"

    config = configparser.ConfigParser()
    config["recall_storage"] = {"type": "sqlite", "path": ":memory:"}
    config["archival_storage"] = {"type": "sqlite", "path": ":memory:"}
    config["metadata_storage"] = {"type": "sqlite", "path": ":memory:"}
    with config_path.open("w", encoding="utf-8") as handle:
        config.write(handle)

    os.environ["MEMGPT_CONFIG_PATH"] = str(config_path)

    from mirix.server.server import SyncServer
    from mirix.settings import settings

    server = SyncServer()
    original_embeddings = settings.build_embeddings_for_memory
    settings.build_embeddings_for_memory = False

    try:
        # Organization
        try:
            org = server.organization_manager.get_organization_by_id(TEST_ORG_ID)
        except Exception:
            org = server.organization_manager.create_organization(
                pydantic_org=Organization(id=TEST_ORG_ID, name=TEST_ORG_ID)
            )

        # Client
        try:
            client = server.client_manager.get_client_by_id(TEST_CLIENT_ID)
        except Exception:
            client = server.client_manager.create_client(
                pydantic_client=Client(
                    id=TEST_CLIENT_ID,
                    name=TEST_CLIENT_ID,
                    organization_id=org.id,
                    status="active",
                    scope="read_write",
                )
            )

        # User
        try:
            user = server.user_manager.get_user_by_id(TEST_USER_ID)
        except Exception:
            user = server.user_manager.create_user(
                pydantic_user=PydanticUser(
                    id=TEST_USER_ID,
                    name=TEST_USER_ID,
                    organization_id=org.id,
                    timezone=server.user_manager.DEFAULT_TIME_ZONE,
                ),
                client_id=client.id,
            )

        # Meta agent with decay config
        llm_config = LLMConfig(
            model="gpt-4o-mini",
            model_endpoint_type="openai",
            model_endpoint="https://api.openai.com/v1",
            context_window=128000,
        )
        memory_config = MemoryConfig(
            decay=MemoryDecayConfig(**DECAY_CONFIG)
        )

        meta_agent = server.agent_manager.create_meta_agent(
            meta_agent_create=CreateMetaAgent(
                name=f"meta_memory_agent_{TEST_RUN_ID}",
                agents=["meta_memory_agent", "episodic_memory_agent"],
                llm_config=llm_config,
                embedding_config=None,
                memory=memory_config,
            ),
            actor=client,
            user_id=user.id,
        )

        if not meta_agent.memory_config:
            meta_agent = server.agent_manager.update_agent(
                agent_id=meta_agent.id,
                agent_update=UpdateAgent(memory_config={"decay": DECAY_CONFIG}),
                actor=client,
            )

        meta_agent = server.agent_manager.get_agent_by_id(
            agent_id=meta_agent.id,
            actor=client,
        )

        episodic_agent = None
        sub_agents = server.agent_manager.list_agents(actor=client, parent_id=meta_agent.id)
        for agent in sub_agents:
            if agent.agent_type == AgentType.episodic_memory_agent:
                episodic_agent = agent
                break
        if episodic_agent is None:
            raise AssertionError("Episodic memory agent not created")

        context = {
            "server": server,
            "client": client,
            "user": user,
            "meta_agent": meta_agent,
            "episodic_agent": episodic_agent,
        }
        yield context
    finally:
        settings.build_embeddings_for_memory = original_embeddings
        if original_config_path is None:
            os.environ.pop("MEMGPT_CONFIG_PATH", None)
        else:
            os.environ["MEMGPT_CONFIG_PATH"] = original_config_path


@pytest.fixture(scope="module")
def seeded_memories(server_context):
    server = server_context["server"]
    client = server_context["client"]
    user = server_context["user"]
    episodic_agent = server_context["episodic_agent"]

    old_time = (datetime.now(timezone.utc) - timedelta(days=3)).replace(microsecond=0)
    recent_time = (datetime.now(timezone.utc) - timedelta(hours=6)).replace(microsecond=0)

    server.episodic_memory_manager.insert_event(
        actor=client,
        agent_state=episodic_agent,
        agent_id=server_context["meta_agent"].id,
        event_type="activity",
        timestamp=old_time,
        event_actor="user",
        summary="Filed an expense report",
        details="Three days ago I filed an expense report.",
        organization_id=user.organization_id,
        filter_tags=OLD_TAGS,
        user_id=user.id,
    )

    server.episodic_memory_manager.insert_event(
        actor=client,
        agent_state=episodic_agent,
        agent_id=server_context["meta_agent"].id,
        event_type="activity",
        timestamp=recent_time,
        event_actor="user",
        summary="Reviewed quarterly OKRs",
        details="Today I reviewed the quarterly OKRs.",
        organization_id=user.organization_id,
        filter_tags=RECENT_TAGS,
        user_id=user.id,
    )

    return server_context


def test_fade_excludes_old_memory(seeded_memories):
    server = seeded_memories["server"]
    user = seeded_memories["user"]
    episodic_agent = seeded_memories["episodic_agent"]
    meta_agent = seeded_memories["meta_agent"]

    decay = (meta_agent.memory_config or {}).get("decay")
    assert decay is not None
    fade_after_days = decay.get("fade_after_days")
    assert fade_after_days is not None

    old_results = server.episodic_memory_manager.list_episodic_memory(
        agent_state=episodic_agent,
        user=user,
        limit=10,
        filter_tags=OLD_TAGS,
        include_faded=False,
        fade_after_days=fade_after_days,
    )
    assert len(old_results) == 0

    recent_results = server.episodic_memory_manager.list_episodic_memory(
        agent_state=episodic_agent,
        user=user,
        limit=10,
        filter_tags=RECENT_TAGS,
        include_faded=False,
        fade_after_days=fade_after_days,
    )
    assert len(recent_results) > 0

    old_included = server.episodic_memory_manager.list_episodic_memory(
        agent_state=episodic_agent,
        user=user,
        limit=10,
        filter_tags=OLD_TAGS,
        include_faded=True,
        fade_after_days=fade_after_days,
    )
    assert len(old_included) > 0


def test_cleanup_deletes_expired(seeded_memories):
    server = seeded_memories["server"]
    user = seeded_memories["user"]
    episodic_agent = seeded_memories["episodic_agent"]
    meta_agent = seeded_memories["meta_agent"]

    decay = (meta_agent.memory_config or {}).get("decay")
    assert decay is not None
    expire_after_days = decay.get("expire_after_days")
    assert expire_after_days is not None

    deleted = server.episodic_memory_manager.delete_expired_memories(
        user=user,
        expire_after_days=expire_after_days,
    )
    assert deleted >= 1

    old_after = server.episodic_memory_manager.list_episodic_memory(
        agent_state=episodic_agent,
        user=user,
        limit=10,
        filter_tags=OLD_TAGS,
        include_faded=True,
        fade_after_days=expire_after_days,
    )
    assert len(old_after) == 0

    recent_after = server.episodic_memory_manager.list_episodic_memory(
        agent_state=episodic_agent,
        user=user,
        limit=10,
        filter_tags=RECENT_TAGS,
        include_faded=True,
        fade_after_days=expire_after_days,
    )
    assert len(recent_after) > 0

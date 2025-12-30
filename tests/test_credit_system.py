import importlib.machinery
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest


def _ensure_mirix_package():
    if "mirix" in sys.modules:
        return
    package_root = Path(__file__).resolve().parents[1] / "mirix"
    stub = ModuleType("mirix")
    stub.__file__ = str(package_root / "__init__.py")
    stub.__path__ = [str(package_root)]
    stub.__version__ = "0.0"
    stub.__spec__ = importlib.machinery.ModuleSpec(
        "mirix", loader=None, is_package=True
    )
    stub.__spec__.submodule_search_locations = stub.__path__
    sys.modules["mirix"] = stub


_ensure_mirix_package()

import mirix.services.client_manager as client_manager_module  # noqa: E402
from mirix.utils import get_utc_time  # noqa: E402
from mirix.llm_api.llm_client import LLMClient  # noqa: E402
from mirix.pricing import calculate_cost  # noqa: E402
from mirix.schemas.client import Client as PydanticClient  # noqa: E402
from mirix.schemas.llm_config import LLMConfig  # noqa: E402
from mirix.schemas.openai.chat_completion_response import (  # noqa: E402
    ChatCompletionResponse,
    Choice,
    Message,
    PromptTokensDetails,
    UsageStatistics,
)
from mirix.services.client_manager import ClientManager  # noqa: E402


class StubLLMClient:
    def __init__(self, response: ChatCompletionResponse):
        self.response = response
        self.call_count = 0

    def send_llm_request(
        self,
        messages,
        tools=None,
        stream=False,
        force_tool_call=None,
        get_input_data_for_debugging=False,
        existing_file_uris=None,
    ):
        self.call_count += 1
        return self.response


class FakeClient:
    def __init__(self, client_id: str, credits: float):
        self.id = client_id
        self.credits = credits
        self.update_calls = 0

    def update_with_redis(self, session, actor=None):
        self.update_calls += 1

    def to_pydantic(self) -> PydanticClient:
        return PydanticClient(
            id=self.id,
            name=f"Test Client {self.id}",
            organization_id="org-test",
            scope="read_write",
            credits=self.credits,
        )


def _build_response(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> ChatCompletionResponse:
    prompt_details = (
        PromptTokensDetails(cached_tokens=cached_tokens)
        if cached_tokens > 0
        else None
    )
    usage = UsageStatistics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=prompt_details,
    )
    return ChatCompletionResponse(
        id=str(uuid.uuid4()),
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="Test response"),
            )
        ],
        created=get_utc_time(),
        model=model,
        usage=usage,
    )


def _patch_llm_client(monkeypatch, expected_endpoint_type: str, response):
    stub = StubLLMClient(response)

    def _create(llm_config):
        assert llm_config.model_endpoint_type == expected_endpoint_type
        assert llm_config.model == response.model
        return stub

    monkeypatch.setattr(LLMClient, "create", staticmethod(_create))
    return stub


@contextmanager
def _fake_session():
    yield object()


def _setup_client_manager(monkeypatch, store: dict[str, FakeClient]) -> ClientManager:
    def fake_read(cls, db_session, identifier):
        return store[identifier]

    monkeypatch.setattr(
        client_manager_module.ClientModel,
        "read",
        classmethod(fake_read),
    )
    manager = ClientManager.__new__(ClientManager)
    manager.session_maker = _fake_session
    return manager


def _deduct_from_usage(
    client_mgr: ClientManager,
    client_id: str,
    model: str,
    usage: UsageStatistics,
):
    cached_tokens = usage.cached_tokens
    non_cached_prompt_tokens = max(usage.prompt_tokens - cached_tokens, 0)
    cost = calculate_cost(
        model=model,
        prompt_tokens=non_cached_prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=cached_tokens,
    )
    client_mgr.deduct_credits(client_id, cost)
    return cost


def test_deduct_credits_updates_balance(monkeypatch):
    client_id = f"test-credit-direct-{uuid.uuid4().hex[:8]}"
    store = {client_id: FakeClient(client_id, 25.0)}
    client_mgr = _setup_client_manager(monkeypatch, store)

    updated = client_mgr.deduct_credits(client_id, 4.5)
    assert updated.credits == pytest.approx(20.5)
    assert store[client_id].credits == pytest.approx(20.5)
    assert store[client_id].update_calls == 1


def test_credit_deduction_openai_call(monkeypatch):
    model = "gpt-4o-mini"
    prompt_tokens = 120
    completion_tokens = 30
    cached_tokens = 10

    llm_config = LLMConfig(
        model=model,
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
        context_window=8192,
    )
    response = _build_response(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )
    stub = _patch_llm_client(monkeypatch, "openai", response)

    client_id = f"test-credit-openai-{uuid.uuid4().hex[:8]}"
    store = {client_id: FakeClient(client_id, 100.0)}
    client_mgr = _setup_client_manager(monkeypatch, store)

    llm_client = LLMClient.create(llm_config)
    llm_response = llm_client.send_llm_request(messages=[])
    before = store[client_id].credits
    cost = _deduct_from_usage(client_mgr, client_id, model, llm_response.usage)
    after = store[client_id].credits

    assert stub.call_count == 1
    assert cost == pytest.approx(
        calculate_cost(
            model=model,
            prompt_tokens=prompt_tokens - cached_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
    )
    assert after == pytest.approx(before - cost)


def test_credit_deduction_gemini_call(monkeypatch):
    model = "gemini-3-flash-preview"
    prompt_tokens = 80
    completion_tokens = 25
    cached_tokens = 0

    llm_config = LLMConfig(
        model=model,
        model_endpoint_type="google_ai",
        model_endpoint="https://generativelanguage.googleapis.com",
        context_window=8192,
    )
    response = _build_response(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )
    stub = _patch_llm_client(monkeypatch, "google_ai", response)

    client_id = f"test-credit-gemini-{uuid.uuid4().hex[:8]}"
    store = {client_id: FakeClient(client_id, 100.0)}
    client_mgr = _setup_client_manager(monkeypatch, store)

    llm_client = LLMClient.create(llm_config)
    llm_response = llm_client.send_llm_request(messages=[])
    before = store[client_id].credits
    cost = _deduct_from_usage(client_mgr, client_id, model, llm_response.usage)
    after = store[client_id].credits

    assert stub.call_count == 1
    assert cost == pytest.approx(
        calculate_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
    )
    assert after == pytest.approx(before - cost)


@pytest.mark.skip(reason="TODO: add credit deduction tests for other model providers")
def test_credit_deduction_other_models_placeholder():
    assert True

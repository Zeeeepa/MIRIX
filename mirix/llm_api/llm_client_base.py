import logging
import time
from abc import abstractmethod
from typing import List, Optional

from mirix.errors import LLMError
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.services.cloud_file_mapping_manager import CloudFileMappingManager
from mirix.services.file_manager import FileManager


class LLMClientBase:
    """
    Abstract base class for LLM clients, formatting the request objects,
    handling the downstream request and parsing into chat completions response format
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        use_tool_naming: bool = True,
    ):
        self.llm_config = llm_config
        self.use_tool_naming = use_tool_naming
        self.file_manager = FileManager()
        self.cloud_file_mapping_manager = CloudFileMappingManager()
        self.logger = logging.getLogger(f"Mirix.LLMClientBase")

    def send_llm_request(
        self,
        messages: List[Message],
        tools: Optional[List[dict]] = None,
        stream: bool = False,
        force_tool_call: Optional[str] = None,
        get_input_data_for_debugging: bool = False,
        existing_file_uris: Optional[List[str]] = None,
    ) -> ChatCompletionResponse:
        """
        Issues a request to the downstream model endpoint and parses response.
        """
        request_data = self.build_request_data(
            messages,
            self.llm_config,
            tools,
            force_tool_call,
            existing_file_uris=existing_file_uris,
        )

        if get_input_data_for_debugging:
            return request_data

        try:
            t1 = time.time()
            response_data = self.request(request_data)
            t2 = time.time()
            self.logger.debug("LLM request time: %.2f seconds", t2 - t1)
        except Exception as e:
            raise self.handle_llm_error(e)

        chat_completion_data = self.convert_response_to_chat_completion(
            response_data, messages
        )

        return chat_completion_data

    @abstractmethod
    def build_request_data(
        self,
        messages: List[Message],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
    ) -> dict:
        """
        Constructs a request object in the expected data format for this client.
        """
        raise NotImplementedError

    @abstractmethod
    def request(self, request_data: dict) -> dict:
        """
        Performs underlying request to llm and returns raw response.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[Message],
    ) -> ChatCompletionResponse:
        """
        Converts custom response format from llm client into an OpenAI
        ChatCompletionsResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    def handle_llm_error(self, e: Exception) -> Exception:
        """
        Maps provider-specific errors to common LLMError types.
        Each LLM provider should implement this to translate their specific errors.

        Args:
            e: The original provider-specific exception

        Returns:
            An LLMError subclass that represents the error in a provider-agnostic way
        """
        return LLMError(f"Unhandled LLM error: {str(e)}")

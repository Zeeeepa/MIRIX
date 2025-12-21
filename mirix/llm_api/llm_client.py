from typing import Optional

from mirix.llm_api.llm_client_base import LLMClientBase
from mirix.schemas.llm_config import LLMConfig


class LLMClient:
    """Factory class for creating LLM clients based on the model endpoint type.

    Each client supports reasoning/thinking capabilities when enabled via llm_config:
    - OpenAI: o-series (o1, o3, o4) and gpt-5 models with reasoning_effort parameter
    - Azure OpenAI: Same as OpenAI for reasoning models deployed on Azure
    - Anthropic: Claude models with extended thinking (enable_reasoner + max_reasoning_tokens)
    - Google AI: Gemini 2.5+ models with thinkingConfig (enable_reasoner + max_reasoning_tokens)

    To enable reasoning, set in LLMConfig:
    - enable_reasoner: True (auto-enabled for o1/o3/o4/gpt-5 models)
    - max_reasoning_tokens: Number of tokens for thinking (e.g., 1024-8192)
    - reasoning_effort: "low" | "medium" | "high" (OpenAI only)
    """

    @staticmethod
    def create(
        llm_config: LLMConfig,
    ) -> Optional[LLMClientBase]:
        """
        Create an LLM client based on the model endpoint type.

        Args:
            llm_config: Configuration for the LLM model, including reasoning settings

        Returns:
            An instance of LLMClientBase subclass configured with reasoning support

        Raises:
            ValueError: If the model endpoint type is not supported
        """
        match llm_config.model_endpoint_type:
            case "openai":
                from mirix.llm_api.openai_client import OpenAIClient

                return OpenAIClient(
                    llm_config=llm_config,
                )
            case "azure_openai":
                from mirix.llm_api.azure_openai_client import AzureOpenAIClient

                return AzureOpenAIClient(
                    llm_config=llm_config,
                )
            case "anthropic":
                from mirix.llm_api.anthropic_client import AnthropicClient

                return AnthropicClient(
                    llm_config=llm_config,
                )
            case "google_ai":
                from mirix.llm_api.google_ai_client import GoogleAIClient

                return GoogleAIClient(
                    llm_config=llm_config,
                )
            case _:
                return None

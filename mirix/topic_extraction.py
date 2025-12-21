from typing import Any, Dict, List, Optional

import requests

from mirix.log import get_logger
from mirix.settings import model_settings

logger = get_logger(__name__)


def flatten_messages_to_plain_text(messages: List[Dict[str, Any]]) -> str:
    transcript_parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "user")
        content = msg.get("content")
        parts = []

        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if text:
                        parts.append(text.strip())
                elif isinstance(chunk, str):
                    parts.append(chunk.strip())
        elif isinstance(content, str):
            parts.append(content.strip())

        combined = " ".join(filter(None, parts)).strip()
        if combined:
            transcript_parts.append(f"{role.upper()}: {combined}")

    return "\n".join(transcript_parts)


def extract_topics_with_ollama(
    messages: List[Dict[str, Any]],
    model_name: str,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Extract topics using a locally hosted Ollama model via the /api/chat endpoint.

    Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
    """
    base_url = base_url or model_settings.ollama_base_url
    if not base_url:
        logger.warning(
            "Ollama topic extraction requested (%s) but MIRIX_OLLAMA_BASE_URL is not configured",
            model_name,
        )
        return None

    conversation = flatten_messages_to_plain_text(messages)
    if not conversation:
        logger.debug("No text content found in messages for Ollama topic extraction")
        return None

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that extracts the topic from the user's input. "
                    "Return a concise list of topics separated by ';' and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Conversation transcript:\n"
                    f"{conversation}\n\n"
                    "Respond ONLY with the topic(s) separated by ';'."
                ),
            },
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=30,
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        response_data = response.json()
    except requests.RequestException as exc:
        logger.error("Failed to extract topics with Ollama model %s: %s", model_name, exc)
        return None

    message_payload = response_data.get("message") if isinstance(response_data, dict) else None
    text_response: Optional[str] = None
    if isinstance(message_payload, dict):
        text_response = message_payload.get("content")
    elif isinstance(response_data, dict):
        text_response = response_data.get("content")

    if isinstance(text_response, str):
        topics = text_response.strip()
        logger.debug("Extracted topics via Ollama model %s: %s", model_name, topics)
        return topics or None

    logger.warning(
        "Unexpected response format from Ollama topic extraction: %s",
        response_data,
    )
    return None

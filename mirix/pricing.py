"""
Model pricing configuration for credit calculation.

Pricing is defined as dollars per 1 million tokens.
1 credit = 1 dollar

To add a new model, add an entry to MODEL_PRICING with:
- input: cost per 1M input (prompt) tokens in dollars
- cached_input: cost per 1M cached input tokens in dollars (optional, None if not supported)
- output: cost per 1M output (completion) tokens in dollars
"""

from typing import Optional, Tuple

from mirix.log import get_logger

logger = get_logger(__name__)

# Pricing markup multiplier for profit margin
# 1.0 = cost price (no profit)
# 1.2 = 20% markup
# 1.5 = 50% markup (recommended)
# 2.0 = 100% markup
PRICING_ALPHA = 1.5

# Pricing per 1 million tokens (in dollars)
# Format: "model_name": {"input": $X.XX, "cached_input": $X.XX or None, "output": $Y.YY}
# Updated: December 2024 from https://platform.openai.com/docs/pricing
MODEL_PRICING = {
    # OpenAI Models - GPT-5 Series
    "gpt-5.2": {"input": 0.875, "cached_input": 0.0875, "output": 7.00},
    "gpt-5.2-chat-latest": {"input": 0.875, "cached_input": 0.0875, "output": 7.00},
    "gpt-5.1": {"input": 0.625, "cached_input": 0.0625, "output": 5.00},
    "gpt-5.1-chat-latest": {"input": 0.625, "cached_input": 0.0625, "output": 5.00},
    "gpt-5": {"input": 0.625, "cached_input": 0.0625, "output": 5.00},
    "gpt-5-mini": {"input": 0.125, "cached_input": 0.0125, "output": 1.00},
    "gpt-5-nano": {"input": 0.025, "cached_input": 0.0025, "output": 0.20},

    # OpenAI Models - GPT-4.1 Series
    "gpt-4.1": {"input": 1.00, "cached_input": 0.25, "output": 4.00},
    "gpt-4.1-mini": {"input": 0.20, "cached_input": 0.05, "output": 0.80},
    "gpt-4.1-nano": {"input": 0.05, "cached_input": 0.0125, "output": 0.20},

    # OpenAI Models - GPT-4o Series
    "gpt-4o": {"input": 1.25, "cached_input": 0.3125, "output": 5.00},
    "gpt-4o-mini": {"input": 0.075, "cached_input": 0.01875, "output": 0.30},

    # OpenAI Models - o-Series (Reasoning)
    "o1": {"input": 7.50, "cached_input": 1.875, "output": 30.00},
    "o1-mini": {"input": 0.55, "cached_input": 0.1375, "output": 2.20},
    "o3": {"input": 1.00, "cached_input": 0.25, "output": 4.00},
    "o3-pro": {"input": 10.00, "cached_input": None, "output": 40.00},
    "o3-mini": {"input": 0.55, "cached_input": 0.1375, "output": 2.20},
    "o4-mini": {"input": 0.55, "cached_input": 0.1375, "output": 2.20},

    # Anthropic Claude Models
    # TODO: add pricing for Anthropic Claude models

    # xAI Grok Models
    "grok-4-1-fast-reasoning": {"input": 0.20, "cached_input": 0.05, "output": 0.50},
    "grok-4-1-fast-non-reasoning": {"input": 0.20, "cached_input": 0.05, "output": 0.50},
    "grok-4-fast-reasoning": {"input": 0.20, "cached_input": 0.05, "output": 0.50},
    "grok-4-fast-non-reasoning": {"input": 0.20, "cached_input": 0.05, "output": 0.50},
    "grok-4-0709": {"input": 3.00, "cached_input": 0.75, "output": 15.00},
    "grok-3-mini": {"input": 0.30, "cached_input": 0.075, "output": 0.50},
    "grok-3": {"input": 3.00, "cached_input": 0.75, "output": 15.00},

    # Google Gemini Models
    "gemini-3-flash-preview": {"input": 0.50, "cached_input": 0.05, "output": 3.00}, # verified
    "gemini-2.0-flash": {"input": 0.30, "cached_input": 0.03, "output": 2.50},  # verified
    "gemini-2.0-flash-lite": {"input": 0.10, "cached_input": 0.01, "output": 0.40},  # verified
}


def get_model_pricing(model: str) -> Tuple[float, Optional[float], float]:
    """
    Get the pricing for a specific model (with markup applied).

    Args:
        model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")

    Returns:
        Tuple of (input_price_per_1m, cached_input_price_per_1m, output_price_per_1m) in dollars
        cached_input_price_per_1m is None if caching is not supported for the model
        All prices include the PRICING_ALPHA markup.
    
    Raises:
        ValueError: If the model is not found in MODEL_PRICING
    """

    def apply_alpha(pricing: dict) -> Tuple[float, Optional[float], float]:
        input_price = pricing["input"] * PRICING_ALPHA
        cached_price = pricing.get("cached_input")
        if cached_price is not None:
            cached_price = cached_price * PRICING_ALPHA
        output_price = pricing["output"] * PRICING_ALPHA
        return input_price, cached_price, output_price

    if model in MODEL_PRICING:
        return apply_alpha(MODEL_PRICING[model])

    # Allow versioned models to match base models
    # e.g., "gpt-5-mini-2025-01-01" matches "gpt-5-mini"
    # but "gpt-99" should NOT match anything
    for model_key in MODEL_PRICING:
        if model.startswith(model_key + "-") or model.startswith(model_key):
            logger.debug("Using pricing for %s (matched from %s)", model_key, model)
            return apply_alpha(MODEL_PRICING[model_key])

    # Get list of available models for error message
    available_models = sorted(MODEL_PRICING.keys())
    error_msg = (
        f"Model '{model}' is not supported. "
        f"Available models: {', '.join(available_models[:10])}"
    )
    if len(available_models) > 10:
        error_msg += f"... and {len(available_models) - 10} more"
    
    logger.error(error_msg)
    raise ValueError(error_msg)


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """
    Calculate the cost in dollars (credits) for a given usage.

    Args:
        model: The model name
        prompt_tokens: Number of input/prompt tokens (non-cached)
        completion_tokens: Number of output/completion tokens
        cached_tokens: Number of cached input tokens (default 0)

    Returns:
        Cost in dollars (1 dollar = 1 credit).
        Note: Price includes PRICING_ALPHA markup for profit margin.
    """

    input_price_per_1m, cached_input_price_per_1m, output_price_per_1m = get_model_pricing(
        model
    )

    input_cost = (prompt_tokens / 1_000_000) * input_price_per_1m
    output_cost = (completion_tokens / 1_000_000) * output_price_per_1m

    if cached_tokens > 0:
        cached_price = (
            cached_input_price_per_1m
            if cached_input_price_per_1m is not None
            else input_price_per_1m
        )
        cached_cost = (cached_tokens / 1_000_000) * cached_price
    else:
        cached_cost = 0.0

    total_cost = input_cost + cached_cost + output_cost

    logger.debug(
        "Cost calculation for %s: %d input ($%.6f) + %d cached ($%.6f) + %d output ($%.6f) = $%.6f",
        model,
        prompt_tokens,
        input_cost,
        cached_tokens,
        cached_cost,
        completion_tokens,
        output_cost,
        total_cost,
    )

    return total_cost

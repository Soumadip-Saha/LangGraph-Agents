from enum import StrEnum, auto
from typing import TypeAlias

class Provider(StrEnum):
    OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    LLAMA = auto()
    
class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""
    
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_CHAT = "deepseek-chat"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_3 = "claude-3-haiku"
    HAIKU_35 = "claude-3.5-haiku"
    SONNET_35 = "claude-3.5-sonnet"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_20_FLASH = "gemini-2.0-flash"

class LLAMAModelName(StrEnum):
    """https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"""
    
    LLAMA_32 = "llama-3.2"
    
AllModelEnum: TypeAlias = (
    OpenAIModelName
    | DeepseekModelName
    | AnthropicModelName
    | GoogleModelName
    | LLAMAModelName
)
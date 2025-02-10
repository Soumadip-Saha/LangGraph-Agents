from functools import cache
import os
from typing import TypeAlias
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0, seed=1
)


from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    DeepseekModelName,
    GoogleModelName,
    OpenAIModelName,
    LLAMAModelName
)

_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    DeepseekModelName.DEEPSEEK_CHAT: "deepseek-chat",
    AnthropicModelName.HAIKU_3: "claude-3-haiku",
    AnthropicModelName.HAIKU_35: "claude-3.5-haiku",
    AnthropicModelName.SONNET_35: "claude-3.5-sonnet",
    GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",
    GoogleModelName.GEMINI_20_FLASH: "gemini-2.0-flash",
    LLAMAModelName.LLAMA_32: "llama-3.2"
}

ModelT: TypeAlias = ChatOpenAI      # TODO: Add other chat models

@cache
def get_model(model_name: AllModelEnum) -> ModelT:
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0, seed=1, streaming=True)
    elif model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0, streaming=True)
    elif model_name in LLAMAModelName:
        raise NotImplementedError(f"Model {model_name} is not supported yet")
    else:
        # TODO: Add other chat models [Deepseek, Anthropic, Google]
        raise NotImplementedError(f"Model {model_name} is not supported yet")
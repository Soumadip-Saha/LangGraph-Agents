


from typing import Any
from dotenv import find_dotenv
from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import OpenAIModelName, Provider, GoogleModelName
from core.llm import AllModelEnum


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding='utf-8',
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False
    )
    MODE: str | None = None
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    AUTH_SECRET: SecretStr | None = None
    
    OPENAI_API_KEY: SecretStr | None = None
    
    GOOGLE_API_KEY: SecretStr | None = None
    
    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None
    AVAILABLE_MODELS: set[AllModelEnum] = set()
    
    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("No API keys found")
        
        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case _:
                    raise ValueError(f"Unsupported provider: {provider}")
    
    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"
    
    def is_dev(self) -> bool:
        return self.MODE == "dev"

settings = Settings()
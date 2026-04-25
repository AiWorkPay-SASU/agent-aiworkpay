from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(..., description="Anthropic API key")
    aiworkpay_api_key: str = Field("", description="AIWorkPay API key")
    aiworkpay_base_url: str = Field(
        "https://api.aiworkpay.com/v1",
        description="AIWorkPay REST base URL",
    )
    confidence_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Min AI confidence before delegating task to a human",
    )
    default_task_budget: float = Field(
        0.10,
        gt=0.0,
        description="Default budget (USD) per human-delegated task",
    )
    max_autonomous_loops: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum autonomous self-improvement iterations",
    )
    claude_model: str = Field(
        "claude-opus-4-7",
        description="Claude model identifier",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

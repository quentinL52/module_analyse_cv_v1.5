import os
import litellm
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Configuration globale LiteLLM pour éviter les 429 Mistral
litellm.num_retries = 4
litellm.retry_policy = "exponential_backoff"
litellm.suppress_debug_info = True

# Force l'écrasement des variables systèmes par le .env
load_dotenv(override=True)

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "AIRH module analyse de CV -API"
    VERSION: str = "1.5"

    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    VISION_LLM_PROVIDER: str = os.getenv("VISION_LLM_PROVIDER", "openai")
    VISION_LLM_MODEL: str = os.getenv("VISION_LLM_MODEL", "gpt-4o")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")

    # Security
    INTERNAL_API_KEY: str | None = os.getenv("INTERNAL_API_KEY")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Observability
    LANGTRACE_API_KEY: str = os.getenv("LANGTRACE_API_KEY")

    # Thresholds & Limits
    LAYER1_TIMEOUT_SECONDS: int = 60
    LAYER2_TIMEOUT_SECONDS: int = 2
    LAYER3_TIMEOUT_SECONDS: int = 180
    MAX_FILE_SIZE_MB: int = 5

    def validate_keys(self):
        # Validation basique des clés selon les providers choisis
        if not self.INTERNAL_API_KEY or len(self.INTERNAL_API_KEY) < 16:
            raise ValueError("INTERNAL_API_KEY est obligatoire et doit faire au moins 16 caractères. Générer avec: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        if self.DEFAULT_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when DEFAULT_LLM_PROVIDER is openai")
        if self.DEFAULT_LLM_PROVIDER == "mistral" and not self.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is required when DEFAULT_LLM_PROVIDER is mistral")
        if self.VISION_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when VISION_LLM_PROVIDER is openai")
        if self.VISION_LLM_PROVIDER == "mistral" and not self.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is required when VISION_LLM_PROVIDER is mistral")
        return self

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
settings.validate_keys()

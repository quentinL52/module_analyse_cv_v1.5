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
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER")  
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")

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

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()

import instructor
import litellm
from openai import OpenAI
from mistralai import Mistral
from src.core.config import settings

# Configuration globale LiteLLM pour la robustesse (Rate Limits)
litellm.num_retries = 4
litellm.retry_policy = "exponential_backoff"
litellm.suppress_logs = True


def _build_instructor_client(provider: str):
    """Construit un client Instructor pour le provider demandé."""
    provider = provider.lower()

    if provider == "openai":
        return instructor.from_openai(OpenAI(api_key=settings.OPENAI_API_KEY))

    elif provider == "mistral":
        return instructor.from_mistral(Mistral(api_key=settings.MISTRAL_API_KEY))

    else:
        raise ValueError(f"Fournisseur LLM non supporté : {provider}")


def get_instructor_client():
    """
    Retourne un client Instructor correctement patché selon le fournisseur LLM défini dans la configuration.
    """
    return _build_instructor_client(settings.DEFAULT_LLM_PROVIDER)


def get_default_model_name() -> str:
    """
    Retourne le nom du modèle configuré pour éviter de l'écrire en dur dans le code métier.
    """
    return settings.DEFAULT_LLM_MODEL


def get_layer1_client_and_model():
    """
    Retourne un tuple (client, model_name) dédié à la Couche 1 (extraction structurée).
    Permet d'utiliser un petit modèle économique, indépendant de celui des agents.
    """
    return _build_instructor_client(settings.LAYER1_LLM_PROVIDER), settings.LAYER1_LLM_MODEL


def get_vision_client_and_model():
    """
    Retourne un tuple (client, model_name) pour le provider Vision configuré.
    """
    return _build_instructor_client(settings.VISION_LLM_PROVIDER), settings.VISION_LLM_MODEL

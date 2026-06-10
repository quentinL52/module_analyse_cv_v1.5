import instructor
import litellm
from openai import OpenAI
from mistralai import Mistral
from src.core.config import settings

# Configuration globale LiteLLM pour la robustesse (Rate Limits)
litellm.num_retries = 4
litellm.retry_policy = "exponential_backoff"
litellm.suppress_logs = True

def get_instructor_client():
    """
    Retourne un client Instructor correctement patché selon le fournisseur LLM défini dans la configuration.
    """
    provider = settings.DEFAULT_LLM_PROVIDER.lower()
    
    if provider == "openai":
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return instructor.from_openai(client)
        
    elif provider == "mistral":
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        return instructor.from_mistral(client)
        
    else:
        raise ValueError(f"Fournisseur LLM non supporté : {provider}")

def get_default_model_name() -> str:
    """
    Retourne le nom du modèle configuré pour éviter de l'écrire en dur dans le code métier.
    """
    return settings.DEFAULT_LLM_MODEL

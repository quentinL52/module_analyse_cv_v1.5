import threading
import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Compteur global thread-safe de tokens consommés.
    Le pipeline prend un snapshot avant/après pour mesurer le delta d'une analyse.
    Note : sous forte concurrence (plusieurs CV en parallèle dans le même worker),
    l'attribution par analyse est approximative mais le total reste exact.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self._by_model: dict[str, list[int]] = {}

    def add(self, prompt: int, completion: int, model: str = None):
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.calls += 1
            key = model or "unknown"
            if key not in self._by_model:
                self._by_model[key] = [0, 0]
            self._by_model[key][0] += prompt
            self._by_model[key][1] += completion

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return (self.prompt_tokens, self.completion_tokens, self.calls)

    def snapshot_by_model(self) -> dict[str, tuple[int, int]]:
        """Copie de l'usage cumulé par modèle : {model: (prompt, completion)}."""
        with self._lock:
            return {m: (v[0], v[1]) for m, v in self._by_model.items()}


token_counter = TokenCounter()


def track_completion_usage(completion) -> None:
    """Extrait l'usage d'une réponse LLM (objet OpenAI/Mistral/LiteLLM ou dict) et l'accumule."""
    if completion is None:
        return
    usage = getattr(completion, "usage", None)
    if usage is None and isinstance(completion, dict):
        usage = completion.get("usage")
    if usage is None:
        return
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens") or 0
        completion_t = usage.get("completion_tokens") or 0
    else:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion_t = getattr(usage, "completion_tokens", 0) or 0
    model = getattr(completion, "model", None)
    if model is None and isinstance(completion, dict):
        model = completion.get("model")
    token_counter.add(int(prompt), int(completion_t), model=model)


def litellm_success_callback(kwargs, completion_response, start_time, end_time):
    """Callback LiteLLM (CrewAI) : accumule l'usage de chaque appel, y compris les
    appels de conversion Pydantic internes à CrewAI, invisibles autrement."""
    try:
        track_completion_usage(completion_response)
    except Exception as e:
        logger.debug(f"Token tracking callback error (non bloquant): {e}")

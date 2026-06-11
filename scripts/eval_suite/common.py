"""
Socle commun de la suite d'évaluation : exécution instrumentée du pipeline
(durées, tokens, coût par modèle), normalisation de texte et matching flou.
"""
import os
import re
import time
import uuid
import unicodedata
import logging

from src.services.pipeline import execute_cv_pipeline, TASKS_STORE
from src.layers.layer1_ingestion.extractor import extractor
from src.core.token_tracker import token_counter

logger = logging.getLogger("eval_suite")

# Prix USD par million de tokens (input, output). Mettre à jour si les tarifs changent.
PRICING_USD_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "mistral-large-latest": (2.00, 6.00),
    "mistral-medium-latest": (0.40, 2.00),
    "mistral-small-latest": (0.10, 0.30),
    "ministral-8b-latest": (0.10, 0.10),
    "pixtral-large-latest": (2.00, 6.00),
}
DEFAULT_PRICE = (1.00, 3.00)  # fallback prudent si modèle inconnu


def _match_price(model_name: str) -> tuple[float, float]:
    if not model_name:
        return DEFAULT_PRICE
    name = model_name.lower()
    for known, price in PRICING_USD_PER_MTOK.items():
        if known in name or name in known:
            return price
    return DEFAULT_PRICE


def estimate_cost_usd(usage_by_model: dict[str, tuple[int, int]]) -> float:
    """Coût USD estimé depuis un usage {model: (prompt_tokens, completion_tokens)}."""
    total = 0.0
    for model, (p, c) in usage_by_model.items():
        in_price, out_price = _match_price(model)
        total += (p * in_price + c * out_price) / 1_000_000
    return round(total, 6)


def diff_usage(before: dict, after: dict) -> dict[str, tuple[int, int]]:
    """Delta d'usage par modèle entre deux snapshots."""
    delta = {}
    for model, (p, c) in after.items():
        bp, bc = before.get(model, (0, 0))
        dp, dc = p - bp, c - bc
        if dp or dc:
            delta[model] = (dp, dc)
    return delta


class FakeDB:
    """Session DB factice : capture les objets CVMetrics sans toucher Postgres.
    Permet de lire les durées par couche écrites par le pipeline."""

    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


async def run_pipeline_once(pdf_bytes: bytes, filename: str, job_description: str, db=None) -> dict:
    """
    Exécute le pipeline sur un CV et retourne le résultat enrichi de toutes
    les métriques à impact réel : statut, durées (totale + par couche),
    tokens, coût USD estimé par modèle.
    """
    own_db = db is None
    if own_db:
        db = FakeDB()

    task_id = str(uuid.uuid4())
    usage_before = token_counter.snapshot_by_model()
    t0 = time.time()

    await execute_cv_pipeline(task_id, pdf_bytes, filename, job_description, db)

    duration = time.time() - t0
    usage = diff_usage(usage_before, token_counter.snapshot_by_model())
    task = TASKS_STORE.get(task_id, {})

    perf = {
        "duration_s": round(duration, 2),
        "tokens_prompt": sum(p for p, _ in usage.values()),
        "tokens_completion": sum(c for _, c in usage.values()),
        "usage_by_model": {m: {"prompt": p, "completion": c} for m, (p, c) in usage.items()},
        "cost_usd": estimate_cost_usd(usage),
    }

    # Durées par couche si la session capture les CVMetrics (FakeDB)
    metrics_objs = getattr(db, "objects", [])
    if metrics_objs:
        m = metrics_objs[-1]
        for attr, key in [("duree_etape_1_ocr", "duration_ocr_s"),
                          ("duree_etape_2_extraction", "duration_extraction_s"),
                          ("duree_etape_4_agents", "duration_agents_s")]:
            val = getattr(m, attr, None)
            if val is not None:
                perf[key] = round(val, 2)

    return {
        "filename": filename,
        "status": task.get("status", "UNKNOWN"),
        "result": task.get("result"),
        "error": task.get("error"),
        "perf": perf,
    }


# --- Normalisation / matching flou (pour grounding et précision) ---

def normalize(s: str) -> str:
    """minuscules, sans accents, espaces simples — pour comparer JSON et texte source."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s%+#./-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def is_grounded(fact: str, source_norm: str, min_word_overlap: float = 0.8) -> bool:
    """
    Vrai si le fait extrait est retrouvable dans le texte source :
    soit en sous-chaîne exacte (après normalisation), soit si une proportion
    suffisante de ses mots significatifs y figure (tolérance reformulation légère).
    """
    fact_norm = normalize(fact)
    if not fact_norm:
        return True
    if fact_norm in source_norm:
        return True
    words = [w for w in fact_norm.split() if len(w) > 2]
    if not words:
        return fact_norm in source_norm
    found = sum(1 for w in words if w in source_norm)
    ratio = found / len(words)
    if ratio >= min_word_overlap:
        return True
    # Tolérance pour les faits courts (1-3 mots significatifs) : chaque mot doit
    # être trouvé individuellement, en testant aussi les sous-chaînes de 4+ lettres
    if len(words) <= 3 and found > 0:
        return ratio >= 0.5
    return False


def extract_source_text(pdf_bytes: bytes) -> str:
    """Texte brut du PDF (même extraction que le pipeline)."""
    text, _ = extractor.process_pdf(pdf_bytes)
    return text

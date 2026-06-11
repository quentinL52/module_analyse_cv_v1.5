"""
Évaluation LLM-as-a-Judge : compare le résultat du pipeline au CV réel.

Deux juges distincts :
1. Juge EXTRACTION : fidélité du JSON `candidat` au texte brut (exhaustivité
   par section + hallucinations citées précisément).
2. Juge ANALYSES : qualité des livrables agents (pertinence des avis, scores
   fondés sur des preuves, conseils actionnables, adéquation vision produit).

Le coût des juges est mesuré séparément du coût pipeline.
"""
import json
import asyncio
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from src.core.llm_factory import get_instructor_client, get_default_model_name
from src.core.token_tracker import token_counter, track_completion_usage
from scripts.eval_suite.common import diff_usage, estimate_cost_usd

logger = logging.getLogger("eval_suite.judge")

# Vision produit injectée dans le juge ANALYSES — à adapter si la vision évolue.
VISION_PRODUIT = """
L'outil est un coach CV pour candidats tech et un assistant de pré-qualification pour recruteurs :
- le feedback s'adresse DIRECTEMENT au candidat, en tutoiement, avec bienveillance mais sans complaisance ;
- chaque jugement et chaque score doit être FONDÉ sur des éléments présents dans le CV (citations), jamais inventé ;
- les conseils doivent être concrets et actionnables (le candidat sait quoi modifier après lecture) ;
- le recruteur reçoit une synthèse factuelle et des questions d'entretien ciblées et exploitables ;
- aucune discrimination ni jugement sur des éléments personnels (âge, origine, photo...).
"""

JUDGE_SEUIL_FIDELITE = 7
JUDGE_SEUIL_ANALYSES = 7


class HallucinationItem(BaseModel):
    champ: str = Field(description="Champ du JSON concerné (ex: experiences[0].entreprise)")
    valeur_json: str = Field(description="Valeur présente dans le JSON")
    justification: str = Field(description="Pourquoi cette valeur est introuvable ou contredite par le texte brut")


class SectionScore(BaseModel):
    section: str = Field(description="Nom de la section (identite, skills, experiences, projets, formations, langues_certifs)")
    exhaustivite: int = Field(ge=0, le=10, description="10 = rien d'important du texte brut ne manque dans cette section")
    fidelite: int = Field(ge=0, le=10, description="10 = aucune valeur inventée, déformée ou traduite")
    elements_manquants: List[str] = Field(default_factory=list, description="Éléments du texte brut absents du JSON")


class JudgeExtraction(BaseModel):
    sections: List[SectionScore] = Field(description="Évaluation par section")
    hallucinations: List[HallucinationItem] = Field(default_factory=list, description="Informations du JSON introuvables dans le texte brut")
    langue_preservee: bool = Field(description="Vrai si le texte extrait est resté dans la langue originale du CV (pas de traduction)")
    score_fidelite_global: int = Field(ge=0, le=10, description="Note globale de fidélité de l'extraction")
    remarques: str = Field(description="Synthèse courte des forces et faiblesses de l'extraction")


class JugementNonFonde(BaseModel):
    livrable: str = Field(description="Livrable concerné (avis_roni, avis_cto, tips, contexte_recruteur, score...)")
    probleme: str = Field(description="En quoi le jugement n'est pas fondé sur le contenu réel du CV")


class JudgeAnalyses(BaseModel):
    pertinence_avis: int = Field(ge=0, le=10, description="Les avis (Roni, CTO) correspondent-ils au contenu réel du CV ?")
    scores_fondes_sur_preuves: int = Field(ge=0, le=10, description="Les notes sont-elles justifiées par des citations/éléments réels du CV ?")
    actionnabilite_conseils: int = Field(ge=0, le=10, description="Les tips permettent-ils au candidat de savoir précisément quoi améliorer ?")
    qualite_livrables_recruteur: int = Field(ge=0, le=10, description="Synthèse recruteur et axes d'entretien : factuels, ciblés, exploitables ?")
    tutoiement_respecte: bool = Field(description="Vrai si tout le feedback candidat est en tutoiement")
    jugements_non_fondes: List[JugementNonFonde] = Field(default_factory=list)
    adequation_vision_produit: int = Field(ge=0, le=10, description="Note globale d'adéquation avec la vision produit fournie")
    remarques: str = Field(description="Synthèse courte")


def _judge_call(messages, response_model):
    client = get_instructor_client()
    result, completion = client.chat.completions.create_with_completion(
        model=get_default_model_name(),
        response_model=response_model,
        messages=messages,
        temperature=0.0,
        max_retries=1,
    )
    track_completion_usage(completion)
    return result


async def judge_extraction(texte_brut: str, candidat: dict) -> dict:
    prompt = f"""Tu es un évaluateur expert d'extraction de données (LLM-as-a-Judge), neutre et rigoureux.
Compare le TEXTE BRUT d'un CV avec le JSON structuré produit par un pipeline d'extraction.

TEXTE BRUT DU CV :
---
{texte_brut}
---

JSON EXTRAIT (section candidat) :
---
{json.dumps(candidat, ensure_ascii=False)}
---

Consignes :
1. Pour chaque section (identite, skills, experiences, projets, formations, langues_certifs), note l'exhaustivité (tout y est ?) et la fidélité (rien d'inventé ?).
2. Liste PRÉCISÉMENT chaque hallucination : valeur du JSON introuvable ou contredite par le texte brut. Ignore les différences de casse, d'accents ou les artefacts OCR.
3. Vérifie que la langue originale du CV est préservée (pas de traduction).
4. Ne pénalise PAS les champs d'analyse (statut_temporel, evaluation_roni, avis...) : seuls les faits extraits comptent."""
    return (await asyncio.to_thread(_judge_call,
        [{"role": "system", "content": "Tu es un expert neutre et rigoureux en qualité de données."},
         {"role": "user", "content": prompt}],
        JudgeExtraction)).model_dump()


async def judge_analyses(texte_brut: str, result: dict) -> dict:
    recos = result.get("recommandations_agentiques", {})
    candidat = result.get("candidat", {})
    avis_experiences = {e.get("poste"): e.get("evaluation_roni") for e in candidat.get("experiences", []) if e.get("evaluation_roni")}
    avis_projets = [{"title": p.get("title"), "avis_cto": p.get("avis_cto"),
                     "axes": {k: p.get(k) for k in ("axe_impact_metier", "axe_complexite_tech", "axe_alignement_strat")}}
                    for p in candidat.get("projets") or []]

    prompt = f"""Tu es un évaluateur expert de systèmes IA d'analyse de CV (LLM-as-a-Judge), neutre et rigoureux.

VISION PRODUIT DE L'OUTIL :
{VISION_PRODUIT}

TEXTE BRUT DU CV (la seule source de vérité) :
---
{texte_brut}
---

LIVRABLES PRODUITS PAR LES AGENTS :
---
{json.dumps({
    "avis_experiences": avis_experiences,
    "avis_projets": avis_projets,
    "qualite_cv": recos.get("qualite_cv"),
    "contexte_recruteur": recos.get("contexte_recruteur"),
    "axes_entretien_simulateur": recos.get("axes_entretien_simulateur"),
    "analyse_matching_offre": recos.get("analyse_matching_offre"),
}, ensure_ascii=False)}
---

Évalue ces livrables par rapport au CV réel et à la vision produit :
1. Les avis correspondent-ils au contenu réel du CV (pas de généralités plaquées) ?
2. Les scores sont-ils justifiés par des preuves/citations du CV ?
3. Les conseils sont-ils actionnables ?
4. Les livrables recruteur sont-ils factuels et exploitables ?
5. Liste tout jugement non fondé sur le CV réel."""
    return (await asyncio.to_thread(_judge_call,
        [{"role": "system", "content": "Tu es un expert neutre et rigoureux en évaluation de systèmes IA."},
         {"role": "user", "content": prompt}],
        JudgeAnalyses)).model_dump()


async def run_judges(texte_brut: str, result: dict) -> dict:
    """Exécute les deux juges et mesure leur coût séparément."""
    usage_before = token_counter.snapshot_by_model()

    extraction_eval, analyses_eval = {}, {}
    try:
        extraction_eval = await judge_extraction(texte_brut, result.get("candidat", {}))
    except Exception as e:
        logger.error(f"Juge extraction en échec: {e}")
        extraction_eval = {"error": str(e)}
    try:
        analyses_eval = await judge_analyses(texte_brut, result)
    except Exception as e:
        logger.error(f"Juge analyses en échec: {e}")
        analyses_eval = {"error": str(e)}

    usage = diff_usage(usage_before, token_counter.snapshot_by_model())

    fidelite = extraction_eval.get("score_fidelite_global")
    adequation = analyses_eval.get("adequation_vision_produit")
    verdict = "PASS"
    if (fidelite is not None and fidelite < JUDGE_SEUIL_FIDELITE) or \
       (adequation is not None and adequation < JUDGE_SEUIL_ANALYSES) or \
       extraction_eval.get("error") or analyses_eval.get("error"):
        verdict = "FAIL"

    return {
        "extraction": extraction_eval,
        "analyses": analyses_eval,
        "cout_juge_usd": estimate_cost_usd(usage),
        "verdict": verdict,
        "seuils": {"fidelite_min": JUDGE_SEUIL_FIDELITE, "analyses_min": JUDGE_SEUIL_ANALYSES},
    }

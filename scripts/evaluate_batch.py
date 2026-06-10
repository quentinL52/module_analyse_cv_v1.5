import os
import sys
import asyncio
import json
import logging
import uuid
from typing import List

# Ajouter la racine du projet IA/module_analyse_cv_v1.5 au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field
from src.core.db import SessionLocal
from src.services.pipeline import execute_cv_pipeline, TASKS_STORE
from src.layers.layer1_ingestion.extractor import extractor
from src.core.llm_factory import get_instructor_client, get_default_model_name

# Configuration Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_batch")

# Schéma d'évaluation
class EvaluationLLMJudge(BaseModel):
    exhaustivite_score: int = Field(..., ge=0, le=10, description="Score sur 10 mesurant si tous les éléments clés du CV (expériences, formations, compétences) sont bien présents dans le JSON.")
    hallucinations_detectees: bool = Field(..., description="Vrai si des informations présentes dans le JSON ont été inventées (non trouvées dans le texte brut).")
    details_hallucinations: List[str] = Field(default_factory=list, description="Liste des informations inventées.")
    biais_detectes: bool = Field(..., description="Vrai si un biais d'agent ou une erreur manifeste de jugement est détecté dans les notes de qualité ou recommandations.")
    details_biais: List[str] = Field(default_factory=list, description="Liste des biais détectés.")
    qualite_jugement_score: int = Field(..., ge=0, le=10, description="Score sur 10 de la pertinence et cohérence des notes (Roni, Profileur, etc.) vis-à-vis du CV.")
    remarques_generales: str = Field(..., description="Commentaire global sur la qualité de l'extraction.")

async def evaluate_cv_with_llm(texte_brut: str, result_json: dict) -> dict:
    """
    Fait appel à un LLM avec Instructor pour évaluer le JSON final en le comparant au texte brut.
    """
    client = get_instructor_client()
    model_name = get_default_model_name()
    
    prompt = f"""
Tu es un évaluateur expert d'extraction de données (LLM-as-a-Judge).
Ton rôle est de comparer un texte brut extrait d'un CV (OCR) avec le JSON structuré produit par un pipeline IA.

Voici le TEXTE BRUT DU CV :
---
{texte_brut}
---

Voici le JSON PRODUIT PAR LE PIPELINE :
---
{json.dumps(result_json, indent=2, ensure_ascii=False)}
---

Tâche :
1. Vérifie que toutes les expériences, formations et compétences importantes du CV sont dans le JSON (Exhaustivité).
2. Vérifie scrupuleusement qu'aucune information dans le JSON n'a été inventée ou extrapolée par rapport au texte brut (Hallucinations).
3. Vérifie si les agents (profileur, roni, recruteur) ont émis des jugements biaisés ou disproportionnés (Biais).
4. Évalue la qualité du jugement global.
    """
    
    try:
        # Appel Instructor
        eval_result = client.chat.completions.create(
            model=model_name,
            response_model=EvaluationLLMJudge,
            messages=[
                {"role": "system", "content": "Tu es un expert neutre et rigoureux en qualité de données."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return eval_result.model_dump()
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation LLM : {e}")
        return {"error": str(e)}

async def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Ressources/base de CV tests"))
    if not os.path.exists(base_dir):
        logger.error(f"Dossier de CV introuvable : {base_dir}")
        return

    cv_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".pdf")]
    logger.info(f"{len(cv_files)} CV(s) trouvé(s) pour l'évaluation.")
    
    job_description = "Ingénieur IA / Data Scientist avec de solides compétences en machine learning, NLP et Python. Bonne expérience de déploiement (MLOps)."
    
    report = {
        "job_description_utilisee": job_description,
        "evaluations": []
    }
    
    db = SessionLocal()
    
    try:
        for idx, filename in enumerate(cv_files, 1):
            filepath = os.path.join(base_dir, filename)
            logger.info(f"--- Traitement du CV {idx}/{len(cv_files)} : {filename} ---")
            
            with open(filepath, "rb") as f:
                pdf_bytes = f.read()
                
            task_id = str(uuid.uuid4())
            
            # Extraction du texte brut pour l'évaluateur
            texte_brut, _ = extractor.process_pdf(pdf_bytes)
            
            # 1. Exécution du pipeline
            # Ceci ajoute les métriques d'extraction à la table cv_metrics de Postgres
            await execute_cv_pipeline(task_id, pdf_bytes, filename, job_description, db)
            
            task_status = TASKS_STORE.get(task_id, {})
            if task_status.get("status") == "COMPLETED":
                result_json = task_status.get("result", {})
                
                # 2. Évaluation LLM (Judge)
                logger.info(f"Évaluation LLM-as-a-Judge en cours pour {filename}...")
                eval_data = await evaluate_cv_with_llm(texte_brut, result_json)
                
                report["evaluations"].append({
                    "filename": filename,
                    "pipeline_status": "COMPLETED",
                    "evaluation_llm": eval_data
                })
                logger.info(f"Résultat de l'évaluation pour {filename} : {eval_data.get('exhaustivite_score')}/10 exhaustivité, Hallucinations: {eval_data.get('hallucinations_detectees')}")
            else:
                logger.error(f"Échec du pipeline pour {filename} : {task_status.get('error')}")
                report["evaluations"].append({
                    "filename": filename,
                    "pipeline_status": "FAILED",
                    "error": task_status.get("error")
                })
                
    finally:
        db.close()
        
    # Sauvegarde du rapport JSON
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Rapport d'évaluation sauvegardé dans : {report_path}")

if __name__ == "__main__":
    asyncio.run(main())

import os
import sys
import asyncio
import json
import logging
import uuid
import numpy as np
from collections import defaultdict
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.db import SessionLocal
from src.services.pipeline import execute_cv_pipeline, TASKS_STORE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("measure_variance")

NUM_RUNS = 5

async def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Ressources/base de CV tests"))
    if not os.path.exists(base_dir):
        logger.error(f"Dossier de CV introuvable : {base_dir}")
        return

    cv_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".pdf")]
    if not cv_files:
        logger.error("Aucun CV trouvé pour mesurer la variance.")
        return

    logger.info(f"{len(cv_files)} CV(s) trouvé(s). Début de {NUM_RUNS} itérations de test pour mesurer la variance.")
    
    job_description = "Ingénieur IA / Data Scientist avec de solides compétences en machine learning."
    db = SessionLocal()
    
    # Structure pour stocker les scores:
    # scores_data[filename]["production_readiness"] = [score1, score2, ...]
    # scores_data[filename]["score_global"] = [score1, score2, ...]
    scores_data = defaultdict(lambda: defaultdict(list))
    
    try:
        for run_idx in range(1, NUM_RUNS + 1):
            logger.info(f"==== RUN {run_idx}/{NUM_RUNS} ====")
            for filename in cv_files:
                filepath = os.path.join(base_dir, filename)
                
                with open(filepath, "rb") as f:
                    pdf_bytes = f.read()
                    
                task_id = str(uuid.uuid4())
                try:
                    await execute_cv_pipeline(task_id, pdf_bytes, filename, job_description, db)
                    
                    task_status = TASKS_STORE.get(task_id, {})
                    if task_status.get("status") == "COMPLETED":
                        result = task_status.get("result", {})
                        
                        # Extraire les scores
                        recos = result.get("recommandations_agentiques", {})
                        prod_score = recos.get("production_readiness_score")
                        
                        qualite_cv = recos.get("qualite_cv", {})
                        score_multi = qualite_cv.get("score_multidimensionnel", {})
                        score_global = score_multi.get("score_global")
                        
                        if prod_score is not None:
                            scores_data[filename]["production_readiness"].append(prod_score)
                        if score_global is not None:
                            scores_data[filename]["score_global"].append(score_global)

                        # Extract is_tech for experiences
                        candidat = result.get("candidat", {})
                        for exp in candidat.get("experiences", []):
                            poste = exp.get("poste", "Inconnu")
                            eval_roni = exp.get("evaluation_roni")
                            if eval_roni is not None and "is_tech" in eval_roni:
                                scores_data[filename][f"is_tech::{poste}"].append(eval_roni["is_tech"])
                                
                        # Extract axes for projets
                        for projet in candidat.get("projets", []) or []:
                            title = projet.get("title", "Inconnu")
                            axe_impact = projet.get("axe_impact_metier")
                            if axe_impact is not None and "score" in axe_impact:
                                scores_data[filename][f"axe_impact::{title}"].append(axe_impact["score"])
                            axe_complexite = projet.get("axe_complexite_tech")
                            if axe_complexite is not None and "score" in axe_complexite:
                                scores_data[filename][f"axe_complexite::{title}"].append(axe_complexite["score"])
                            axe_alignement = projet.get("axe_alignement_strat")
                            if axe_alignement is not None and "score" in axe_alignement:
                                scores_data[filename][f"axe_alignement::{title}"].append(axe_alignement["score"])
                    else:
                        logger.error(f"Pipeline échoué pour {filename} au run {run_idx}.")
                        
                except Exception as e:
                    logger.error(f"Erreur d'exécution pipeline pour {filename} (Run {run_idx}): {e}")
                    
    finally:
        db.close()
        
    logger.info("==== CALCUL DE LA VARIANCE ET ÉCART-TYPE ====")
    report = {}
    
    for filename, scores in scores_data.items():
        file_report = {}
        
        prod_scores = scores.get("production_readiness", [])
        if prod_scores:
            file_report["production_readiness"] = {
                "valeurs": prod_scores,
                "moyenne": round(float(np.mean(prod_scores)), 2),
                "ecart_type": round(float(np.std(prod_scores)), 2),
                "variance": round(float(np.var(prod_scores)), 2)
            }
            
        global_scores = scores.get("score_global", [])
        if global_scores:
            file_report["score_global"] = {
                "valeurs": global_scores,
                "moyenne": round(float(np.mean(global_scores)), 2),
                "ecart_type": round(float(np.std(global_scores)), 2),
                "variance": round(float(np.var(global_scores)), 2)
            }
            
        is_tech_stable = True
        std_axes_pass = True
        std_global_pass = (not global_scores) or (np.std(global_scores) <= 4.0)
        failed_thresholds = []

        if global_scores and np.std(global_scores) > 4.0:
            failed_thresholds.append(f"score_global std > 4.0 ({round(float(np.std(global_scores)), 2)})")
            
        for key, vals in scores.items():
            if key.startswith("is_tech::"):
                if len(set(vals)) == 1 and len(vals) == NUM_RUNS:
                    file_report[key] = {"status": "STABLE", "valeurs": vals}
                else:
                    file_report[key] = {"status": "INSTABLE", "valeurs": vals}
                    is_tech_stable = False
                    failed_thresholds.append(f"{key} INSTABLE")
            elif key.startswith("axe_"):
                std_val = round(float(np.std(vals)), 2)
                file_report[key] = {
                    "valeurs": vals,
                    "moyenne": round(float(np.mean(vals)), 2),
                    "ecart_type": std_val
                }
                if std_val > 1.0:
                    std_axes_pass = False
                    failed_thresholds.append(f"{key} std > 1.0 ({std_val})")
                    
        verdict = "PASS" if (is_tech_stable and std_axes_pass and std_global_pass) else "FAIL"
        file_report["verdict_global"] = {
            "resultat": verdict,
            "seuils_violes": failed_thresholds
        }
            
        report[filename] = file_report
        logger.info(f"Fichier: {filename}")
        if "production_readiness" in file_report:
            logger.info(f"  Production Readiness -> Ecart-type: {file_report['production_readiness']['ecart_type']}")
        if "score_global" in file_report:
            logger.info(f"  Score Global -> Ecart-type: {file_report['score_global']['ecart_type']}")

    report_path = os.path.join(os.path.dirname(__file__), "variance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Rapport de variance sauvegardé dans : {report_path}")

if __name__ == "__main__":
    asyncio.run(main())

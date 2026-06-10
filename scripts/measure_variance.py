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
                    else:
                        logger.error(f"Pipeline échoué pour {filename} au run {run_idx}.")
                        
                except Exception as e:
                    logger.error(f"Erreur d'exécution pipeline pour {filename} (Run {run_idx}): {e}")
                    
    finally:
        db.close()
        
    logger.info("==== CALCUL DE LA VARIANCE ET ÉCART-TYPE ====")
    report = {}
    
    for filename, scores in scores_data.items():
        prod_scores = scores.get("production_readiness", [])
        global_scores = scores.get("score_global", [])
        
        file_report = {}
        
        if prod_scores:
            mean_prod = np.mean(prod_scores)
            std_prod = np.std(prod_scores)
            var_prod = np.var(prod_scores)
            file_report["production_readiness"] = {
                "valeurs": prod_scores,
                "moyenne": round(float(mean_prod), 2),
                "ecart_type": round(float(std_prod), 2),
                "variance": round(float(var_prod), 2)
            }
            
        if global_scores:
            mean_glob = np.mean(global_scores)
            std_glob = np.std(global_scores)
            var_glob = np.var(global_scores)
            file_report["score_global"] = {
                "valeurs": global_scores,
                "moyenne": round(float(mean_glob), 2),
                "ecart_type": round(float(std_glob), 2),
                "variance": round(float(var_glob), 2)
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

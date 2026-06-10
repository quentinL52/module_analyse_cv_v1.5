import logging
import asyncio
from typing import Dict, Any

from src.layers.layer1_ingestion.extractor import extractor
from src.layers.layer1_ingestion.llm_parser import parse_cv_brute
from src.layers.layer2_graph.matcher import get_dominant_profile_async
from src.layers.layer3_agents.crew_orchestrator import run_agentic_eval_async
from src.schemas.cv_schema import CVParsedFinal, Candidat, RecommandationsAgentiques, QualiteCV, AnalyseMatchingOffre, Projet, EvaluationExperienceRoni, CompetenceTransferable
from src.services.scoring_cv import compute_multidimensional_score
from src.core.db import CVMetrics
from src.core.llm_factory import get_default_model_name
import time
from datetime import datetime
import re

logger = logging.getLogger(__name__)

TASKS_STORE = {}

def annotate_temporal_status(experiences, today):
    for exp in experiences:
        end = exp.end_date.strip().lower() if exp.end_date else ""
        if not end:
            exp.statut_temporel = "INDETERMINE"
            continue
        if end in ["present", "présent", "en cours", "aujourd'hui", "now"]:
            exp.statut_temporel = "EN_COURS"
            continue
        
        match = re.search(r'(\d{4})', end)
        if match:
            year = int(match.group(1))
            month_match = re.search(r'(\d{4})-(\d{2})', end)
            if month_match:
                month = int(month_match.group(2))
            else:
                month = 12
            
            if year > today.year or (year == today.year and month >= today.month):
                exp.statut_temporel = "EN_COURS"
            else:
                exp.statut_temporel = "TERMINEE"
        else:
            exp.statut_temporel = "INDETERMINE"
            
    return experiences


async def execute_cv_pipeline(task_id: str, pdf_bytes: bytes, filename: str, job_description: str, db):
    try:
        TASKS_STORE[task_id] = {"status": "PROCESSING"}
        
        # Phase 1: Ingestion
        logger.info(f"Task {task_id}: Extraction de texte depuis le PDF.")
        t0 = time.time()
        texte_cv, is_ocr = extractor.process_pdf(pdf_bytes)
        t_ocr = time.time() - t0
        
        # Couche 1: Parsing LLM structurel
        logger.info(f"Task {task_id}: Lancement de la Couche 1 (Parsing Brut).")
        t0 = time.time()
        brute_data = await parse_cv_brute(texte_cv, pdf_bytes)
        t_extraction = time.time() - t0
        
        today = datetime.now()
        brute_data.experiences = annotate_temporal_status(brute_data.experiences, today)
        
        # Couche 2: Graph Matching
        logger.info(f"Task {task_id}: Lancement de la Couche 2 (Matcher Graph).")
        flat_hard_skills = [s.skill for s in brute_data.skills.hard_skills]
        if hasattr(brute_data, "formations") and brute_data.formations:
            flat_hard_skills.extend([f.titre_diplome for f in brute_data.formations if f.titre_diplome])
            
        top_profiles = await get_dominant_profile_async(flat_hard_skills, top_k=3)
        
        # Couche 3: Évaluation Agentique
        logger.info(f"Task {task_id}: Lancement de la Couche 3 (Agents).")
        t0 = time.time()
        layer3_results = await run_agentic_eval_async(brute_data.model_dump(), job_description)
        t_agents = time.time() - t0
        
        # Mapping Final
        logger.info(f"Task {task_id}: Construction du résultat final.")
        
        # Extraction des sous-parties avec fallback
        if not isinstance(layer3_results, dict) or "error" in layer3_results:
            logger.warning(f"Task {task_id}: Mode dégradé ou échec de la couche 3.")
            # Dans un cas réel, on gèrerait une réponse partielle
            raise Exception("Couche 3 a retourné une erreur ou est incomplète.")
            
        profileur_data = layer3_results.get("profileur", {})
        if hasattr(profileur_data, "model_dump"):
            profileur_data = profileur_data.model_dump()
            
        transferables = profileur_data.get("transferables_par_skill", {})
        
        # Mapping Transferables sur les Soft Skills
        final_soft_skills = []
        for s in brute_data.skills.soft_skills:
            skill_name = s.skill
            if skill_name in transferables:
                t_data = transferables[skill_name]
                s.transferable = CompetenceTransferable(**t_data) if isinstance(t_data, dict) else t_data
            final_soft_skills.append(s)
            
        brute_data.skills.soft_skills = final_soft_skills
                    # Mapping Audits
        audit_projets = layer3_results.get("audit_projets", {})
        if hasattr(audit_projets, "model_dump"):
            audit_projets = audit_projets.model_dump()
            
        roni_data = layer3_results.get("roni", {})
        if hasattr(roni_data, "model_dump"):
            roni_data = roni_data.model_dump()
            
        stratege_data = layer3_results.get("stratege", {})
        if hasattr(stratege_data, "model_dump"):
            stratege_data = stratege_data.model_dump()
            
        ats_data = layer3_results.get("ats", {})
        if hasattr(ats_data, "model_dump"):
            ats_data = ats_data.model_dump()
            
        final_projets = []
        if brute_data.projets:
            evaluated_projets_dict = {p.get("title", ""): p for p in audit_projets.get("projets", [])}
            for pb in brute_data.projets:
                eval_dict = evaluated_projets_dict.get(pb.title, {})
                merged_dict = pb.model_dump()
                merged_dict.update({k: v for k, v in eval_dict.items() if k != "title"})
                final_projets.append(Projet(**merged_dict))
            
        for exp in brute_data.experiences:
            poste = exp.poste
            eval_roni = roni_data.get("evaluations_experiences", {}).get(poste)
            exp.evaluation_roni = EvaluationExperienceRoni(**eval_roni) if eval_roni else None
            
        candidat = Candidat(
            first_name=brute_data.first_name,
            poste_vise_header=brute_data.poste_vise_header,
            poste_vise_confidence=brute_data.poste_vise_confidence,
            introduction=brute_data.introduction,
            liens_externes=brute_data.liens_externes,
            grammaire_orthographe=brute_data.grammaire_orthographe,
            is_reconversion=profileur_data.get("is_reconversion"),
            is_etudiant=profileur_data.get("is_etudiant"),
            is_disponible=profileur_data.get("is_disponible"),
            skills=brute_data.skills,
            experiences=brute_data.experiences,
            projets=final_projets,
            formations=brute_data.formations,
            langues=brute_data.langues,
            certifications=brute_data.certifications
        )
        
        score_multi, prod_readiness_score = compute_multidimensional_score(candidat)
        
        qualite_cv_data = roni_data.get("qualite_cv")
        if qualite_cv_data:
            qualite_cv_obj = QualiteCV(**qualite_cv_data)
            qualite_cv_obj.score_multidimensionnel = score_multi
        else:
            qualite_cv_obj = None
        contexte_recruteur = stratege_data.get("contexte_recruteur", "")
        
        has_en_cours = any(exp.statut_temporel == "EN_COURS" for exp in brute_data.experiences)
        if not has_en_cours and contexte_recruteur:
            if re.search(r'\b(actuellement|aujourd\'hui|occupe|travaille|en poste)\b', contexte_recruteur, re.IGNORECASE):
                logger.warning(f"temporal_inconsistency Task {task_id}: Le contexte_recruteur semble utiliser le présent alors qu'aucune expérience n'est EN_COURS.")
        
        recos = RecommandationsAgentiques(
            profil_calcule_knowledge_graph=top_profiles,
            production_readiness_score=prod_readiness_score,
            analyse_matching_offre=AnalyseMatchingOffre(**ats_data) if ats_data else None,
            qualite_cv=qualite_cv_obj,
            contexte_recruteur=contexte_recruteur,
            axes_entretien_simulateur=stratege_data.get("axes_entretien_simulateur", [])
        )
        
        final_result = CVParsedFinal(
            candidat=candidat,
            recommandations_agentiques=recos
        )
        
        metrics = CVMetrics(
            filename=filename,
            model_used=f"{get_default_model_name()}+crewai",
            duree_etape_1_ocr=t_ocr,
            duree_etape_2_extraction=t_extraction,
            duree_etape_4_agents=t_agents,
            duree_totale=t_ocr + t_extraction + t_agents,
            fallback_ocr_triggered=is_ocr,
            statut_final="COMPLETED"
        )
        db.add(metrics)
        db.commit()
        
        TASKS_STORE[task_id] = {"status": "COMPLETED", "result": final_result.model_dump()}
        logger.info(f"Task {task_id}: Pipeline terminé avec succès.")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        db.add(CVMetrics(filename=filename, statut_final="FAILED"))
        db.commit()
        TASKS_STORE[task_id] = {"status": "FAILED", "error": str(e)}

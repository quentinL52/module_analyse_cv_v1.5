from src.schemas.cv_schema import Candidat, ScoreMultidimensionnel

from typing import Tuple

def compute_multidimensional_score(candidat: Candidat) -> Tuple[ScoreMultidimensionnel, int]:
    """
    Calcule le score multidimensionnel déterministe d'un CV.
    Retourne un objet ScoreMultidimensionnel avec des notes sur 100 et le production_readiness_score.
    """
    # ---------------------------------------------------------
    # 1. Dimension Aspect (Structure)
    # ---------------------------------------------------------
    score_aspect = 0.0
    aspect_criteria_count = 4
    
    # Titre poste visé
    if candidat.poste_vise_header:
        score_aspect += 100 / aspect_criteria_count
        
    # Introduction
    if candidat.introduction:
        score_aspect += 100 / aspect_criteria_count
    
    # Liens
    if candidat.liens_externes:
        has_valid_link = any(lien.lien_cliquable for lien in candidat.liens_externes)
        if has_valid_link:
            score_aspect += 100 / aspect_criteria_count
            
    # Projets
    if candidat.projets and len(candidat.projets) > 0:
        score_aspect += 100 / aspect_criteria_count
        
    score_aspect_final = min(100.0, score_aspect)

    # ---------------------------------------------------------
    # 2. Dimension Projets
    # ---------------------------------------------------------
    score_projets = 0.0
    if candidat.projets and len(candidat.projets) > 0:
        total_projets_score = 0.0
        for p in candidat.projets:
            # Le score du projet est calculé à partir des 3 axes
            if p.axe_impact_metier and p.axe_complexite_tech and p.axe_alignement_strat:
                p_score = (p.axe_impact_metier.score + p.axe_complexite_tech.score + p.axe_alignement_strat.score) / 3.0 * 10.0
                p.score_general_projet_pourcent = int(round(p_score))
            else:
                p_score = float(p.score_general_projet_pourcent) if p.score_general_projet_pourcent is not None else 50.0
            total_projets_score += p_score
            
        score_projets = total_projets_score / len(candidat.projets)

    # ---------------------------------------------------------
    # 3. Dimension Expériences
    # ---------------------------------------------------------
    score_experiences = 0.0
    if candidat.experiences and len(candidat.experiences) > 0:
        total_exp_score = 0.0
        for exp in candidat.experiences:
            e_score = 0.0
            roni_eval = exp.evaluation_roni
            
            if roni_eval:
                if roni_eval.is_tech:
                    # Expérience Tech (50% métriques, 50% cohérence tech)
                    if exp.metriques_identifiees and len(exp.metriques_identifiees) > 0:
                        e_score += 50.0
                    e_score += (roni_eval.note_coherence_tech or 0) * 5.0
                else:
                    # Expérience Non-Tech (100% soft skills)
                    e_score += (roni_eval.note_soft_skills_valeur or 0) * 10.0
            else:
                # Fallback si Roni n'a pas pu évaluer l'expérience
                if exp.metriques_identifiees and len(exp.metriques_identifiees) > 0:
                    e_score += 50.0
                # Points bonus basiques
                e_score += 25.0
                
            total_exp_score += min(100.0, e_score)
            
        score_experiences = total_exp_score / len(candidat.experiences)

    # ---------------------------------------------------------
    # Score Global
    # ---------------------------------------------------------
    score_global = round((score_aspect_final + score_projets + score_experiences) / 3.0, 2)
    
    # Calcul du production readiness score (Option A)
    bonus = 0.0
    if candidat.skills and (len(candidat.skills.hard_skills) > 5):
        bonus += 5.0
    production_readiness_score = int(round(0.4 * score_experiences + 0.4 * score_projets + 0.2 * score_aspect_final + bonus))
    production_readiness_score = min(100, max(0, production_readiness_score))
    
    score_multi = ScoreMultidimensionnel(
        score_aspect=round(score_aspect_final, 2),
        score_projets=round(score_projets, 2),
        score_experiences=round(score_experiences, 2),
        score_global=score_global
    )
    return score_multi, production_readiness_score

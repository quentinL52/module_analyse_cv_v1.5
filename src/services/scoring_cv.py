from src.schemas.cv_schema import Candidat, ScoreMultidimensionnel

def compute_multidimensional_score(candidat: Candidat) -> ScoreMultidimensionnel:
    """
    Calcule le score multidimensionnel déterministe d'un CV.
    Retourne un objet ScoreMultidimensionnel avec des notes sur 100.
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
            # Le score du projet est directement fourni par le LLM en pourcentage
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
    
    return ScoreMultidimensionnel(
        score_aspect=round(score_aspect_final, 2),
        score_projets=round(score_projets, 2),
        score_experiences=round(score_experiences, 2),
        score_global=score_global
    )

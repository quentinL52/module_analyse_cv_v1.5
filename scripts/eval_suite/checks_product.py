"""
Tests d'ADÉQUATION PRODUIT (déterministes) : le résultat final respecte-t-il
les engagements de l'outil tels que définis dans les prompts ?

- feedback adressé au candidat en tutoiement (jamais de vouvoiement) ;
- conseils conditionnels corrects (introduction / projets absents) ;
- livrables recruteur complets (contexte, axes d'entretien) ;
- scores dans leurs bornes ;
- cohérence temporelle de la synthèse.
"""
import re

TU_PATTERN = re.compile(r"\b(tu|ton|ta|tes|toi)\b", re.IGNORECASE)
VOUS_PATTERN = re.compile(r"\b(vous|votre|vos)\b", re.IGNORECASE)
PRESENT_PATTERN = re.compile(r"\b(actuellement|aujourd'hui|occupe|travaille|en poste)\b", re.IGNORECASE)

MIN_TIPS = 2
MIN_AXES_ENTRETIEN = 2


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def _collect_avis_textes(result: dict) -> list[str]:
    """Tous les textes adressés au candidat (avis, tips, critiques)."""
    textes = []
    candidat = result.get("candidat", {})
    recos = result.get("recommandations_agentiques", {})

    for exp in candidat.get("experiences", []):
        ev = exp.get("evaluation_roni")
        if ev and ev.get("avis_roni"):
            textes.append(ev["avis_roni"])
    for p in candidat.get("projets") or []:
        if p.get("avis_cto"):
            textes.append(p["avis_cto"])
    qualite = recos.get("qualite_cv") or {}
    textes.extend(qualite.get("tips_de_roni", []) or [])
    ev_intro = qualite.get("evaluation_introduction")
    if ev_intro and ev_intro.get("avis_critique"):
        textes.append(ev_intro["avis_critique"])
    return textes


def check_product_fit(result: dict) -> dict:
    checks = []
    candidat = result.get("candidat", {})
    recos = result.get("recommandations_agentiques", {})
    qualite = recos.get("qualite_cv") or {}

    # 1. Tutoiement : le feedback candidat ne doit jamais vouvoyer
    avis = _collect_avis_textes(result)
    if avis:
        nb_vous = sum(1 for t in avis if VOUS_PATTERN.search(t))
        nb_tu = sum(1 for t in avis if TU_PATTERN.search(t))
        checks.append(_check(
            "tutoiement_feedback_candidat",
            nb_vous == 0,
            f"{nb_tu}/{len(avis)} textes tutoient, {nb_vous} vouvoient"
        ))

    # 2. Tips de Roni présents et en nombre suffisant
    tips = qualite.get("tips_de_roni", []) or []
    checks.append(_check("tips_de_roni_presents", len(tips) >= MIN_TIPS,
                         f"{len(tips)} conseil(s), minimum attendu {MIN_TIPS}"))

    # 3. Conseil conditionnel : introduction absente => conseil d'en ajouter une
    tips_text = " ".join(tips).lower()
    if not candidat.get("introduction"):
        checks.append(_check(
            "conseil_introduction_si_absente",
            any(kw in tips_text for kw in ["introduction", "résumé", "resume", "à propos", "a propos", "profil"]),
            "introduction absente du CV : un conseil doit le mentionner"
        ))

    # 4. Conseil conditionnel : projets absents => conseil d'ajouter une section projets
    if not candidat.get("projets"):
        checks.append(_check(
            "conseil_projets_si_absents",
            "projet" in tips_text,
            "aucun projet dans le CV : un conseil doit recommander d'en ajouter"
        ))

    # 5. Évaluation d'introduction présente quand le CV en a une
    if candidat.get("introduction"):
        ev_intro = qualite.get("evaluation_introduction")
        checks.append(_check(
            "evaluation_introduction_presente",
            bool(ev_intro and ev_intro.get("avis_critique")),
            "le CV a une introduction : elle doit être évaluée"
        ))

    # 6. Livrables recruteur : contexte + axes d'entretien exploitables
    contexte = recos.get("contexte_recruteur", "") or ""
    checks.append(_check("contexte_recruteur_rempli", len(contexte.strip()) >= 100,
                         f"{len(contexte.strip())} caractères"))
    axes = recos.get("axes_entretien_simulateur", []) or []
    axes_complets = [a for a in axes if a.get("theme") and a.get("question_challenge") and a.get("competence_ciblee")]
    checks.append(_check("axes_entretien_exploitables",
                         len(axes_complets) >= MIN_AXES_ENTRETIEN,
                         f"{len(axes_complets)} axe(s) complet(s), minimum {MIN_AXES_ENTRETIEN}"))

    # 7. Bornes des scores
    bounds_ok, bounds_details = True, []
    for p in candidat.get("projets") or []:
        for axe in ["axe_impact_metier", "axe_complexite_tech", "axe_alignement_strat"]:
            a = p.get(axe)
            if a and not (0 <= a.get("score", 0) <= 10):
                bounds_ok = False
                bounds_details.append(f"{p.get('title')}.{axe}={a.get('score')}")
    score_multi = qualite.get("score_multidimensionnel") or {}
    for k, v in score_multi.items():
        if isinstance(v, (int, float)) and not (0 <= v <= 100):
            bounds_ok = False
            bounds_details.append(f"score_multidimensionnel.{k}={v}")
    prs = recos.get("production_readiness_score")
    if prs is not None and not (0 <= prs <= 100):
        bounds_ok = False
        bounds_details.append(f"production_readiness_score={prs}")
    checks.append(_check("scores_dans_bornes", bounds_ok, "; ".join(bounds_details)))

    # 8. Cohérence temporelle : pas de présent dans la synthèse si aucune exp EN_COURS
    has_en_cours = any(e.get("statut_temporel") == "EN_COURS" for e in candidat.get("experiences", []))
    if not has_en_cours and contexte:
        checks.append(_check(
            "coherence_temporelle_synthese",
            not PRESENT_PATTERN.search(contexte),
            "aucune expérience EN_COURS : la synthèse ne doit pas employer le présent d'activité"
        ))

    # 9. Profil knowledge graph calculé
    profils = recos.get("profil_calcule_knowledge_graph", []) or []
    checks.append(_check("profil_metier_calcule",
                         bool(profils) and profils[0].get("nom") != "Générique",
                         f"profil dominant: {profils[0].get('nom') if profils else 'aucun'}"))

    nb_pass = sum(1 for c in checks if c["status"] == "PASS")
    return {
        "checks": checks,
        "taux_pass": round(nb_pass / len(checks), 3) if checks else 1.0,
        "verdict": "PASS" if nb_pass == len(checks) else "FAIL",
    }

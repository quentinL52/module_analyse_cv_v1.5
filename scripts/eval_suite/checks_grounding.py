"""
Test anti-hallucination DÉTERMINISTE (sans LLM, gratuit, reproductible) :
chaque fait atomique du JSON extrait doit être retrouvable dans le texte
source du PDF. Un fait introuvable = hallucination potentielle.

Deux familles :
- faits "durs" (littéraux par nature : entreprises, postes, dates, métriques,
  diplômes, hard skills, technos, titres de projets) -> seuil strict.
- faits "souples" (soft skills, souvent légèrement reformulés) -> reporté
  séparément, non bloquant.
"""
from scripts.eval_suite.common import normalize, is_grounded

GROUNDING_HARD_THRESHOLD = 0.95  # en dessous : FAIL
GROUNDING_SOFT_THRESHOLD = 0.80  # indicatif


def _collect_facts(candidat: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Retourne (faits_durs, faits_souples) sous forme (champ, valeur)."""
    hard, soft = [], []

    if candidat.get("first_name"):
        hard.append(("first_name", candidat["first_name"]))
    if candidat.get("poste_vise_header"):
        hard.append(("poste_vise_header", candidat["poste_vise_header"]))

    skills = candidat.get("skills") or {}
    for s in skills.get("hard_skills", []):
        hard.append(("hard_skill", s.get("skill", "")))
    for s in skills.get("soft_skills", []):
        soft.append(("soft_skill", s.get("skill", "")))

    for exp in candidat.get("experiences", []):
        hard.append(("experience.poste", exp.get("poste", "")))
        hard.append(("experience.entreprise", exp.get("entreprise", "")))
        for m in exp.get("metriques_identifiees", []) or []:
            hard.append(("experience.metrique", m))

    for p in candidat.get("projets") or []:
        hard.append(("projet.title", p.get("title", "")))
        for t in p.get("technologies", []) or []:
            hard.append(("projet.technologie", t))

    for f in candidat.get("formations", []):
        hard.append(("formation.titre_diplome", f.get("titre_diplome", "")))
        hard.append(("formation.etablissement", f.get("etablissement", "")))

    for l in candidat.get("langues", []):
        hard.append(("langue", l.get("langue", "")))

    for c in candidat.get("certifications", []):
        hard.append(("certification", c.get("nom", "")))

    return hard, soft


def check_grounding(candidat: dict, source_text: str) -> dict:
    """Calcule le taux de grounding des faits extraits par rapport au texte source."""
    source_norm = normalize(source_text)
    hard, soft = _collect_facts(candidat)

    def _evaluate(facts, overlap):
        grounded, ungrounded = 0, []
        for field, value in facts:
            if not value:
                continue
            if is_grounded(value, source_norm, min_word_overlap=overlap):
                grounded += 1
            else:
                ungrounded.append({"champ": field, "valeur": value})
        total = grounded + len(ungrounded)
        rate = round(grounded / total, 4) if total else 1.0
        return rate, total, ungrounded

    hard_rate, hard_total, hard_ungrounded = _evaluate(hard, overlap=0.8)
    soft_rate, soft_total, soft_ungrounded = _evaluate(soft, overlap=0.6)

    return {
        "grounding_rate_hard": hard_rate,
        "nb_faits_durs": hard_total,
        "hallucinations_potentielles": hard_ungrounded[:30],
        "grounding_rate_soft": soft_rate,
        "nb_faits_souples": soft_total,
        "soft_skills_non_retrouves": soft_ungrounded[:15],
        "verdict": "PASS" if hard_rate >= GROUNDING_HARD_THRESHOLD else "FAIL",
        "seuil_hard": GROUNDING_HARD_THRESHOLD,
    }

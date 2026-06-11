"""
Test de STABILITÉ / FIABILITÉ des scorings : on relance le pipeline N fois
sur le même CV et on mesure la dispersion de chaque score.

Un scoring fiable doit donner (presque) la même note au même CV :
- scores numériques  -> écart-type sous seuil ;
- décisions booléennes (is_tech, is_reconversion, is_etudiant, is_disponible)
  -> aucun flip toléré (taux de flip = 0).
"""
import statistics
from collections import defaultdict

from scripts.eval_suite.common import run_pipeline_once

# Seuils de dispersion tolérés (écart-type max)
STD_MAX = {
    "production_readiness_score": 5.0,
    "score_global": 4.0,
    "score_compatibilite_sur_100": 5.0,
    "axe_projet": 1.0,            # par axe de projet (sur 10)
    "note_experience": 1.5,       # notes Roni par expérience (sur 10)
}
FLIP_RATE_MAX = 0.0  # les booléens ne doivent jamais changer entre runs


def _extract_scores(result: dict) -> dict:
    """Aplati tous les scores et booléens d'un résultat en {nom_metrique: valeur}."""
    out = {}
    candidat = result.get("candidat", {})
    recos = result.get("recommandations_agentiques", {})

    if recos.get("production_readiness_score") is not None:
        out["production_readiness_score"] = recos["production_readiness_score"]
    score_multi = (recos.get("qualite_cv") or {}).get("score_multidimensionnel") or {}
    if score_multi.get("score_global") is not None:
        out["score_global"] = score_multi["score_global"]
    matching = recos.get("analyse_matching_offre") or {}
    if matching.get("score_compatibilite_sur_100") is not None:
        out["score_compatibilite_sur_100"] = matching["score_compatibilite_sur_100"]

    for flag in ("is_reconversion", "is_etudiant", "is_disponible"):
        if candidat.get(flag) is not None:
            out[f"bool::{flag}"] = candidat[flag]

    for exp in candidat.get("experiences", []):
        ev = exp.get("evaluation_roni")
        if not ev:
            continue
        poste = exp.get("poste", "?")
        out[f"bool::is_tech::{poste}"] = ev.get("is_tech")
        for note_key in ("note_coherence_tech", "note_soft_skills_valeur"):
            if ev.get(note_key) is not None:
                out[f"note_experience::{note_key}::{poste}"] = ev[note_key]

    for p in candidat.get("projets") or []:
        title = p.get("title", "?")
        for axe in ("axe_impact_metier", "axe_complexite_tech", "axe_alignement_strat"):
            a = p.get(axe)
            if a and a.get("score") is not None:
                out[f"axe_projet::{axe}::{title}"] = a["score"]

    return out


def _std_threshold(metric: str) -> float:
    for prefix, seuil in STD_MAX.items():
        if metric.startswith(prefix):
            return seuil
    return 5.0


async def run_stability_test(pdf_bytes: bytes, filename: str, job_description: str, num_runs: int = 3) -> dict:
    """Lance le pipeline num_runs fois et mesure la dispersion des scorings."""
    series = defaultdict(list)
    runs_ok, perf_runs = 0, []

    for i in range(num_runs):
        run = await run_pipeline_once(pdf_bytes, filename, job_description)
        perf_runs.append(run["perf"])
        if run["status"] != "COMPLETED":
            continue
        runs_ok += 1
        for metric, value in _extract_scores(run["result"]).items():
            series[metric].append(value)

    metrics_report, violations = {}, []
    for metric, values in sorted(series.items()):
        if metric.startswith("bool::"):
            stable = len(set(values)) <= 1
            flip_rate = 0.0 if stable else round(1 - values.count(max(set(values), key=values.count)) / len(values), 3)
            metrics_report[metric] = {"valeurs": values, "stable": stable, "flip_rate": flip_rate}
            if flip_rate > FLIP_RATE_MAX:
                violations.append(f"{metric}: flip entre runs {values}")
        else:
            std = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0
            seuil = _std_threshold(metric)
            metrics_report[metric] = {
                "valeurs": values,
                "moyenne": round(statistics.mean(values), 2),
                "ecart_type": std,
                "etendue": round(max(values) - min(values), 2),
                "seuil_std": seuil,
            }
            if std > seuil:
                violations.append(f"{metric}: std {std} > seuil {seuil}")

    return {
        "num_runs": num_runs,
        "runs_completes": runs_ok,
        "metriques": metrics_report,
        "violations": violations,
        "perf_par_run": perf_runs,
        "verdict": "PASS" if (runs_ok == num_runs and not violations) else "FAIL",
    }

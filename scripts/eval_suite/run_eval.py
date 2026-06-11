"""
Orchestrateur de la suite d'évaluation du module d'analyse de CV.

Usage (depuis la racine du projet) :
    python -m scripts.eval_suite.run_eval --cv-dir "chemin/vers/CVs" [options]

Options :
    --cv-dir DIR            Dossier des CV PDF (défaut : ../../../Ressources/base de CV tests)
    --max-cvs N             Limiter le nombre de CV évalués
    --job-description TXT   Offre d'emploi utilisée pour le matching
    --no-judge              Désactiver l'évaluation LLM-as-a-Judge
    --stability-runs N      Nb de runs par CV pour la variance (0 = désactivé, défaut 3)
    --stability-cvs N       Nb de CV soumis au test de stabilité (défaut 2)
    --robustness            Activer la suite de robustesse (cas limites synthétiques)

Produit : scripts/eval_suite/reports/eval_report_<timestamp>.json + .md
"""
import os
import sys
import json
import asyncio
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.eval_suite.common import run_pipeline_once, extract_source_text
from scripts.eval_suite.checks_grounding import check_grounding
from scripts.eval_suite.checks_precision import check_precision, load_ground_truth
from scripts.eval_suite.checks_product import check_product_fit
from scripts.eval_suite.judge_llm import run_judges
from scripts.eval_suite.stability import run_stability_test
from scripts.eval_suite.robustness import run_robustness_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_suite.run")

DEFAULT_JOB_DESC = ("Ingénieur IA / Data Scientist avec de solides compétences en machine learning, "
                    "NLP et Python. Bonne expérience de déploiement (MLOps).")

# Seuils du scorecard global (à impact réel)
GATES = {
    "duree_max_s": 120.0,          # durée max acceptable par analyse
    "cout_max_usd": 0.03,          # coût max acceptable par analyse (~0,028 EUR)
    "grounding_hard_min": 0.95,    # taux de faits durs retrouvés dans le texte source
    "taux_completion_min": 1.0,    # tous les CV valides doivent aboutir
}


async def evaluate_one_cv(filepath: str, job_description: str, with_judge: bool, gt_dir: str = None) -> dict:
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        pdf_bytes = f.read()

    report = {"filename": filename}

    run = await run_pipeline_once(pdf_bytes, filename, job_description)
    report["pipeline"] = {"status": run["status"], "error": run.get("error")}
    report["perf"] = run["perf"]

    if run["status"] != "COMPLETED":
        return report

    result = run["result"]
    source_text = extract_source_text(pdf_bytes)

    # 1. Hallucinations (déterministe)
    report["grounding"] = check_grounding(result.get("candidat", {}), source_text)

    # 2. Précision vs vérité terrain (si disponible)
    gt = load_ground_truth(filename, gt_dir=gt_dir)
    report["precision"] = check_precision(result.get("candidat", {}), gt) if gt else {"verdict": "SKIP", "raison": "pas de ground truth"}

    # 3. Adéquation produit (déterministe)
    report["product_fit"] = check_product_fit(result)

    # 4. Juge LLM (fidélité + qualité des analyses)
    if with_judge:
        report["judge"] = await run_judges(source_text, result)

    return report


def build_scorecard(report: dict) -> dict:
    """Agrège toutes les évaluations en un scorecard global avec verdicts."""
    evals = [e for e in report.get("evaluations", []) if "perf" in e]
    completed = [e for e in evals if e["pipeline"]["status"] == "COMPLETED"]
    card = {}

    if evals:
        durations = [e["perf"]["duration_s"] for e in evals]
        costs = [e["perf"]["cost_usd"] for e in evals]
        tokens_p = [e["perf"]["tokens_prompt"] for e in evals]
        tokens_c = [e["perf"]["tokens_completion"] for e in evals]
        card["performance"] = {
            "duree_moyenne_s": round(sum(durations) / len(durations), 2),
            "duree_max_s": round(max(durations), 2),
            "cout_moyen_usd": round(sum(costs) / len(costs), 5),
            "cout_max_usd": round(max(costs), 5),
            "tokens_prompt_moyen": int(sum(tokens_p) / len(tokens_p)),
            "tokens_completion_moyen": int(sum(tokens_c) / len(tokens_c)),
            "verdict_duree": "PASS" if max(durations) <= GATES["duree_max_s"] else "FAIL",
            "verdict_cout": "PASS" if (sum(costs) / len(costs)) <= GATES["cout_max_usd"] else "FAIL",
        }
        taux_completion = len(completed) / len(evals)
        card["fiabilite_pipeline"] = {
            "taux_completion": round(taux_completion, 3),
            "verdict": "PASS" if taux_completion >= GATES["taux_completion_min"] else "FAIL",
        }

    groundings = [e["grounding"]["grounding_rate_hard"] for e in completed if "grounding" in e]
    if groundings:
        moyen = round(sum(groundings) / len(groundings), 4)
        card["hallucinations"] = {
            "grounding_hard_moyen": moyen,
            "grounding_hard_min": round(min(groundings), 4),
            "nb_cv_sous_seuil": sum(1 for g in groundings if g < GATES["grounding_hard_min"]),
            "verdict": "PASS" if min(groundings) >= GATES["grounding_hard_min"] else "FAIL",
        }

    precisions = [e["precision"] for e in completed if e.get("precision", {}).get("verdict") in ("PASS", "FAIL")]
    if precisions:
        f1s = [p["f1_moyen"] for p in precisions if p.get("f1_moyen") is not None]
        card["precision_extraction"] = {
            "nb_cv_avec_ground_truth": len(precisions),
            "f1_moyen": round(sum(f1s) / len(f1s), 3) if f1s else None,
            "verdict": "PASS" if all(p["verdict"] == "PASS" for p in precisions) else "FAIL",
        }

    product_fits = [e["product_fit"] for e in completed if "product_fit" in e]
    if product_fits:
        taux = [p["taux_pass"] for p in product_fits]
        card["adequation_produit"] = {
            "taux_checks_pass_moyen": round(sum(taux) / len(taux), 3),
            "verdict": "PASS" if all(p["verdict"] == "PASS" for p in product_fits) else "FAIL",
        }

    judges = [e["judge"] for e in completed if "judge" in e]
    if judges:
        fid = [j["extraction"].get("score_fidelite_global") for j in judges if j["extraction"].get("score_fidelite_global") is not None]
        adq = [j["analyses"].get("adequation_vision_produit") for j in judges if j["analyses"].get("adequation_vision_produit") is not None]
        nb_halluc = sum(len(j["extraction"].get("hallucinations", [])) for j in judges)
        card["juge_llm"] = {
            "fidelite_extraction_moyenne": round(sum(fid) / len(fid), 2) if fid else None,
            "adequation_vision_moyenne": round(sum(adq) / len(adq), 2) if adq else None,
            "hallucinations_relevees_par_juge": nb_halluc,
            "cout_total_juge_usd": round(sum(j.get("cout_juge_usd", 0) for j in judges), 5),
            "verdict": "PASS" if all(j["verdict"] == "PASS" for j in judges) else "FAIL",
        }

    if report.get("stabilite"):
        verdicts = [s["verdict"] for s in report["stabilite"].values()]
        card["stabilite_scoring"] = {
            "nb_cv_testes": len(verdicts),
            "verdict": "PASS" if all(v == "PASS" for v in verdicts) else "FAIL",
        }

    if report.get("robustesse"):
        card["robustesse"] = {
            "taux_pass": report["robustesse"]["taux_pass"],
            "verdict": report["robustesse"]["verdict"],
        }

    all_verdicts = []
    for section in card.values():
        for k, v in section.items():
            if k.startswith("verdict") and v in ("PASS", "FAIL"):
                all_verdicts.append(v)
    card["verdict_global"] = "PASS" if all(v == "PASS" for v in all_verdicts) else "FAIL"
    return card


def write_markdown(report: dict, path: str):
    card = report["scorecard"]
    lines = [f"# Rapport d'évaluation — module analyse CV",
             f"Généré le {report['timestamp']} — verdict global : **{card.get('verdict_global', '?')}**", ""]

    def section(title, data):
        lines.append(f"## {title}")
        for k, v in data.items():
            lines.append(f"- {k} : **{v}**" if "verdict" in k else f"- {k} : {v}")
        lines.append("")

    for key, title in [("performance", "Performance (durée / coût / tokens)"),
                       ("fiabilite_pipeline", "Fiabilité du pipeline"),
                       ("hallucinations", "Hallucinations (grounding déterministe)"),
                       ("precision_extraction", "Précision d'extraction (vs vérité terrain)"),
                       ("adequation_produit", "Adéquation vision produit (checks)"),
                       ("juge_llm", "Juge LLM (fidélité + analyses)"),
                       ("stabilite_scoring", "Stabilité des scorings"),
                       ("robustesse", "Robustesse (cas limites)")]:
        if key in card:
            section(title, card[key])

    lines.append("## Détail par CV")
    for e in report.get("evaluations", []):
        status = e["pipeline"]["status"] if "pipeline" in e else "?"
        perf = e.get("perf", {})
        g = e.get("grounding", {})
        lines.append(f"- **{e['filename']}** — {status} | {perf.get('duration_s', '?')}s | "
                     f"{perf.get('cost_usd', '?')}$ | grounding {g.get('grounding_rate_hard', 'n/a')} "
                     f"| produit {e.get('product_fit', {}).get('taux_pass', 'n/a')}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def main():
    parser = argparse.ArgumentParser(description="Suite d'évaluation du module d'analyse de CV")
    default_cv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../Ressources/base de CV tests"))
    parser.add_argument("--cv-dir", default=default_cv_dir)
    parser.add_argument("--max-cvs", type=int, default=None)
    parser.add_argument("--job-description", default=DEFAULT_JOB_DESC)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--stability-runs", type=int, default=3)
    parser.add_argument("--stability-cvs", type=int, default=2)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--ground-truth-dir", default=None, help="Dossier des annotations JSON (défaut: scripts/eval_suite/ground_truth/)")
    args = parser.parse_args()

    if not os.path.isdir(args.cv_dir):
        logger.error(f"Dossier de CV introuvable : {args.cv_dir}")
        return

    cv_files = sorted(f for f in os.listdir(args.cv_dir) if f.lower().endswith(".pdf"))
    if args.max_cvs:
        cv_files = cv_files[:args.max_cvs]
    if not cv_files:
        logger.error("Aucun CV PDF trouvé.")
        return

    logger.info(f"{len(cv_files)} CV à évaluer (juge LLM: {not args.no_judge}, "
                f"stabilité: {args.stability_runs} runs x {args.stability_cvs} CV, "
                f"robustesse: {args.robustness})")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "job_description": args.job_description,
        "evaluations": [],
    }

    # --- Évaluation principale par CV ---
    for idx, filename in enumerate(cv_files, 1):
        logger.info(f"--- CV {idx}/{len(cv_files)} : {filename} ---")
        try:
            report["evaluations"].append(
                await evaluate_one_cv(os.path.join(args.cv_dir, filename), args.job_description, not args.no_judge, gt_dir=args.ground_truth_dir)
            )
        except Exception as e:
            logger.error(f"Évaluation en échec pour {filename} : {e}")
            report["evaluations"].append({"filename": filename, "pipeline": {"status": "EVAL_ERROR", "error": str(e)}})

    # --- Stabilité des scorings ---
    if args.stability_runs > 0 and args.stability_cvs > 0:
        report["stabilite"] = {}
        for filename in cv_files[:args.stability_cvs]:
            logger.info(f"--- Stabilité ({args.stability_runs} runs) : {filename} ---")
            with open(os.path.join(args.cv_dir, filename), "rb") as f:
                pdf_bytes = f.read()
            report["stabilite"][filename] = await run_stability_test(
                pdf_bytes, filename, args.job_description, num_runs=args.stability_runs)

    # --- Robustesse ---
    if args.robustness:
        logger.info("--- Suite de robustesse (cas limites synthétiques) ---")
        report["robustesse"] = await run_robustness_suite(args.job_description)

    # --- Scorecard et rapports ---
    report["scorecard"] = build_scorecard(report)

    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(reports_dir, f"eval_report_{stamp}.json")
    md_path = os.path.join(reports_dir, f"eval_report_{stamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_markdown(report, md_path)

    logger.info(f"Verdict global : {report['scorecard'].get('verdict_global')}")
    logger.info(f"Rapports : {json_path} | {md_path}")


if __name__ == "__main__":
    asyncio.run(main())

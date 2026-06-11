"""
Tests de ROBUSTESSE sur cas limites synthétiques (PDF générés à la volée).

Vérifie que le pipeline ne crashe jamais le process, échoue proprement sur les
entrées invalides, et surtout N'INVENTE RIEN sur les entrées pauvres ou
hors-domaine (le piège classique : un document non-CV qui ressort avec des
expériences fabriquées).
"""
import logging
import fitz

from scripts.eval_suite.common import run_pipeline_once

logger = logging.getLogger("eval_suite.robustness")


def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 555, 800), text, fontsize=10)
    return doc.tobytes()


def _blank_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page()
    return doc.tobytes()


NON_CV_TEXT = """FACTURE N° 2024-0847
Société Dupont Plomberie SARL - 12 rue des Lilas, 75011 Paris
Client : M. Bernard Martin

Désignation                          Qté    PU HT     Total HT
Remplacement chauffe-eau 200L         1     890,00    890,00
Main d'oeuvre                         3h     65,00    195,00
Déplacement                           1      45,00     45,00

Total HT : 1 130,00 EUR
TVA 10% : 113,00 EUR
Total TTC : 1 243,00 EUR
Conditions de règlement : 30 jours fin de mois."""

MINIMAL_CV_TEXT = """Jean Dupont
Développeur Python
jean.dupont@email.com - 06 12 34 56 78"""

UNICODE_CV_TEXT = """Zoé Müller-Nguyễn
Ingénieure Données & IA

Compétences : Python, SQL, échantillonnage statistique, AWS (S3, EC2)
Expérience : Data Analyst — Société Générale (2021-03 - present)
- Création de tableaux de bord (+37% d'adoption)
Formation : Master Statistiques — Université Paris-Saclay (2019 - 2021)
Langues : Français (natif), Allemand (C1), Vietnamien (courant)"""

ENGLISH_CV_TEXT = """John Smith
Machine Learning Engineer

Summary: ML engineer with 4 years of experience building production recommendation systems.

Skills: Python, PyTorch, Kubernetes, MLflow

Experience:
Machine Learning Engineer - Spotify (2021-06 - present)
- Built a ranking model serving 10M daily users
- Reduced inference latency by 40%

Education:
MSc Computer Science - KTH Royal Institute of Technology (2017 - 2019)

Languages: English (native), Swedish (B2)"""


def _gate_no_invention(result: dict) -> tuple[bool, str]:
    """Aucune expérience/formation/projet ne doit être inventée sur un document qui n'en contient pas."""
    candidat = (result or {}).get("candidat", {})
    nb_exp = len(candidat.get("experiences", []))
    nb_form = len(candidat.get("formations", []))
    nb_proj = len(candidat.get("projets") or [])
    ok = nb_exp == 0 and nb_form == 0 and nb_proj == 0
    return ok, f"experiences={nb_exp}, formations={nb_form}, projets={nb_proj} (attendu: 0 partout)"


def _gate_english_preserved(result: dict) -> tuple[bool, str]:
    """L'extraction doit rester en anglais (règle de langue du prompt)."""
    candidat = (result or {}).get("candidat", {})
    responsabilites = []
    for e in candidat.get("experiences", []):
        responsabilites.extend(e.get("responsabilites_brutes", []) or [])
    text = " ".join(responsabilites).lower()
    if not text:
        return False, "aucune responsabilité extraite"
    english_markers = ["built", "reduced", "serving", "model", "users", "latency"]
    found = sum(1 for w in english_markers if w in text)
    return found >= 2, f"{found}/6 marqueurs anglais retrouvés dans les responsabilités"


# Chaque cas : (nom, bytes, statuts acceptés, gates supplémentaires sur le résultat)
def _build_cases() -> list[dict]:
    return [
        {
            "nom": "pdf_corrompu",
            "pdf": b"%PDF-1.4 ceci n'est pas un vrai pdf \x00\x01\x02",
            "statuts_acceptes": {"FAILED"},
            "gates": [],
            "attendu": "échec propre (FAILED), pas de crash du process",
        },
        {
            "nom": "page_blanche",
            "pdf": _blank_pdf(),
            "statuts_acceptes": {"FAILED", "COMPLETED"},
            "gates": [("aucune_invention", _gate_no_invention)],
            "attendu": "échec propre OU complétion sans aucune donnée inventée",
        },
        {
            "nom": "document_non_cv_facture",
            "pdf": _make_pdf(NON_CV_TEXT),
            "statuts_acceptes": {"FAILED", "COMPLETED"},
            "gates": [("aucune_invention", _gate_no_invention)],
            "attendu": "une facture ne doit produire ni expérience, ni formation, ni projet",
        },
        {
            "nom": "cv_minimal",
            "pdf": _make_pdf(MINIMAL_CV_TEXT),
            "statuts_acceptes": {"COMPLETED", "FAILED"},
            "gates": [("aucune_invention", _gate_no_invention)],
            "attendu": "extraction du peu disponible, zéro invention d'expériences",
        },
        {
            "nom": "cv_unicode_accents",
            "pdf": _make_pdf(UNICODE_CV_TEXT),
            "statuts_acceptes": {"COMPLETED"},
            "gates": [],
            "attendu": "complétion normale malgré les caractères spéciaux",
        },
        {
            "nom": "cv_anglais",
            "pdf": _make_pdf(ENGLISH_CV_TEXT),
            "statuts_acceptes": {"COMPLETED"},
            "gates": [("langue_anglaise_preservee", _gate_english_preserved)],
            "attendu": "complétion + texte extrait non traduit en français",
        },
    ]


async def run_robustness_suite(job_description: str = None) -> dict:
    """Exécute tous les cas limites. Attention : consomme des tokens (cas réels du pipeline)."""
    cases_report = []
    for case in _build_cases():
        nom = case["nom"]
        logger.info(f"Cas de robustesse : {nom}")
        try:
            run = await run_pipeline_once(case["pdf"], f"robustness_{nom}.pdf", job_description)
        except Exception as e:
            cases_report.append({
                "cas": nom, "attendu": case["attendu"],
                "statut_pipeline": "CRASH", "verdict": "FAIL",
                "detail": f"exception non gérée: {e}",
            })
            continue

        statut_ok = run["status"] in case["statuts_acceptes"]
        gates_results, gates_ok = [], True
        if run["status"] == "COMPLETED":
            for gate_name, gate_fn in case["gates"]:
                ok, detail = gate_fn(run["result"])
                gates_results.append({"gate": gate_name, "ok": ok, "detail": detail})
                gates_ok = gates_ok and ok

        cases_report.append({
            "cas": nom,
            "attendu": case["attendu"],
            "statut_pipeline": run["status"],
            "statut_accepte": statut_ok,
            "gates": gates_results,
            "perf": run["perf"],
            "verdict": "PASS" if (statut_ok and gates_ok) else "FAIL",
        })

    nb_pass = sum(1 for c in cases_report if c["verdict"] == "PASS")
    return {
        "cas": cases_report,
        "taux_pass": round(nb_pass / len(cases_report), 3) if cases_report else 1.0,
        "verdict": "PASS" if nb_pass == len(cases_report) else "FAIL",
    }

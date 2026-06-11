"""
Test de PRÉCISION d'extraction contre une vérité terrain (golden set).

Pour chaque CV `mon_cv.pdf`, déposer un fichier
`scripts/eval_suite/ground_truth/mon_cv.json` (voir ground_truth/README.md).
Sans vérité terrain le test est SKIP (le grounding et le juge LLM couvrent le reste).

Métriques : précision (ce qui est extrait est-il juste ?), rappel (a-t-on tout
extrait ?), F1 — par section — plus l'exactitude des champs scalaires.
"""
import os
import json
from difflib import SequenceMatcher

from scripts.eval_suite.common import normalize

GROUND_TRUTH_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "ground_truth")
F1_THRESHOLD = 0.85
FUZZY_RATIO = 0.80


def _similar(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= FUZZY_RATIO


def _prf(extracted: list[str], expected: list[str]) -> dict:
    """Précision / rappel / F1 entre deux listes de chaînes (matching flou)."""
    matched_ext = sum(1 for e in extracted if any(_similar(e, g) for g in expected))
    matched_gt = sum(1 for g in expected if any(_similar(g, e) for e in extracted))
    precision = matched_ext / len(extracted) if extracted else (1.0 if not expected else 0.0)
    recall = matched_gt / len(expected) if expected else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "manques": [g for g in expected if not any(_similar(g, e) for e in extracted)][:10],
        "en_trop": [e for e in extracted if not any(_similar(e, g) for g in expected)][:10],
    }


def load_ground_truth(filename: str, gt_dir: str = None) -> dict | None:
    base = os.path.splitext(os.path.basename(filename))[0]
    search_dir = gt_dir or GROUND_TRUTH_DIR
    path = os.path.join(search_dir, f"{base}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_precision(candidat: dict, ground_truth: dict) -> dict:
    """Compare le candidat extrait à la vérité terrain. Toutes les clés GT sont optionnelles."""
    report = {}
    f1_scores = []

    sections = [
        ("experiences", [f"{e.get('poste', '')} {e.get('entreprise', '')}" for e in candidat.get("experiences", [])],
         [f"{e.get('poste', '')} {e.get('entreprise', '')}" for e in ground_truth.get("experiences", [])] if "experiences" in ground_truth else None),
        ("hard_skills", [s.get("skill", "") for s in (candidat.get("skills") or {}).get("hard_skills", [])],
         ground_truth.get("hard_skills")),
        ("formations", [f.get("titre_diplome", "") for f in candidat.get("formations", [])],
         [f.get("titre_diplome", f) if isinstance(f, dict) else f for f in ground_truth["formations"]] if "formations" in ground_truth else None),
        ("langues", [l.get("langue", "") for l in candidat.get("langues", [])],
         ground_truth.get("langues")),
        ("projets", [p.get("title", "") for p in candidat.get("projets") or []],
         ground_truth.get("projets")),
        ("certifications", [c.get("nom", "") for c in candidat.get("certifications", [])],
         ground_truth.get("certifications")),
    ]
    for name, extracted, expected in sections:
        if expected is None:
            continue
        res = _prf(extracted, expected)
        report[name] = res
        f1_scores.append(res["f1"])

    # Champs scalaires
    scalars = {}
    if "first_name" in ground_truth:
        scalars["first_name"] = {
            "attendu": ground_truth["first_name"],
            "extrait": candidat.get("first_name"),
            "ok": _similar(candidat.get("first_name", ""), ground_truth["first_name"]),
        }
    if "poste_vise_header" in ground_truth:
        gt_poste = ground_truth["poste_vise_header"]
        ext_poste = candidat.get("poste_vise_header")
        scalars["poste_vise_header"] = {
            "attendu": gt_poste,
            "extrait": ext_poste,
            "ok": (gt_poste is None and ext_poste is None) or (gt_poste and ext_poste and _similar(ext_poste, gt_poste)),
        }
    if "nb_experiences" in ground_truth:
        nb = len(candidat.get("experiences", []))
        scalars["nb_experiences"] = {"attendu": ground_truth["nb_experiences"], "extrait": nb,
                                     "ok": nb == ground_truth["nb_experiences"]}
    if scalars:
        report["champs_scalaires"] = scalars

    f1_mean = round(sum(f1_scores) / len(f1_scores), 3) if f1_scores else None
    scalars_ok = all(v["ok"] for v in scalars.values()) if scalars else True
    report["f1_moyen"] = f1_mean
    report["verdict"] = "PASS" if ((f1_mean is None or f1_mean >= F1_THRESHOLD) and scalars_ok) else "FAIL"
    report["seuil_f1"] = F1_THRESHOLD
    return report

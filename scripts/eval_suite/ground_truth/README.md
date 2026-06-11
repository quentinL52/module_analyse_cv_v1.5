# Vérité terrain (golden set)

Pour mesurer la **précision d'extraction** d'un CV `mon_cv.pdf`, créer ici un
fichier `mon_cv.json` décrivant ce que le pipeline DOIT extraire. Toutes les
clés sont optionnelles : seules celles présentes sont vérifiées (les CV sans
fichier sont simplement SKIP pour ce test).

Recommandation : annoter 5 à 10 CV représentatifs (junior, senior, reconversion,
anglais, atypique). C'est le meilleur investissement pour fiabiliser les évolutions
de prompts/modèles, car ce test est 100 % déterministe et gratuit.

## Format

```json
{
  "first_name": "Jean",
  "poste_vise_header": "Data Engineer",
  "nb_experiences": 3,
  "experiences": [
    {"poste": "Data Engineer", "entreprise": "Acme"},
    {"poste": "Stage Data Analyst", "entreprise": "Globex"}
  ],
  "hard_skills": ["Python", "SQL", "Airflow", "Docker"],
  "formations": ["Master Informatique"],
  "langues": ["Français", "Anglais"],
  "projets": ["Pipeline ETL temps réel"],
  "certifications": ["AWS Cloud Practitioner"]
}
```

Notes :
- le matching est flou (casse, accents et variations légères tolérées) ;
- `poste_vise_header: null` est vérifiable (utile pour tester la règle
  « zéro inférence sur l'en-tête ») ;
- `nb_experiences` vérifie le découpage exact (ni fusion, ni invention).

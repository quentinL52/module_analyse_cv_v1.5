from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

# --- Modèles Sous-Jacents ---

class ExperienceType(str, Enum):
    PROFESSIONNELLE = "professionnelle"
    STAGE = "stage"
    ALTERNANCE = "alternance"
    PROJET_ETUDIANT = "projet_etudiant"
    ASSOCIATIF = "associatif"
    SPORTIF = "sportif"
    BENEVOLAT = "benevolat"
    AUTRE = "autre"

class Formation(BaseModel):
    titre_diplome: str = Field(description="Nom brut du diplôme ou de la formation")
    etablissement: str = Field(description="Nom de l'école, université ou organisme")
    date_debut: Optional[str] = Field(None, description="Date de début (YYYY-MM ou YYYY)")
    date_fin: Optional[str] = Field(None, description="Date de fin (YYYY-MM ou present)")

class Langue(BaseModel):
    langue: str = Field(description="Nom de la langue")
    niveau: Optional[str] = Field(None, description="Niveau tel qu'écrit dans le CV (ex: 'Courant', 'C1', 'Maternelle'). null si non précisé.")
    certification: Optional[str] = Field(None, description="Certification linguistique et score, copiés à l'identique du CV (ex: 'TOEIC : 945/990'). null si absent.")
class Certification(BaseModel):
    nom: str = Field(description="Nom de la certification")
    organisme: Optional[str] = Field(None, description="Organisme délivrant")
    date_obtention: Optional[str] = Field(None, description="Date d'obtention si précisée")

class LienExterne(BaseModel):
    url: str = Field(description="L'URL EXACTE et nettoyée (ex: linkedin.com/in/nom, github.com/nom). SÉPARER chaque lien dans un objet différent. Ignorer les mots parasites de l'OCR (ex: LINKEDIN GITHUB PORTFOLIO).")
    lien_cliquable: bool = Field(description="Vrai si la chaîne ressemble à une URL (contient .com, .fr, linkedin, github, etc.), même si mal formatée par l'OCR (hts:// au lieu de https://). Faux si c'est juste un mot isolé (ex: 'Portfolio').")

class GrammaireOrthographe(BaseModel):
    fautes_critiques_detectees: bool = Field(description="Vrai si des fautes bloquantes sont détectées")

class CompetenceTransferable(BaseModel):
    is_transferable: bool = Field(description="Indique si la compétence est transférable à un métier tech")
    applicabilite_tech: str = Field(description="Niveau d'applicabilité : Faible, Moyenne, ou Élevée")

class HardSkill(BaseModel):
    skill: str = Field(description="Nom brut de l'outil ou compétence technique")
    context: str = Field(description="Dans quel contexte la compétence a été utilisée")

class SoftSkill(BaseModel):
    skill: str = Field(description="Nom de la compétence comportementale")
    context: str = Field(description="Dans quel contexte la compétence a été démontrée")
    transferable: Optional[CompetenceTransferable] = Field(None, description="Indique si la compétence est transférable à un métier tech")

class Skills(BaseModel):
    hard_skills: List[HardSkill] = Field(default_factory=list, description="Liste exhaustive des compétences techniques (outils, langages, frameworks, etc.)")
    soft_skills: List[SoftSkill] = Field(default_factory=list, description="Liste exhaustive des compétences comportementales et savoir-être")

class AxeEvaluationProjet(BaseModel):
    justification: str
    score: int = Field(ge=0, le=10)

class EvaluationAgentiqueProjet(BaseModel):
    title: str = Field(description="Nom exact du projet")
    avis_cto: Optional[str] = None
    axe_impact_metier: Optional[AxeEvaluationProjet] = None
    axe_complexite_tech: Optional[AxeEvaluationProjet] = None
    axe_alignement_strat: Optional[AxeEvaluationProjet] = None
    score_general_projet_pourcent: Optional[int] = None

class Projet(BaseModel):
    title: str = Field(description="Nom exact du projet")
    technologies: List[str] = Field(default_factory=list, description="Technologies utilisées")
    descriptif: List[str] = Field(default_factory=list, description="Description des tâches sous forme de bullet points")
    avis_cto: Optional[str] = None
    axe_impact_metier: Optional[AxeEvaluationProjet] = None
    axe_complexite_tech: Optional[AxeEvaluationProjet] = None
    axe_alignement_strat: Optional[AxeEvaluationProjet] = None
    score_general_projet_pourcent: Optional[int] = None

class EvaluationExperienceRoni(BaseModel):
    is_tech: bool = Field(description="True si l'expérience est à dominante technologique (outils, code, infra), False sinon")
    note_coherence_tech: Optional[int] = Field(default=None, ge=0, le=10, description="Si is_tech=True : note sur 10 évaluant la cohérence technique de l'expérience")
    note_soft_skills_valeur: Optional[int] = Field(default=None, ge=0, le=10, description="Si is_tech=False : note sur 10 évaluant la mise en valeur des soft skills et compétences transférables")
    avis_roni: str = Field(description="Analyse objective et recommandations adressées au candidat")

class ProjetBrut(BaseModel):
    title: str = Field(description="Titre du projet")
    technologies: List[str] = Field(description="Liste des technos utilisées")
    descriptif: List[str] = Field(description="Description des tâches sous forme de bullet points")

class Experience(BaseModel):
    poste: str = Field(description="Intitulé du poste")
    entreprise: str = Field(description="Nom de l'entreprise")
    start_date: str = Field(description="Date de début (YYYY-MM)")
    end_date: str = Field(description="Date de fin (YYYY-MM ou present)")
    type: ExperienceType = Field(default=ExperienceType.PROFESSIONNELLE, description="Type de l'expérience (déterminé d'après le poste et les mots-clés)")
    responsabilites_brutes: List[str] = Field(description="Descriptif des responsabilités")
    metriques_identifiees: List[str] = Field(description="Liste des chiffres, volumes, KPIs mentionnés")
    statut_temporel: Optional[str] = Field(default=None, description="Statut calculé (EN_COURS, TERMINEE, INDETERMINE)")
    evaluation_roni: Optional[EvaluationExperienceRoni] = None

class Candidat(BaseModel):
    first_name: str = Field(description="Prénom du candidat (exclure nom, email et téléphone)")
    poste_vise_header: Optional[str] = Field(None, description="Titre du poste visé extrait de l'en-tête")
    poste_vise_confidence: float = Field(0.0, description="Confiance dans l'extraction du poste visé")
    introduction: Optional[str] = Field(None, description="Paragraphe d'introduction ou résumé du candidat")
    liens_externes: List[LienExterne] = Field(default_factory=list, description="Liens vers GitHub, LinkedIn, Portfolio...")
    grammaire_orthographe: Optional[GrammaireOrthographe] = None
    is_reconversion: Optional[bool] = Field(None, description="Vrai si le candidat est en reconversion")
    is_etudiant: Optional[bool] = Field(None, description="Vrai si le candidat est étudiant")
    is_disponible: Optional[bool] = Field(None, description="Vrai si le candidat est disponible")
    skills: Skills
    experiences: List[Experience] = Field(default_factory=list)
    projets: Optional[List[Projet]] = Field(default=None)
    formations: List[Formation] = Field(default_factory=list)
    langues: List[Langue] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)

class AnalyseMatchingOffre(BaseModel):
    is_offre_fournie: bool = Field(description="Indique si une description de poste a été traitée")
    score_compatibilite_sur_100: Optional[int] = Field(None, description="Score de matching si offre fournie")
    mots_cles_manquants: List[str] = Field(default_factory=list)

class ScoreMultidimensionnel(BaseModel):
    score_aspect: float = Field(description="Score sur 100 de l'aspect et la structure du CV")
    score_projets: float = Field(description="Score sur 100 de la qualité des projets")
    score_experiences: float = Field(description="Score sur 100 de la qualité et la pertinence des expériences")
    score_global: float = Field(description="Moyenne des trois dimensions sur 100")

class EvaluationIntroduction(BaseModel):
    avis_critique: str = Field(description="Avis objectif et critique rédigé directement au candidat")

class QualiteCV(BaseModel):
    score_global_sur_100: Optional[int] = Field(default=None, description="Note globale du CV (Legacy, ne plus générer par LLM)")
    score_multidimensionnel: Optional[ScoreMultidimensionnel] = Field(default=None, description="Score calculé algorithmiquement")
    evaluation_introduction: Optional[EvaluationIntroduction] = None
    compatibilite_ats: str = Field(description="Évaluation de la compatibilité ATS")
    quantification_resultats: str = Field(description="Avis sur la quantification des résultats")
    fautes_orthographe: List[str] = Field(default_factory=list, description="Liste des fautes détectées. S'il y en a, justifie impérativement où elles se trouvent (contexte/phrase).")
    tips_de_roni: List[str] = Field(description="Conseils prioritaires et recommandations d'amélioration")

class AxeEntretien(BaseModel):
    theme: str = Field(description="Thème de la question")
    question_challenge: str = Field(description="Question difficile pour évaluer le candidat")
    competence_ciblee: str = Field(description="La compétence précise ciblée par cette question")

class ProfilCalcule(BaseModel):
    nom: str
    confidence: float

class RecommandationsAgentiques(BaseModel):
    profil_calcule_knowledge_graph: List[ProfilCalcule] = Field(default_factory=list, description="Top profils métiers dominants calculés")
    production_readiness_score: Optional[int] = Field(None, description="Score de préparation pour la production")
    analyse_matching_offre: Optional[AnalyseMatchingOffre] = None
    qualite_cv: Optional[QualiteCV] = None
    contexte_recruteur: str = Field(description="Synthèse macro pour dashboard")
    axes_entretien_simulateur: List[AxeEntretien] = Field(default_factory=list)

class CVParsedFinal(BaseModel):
    candidat: Candidat
    recommandations_agentiques: RecommandationsAgentiques


# --- Modèles d'Extractions Intermédiaires (Couche 1) ---

class ExtractIdentiteSkills(BaseModel):
    first_name: str = Field(description="Prénom uniquement. RECHERCHE DANS TOUT LE DOCUMENT, y compris les bordures et l'en-tête, même si le format est éclaté.")
    poste_vise_header: Optional[str] = Field(None, description="Titre du poste visé extrait de l'en-tête. RECHERCHE DANS TOUT LE DOCUMENT.")
    poste_vise_confidence: float = Field(0.0, description="Niveau de confiance de 0.0 à 1.0 pour le poste visé.")
    introduction: Optional[str] = Field(None, description="Paragraphe d'introduction ou résumé du candidat.")
    liens_externes: List[LienExterne] = Field(default_factory=list, description="Liens vers GitHub, LinkedIn, Portfolio... Extraire chaque lien individuellement. Ne jamais fusionner plusieurs liens dans une seule string.")
    grammaire_orthographe: Optional[GrammaireOrthographe] = None
    skills: Skills = Field(description="Extraire de manière exhaustive TOUTES les compétences (hard et softs) mentionnées dans le CV avec leur contexte.")

class ExtractExperiences(BaseModel):
    experiences: List[Experience] = Field(default_factory=list, description="Liste exhaustive des expériences professionnelles.")

class ExtractProjets(BaseModel):
    projets: Optional[List[ProjetBrut]] = Field(default=None, description="Liste exhaustive des projets personnels ou académiques.")

class ExtractFormationsLangues(BaseModel):
    formations: List[Formation] = Field(default_factory=list, description="Liste exhaustive des formations et diplômes.")
    langues: List[Langue] = Field(default_factory=list, description="Liste exhaustive des langues parlées.")
    certifications: List[Certification] = Field(default_factory=list, description="Liste exhaustive des certifications.")

class ExtractHeaderVision(BaseModel):
    first_name: str = Field(description="Prénom uniquement.")
    poste_vise_header: Optional[str] = Field(None, description="Titre du poste visé extrait de l'en-tête.")
    introduction: Optional[str] = Field(None, description="Paragraphe d'introduction ou résumé du candidat.")
    liens_externes: List[LienExterne] = Field(default_factory=list, description="Liens vers GitHub, LinkedIn, Portfolio... Extraire chaque lien individuellement. Ne jamais fusionner plusieurs liens dans une seule string.")

class CVExtractionBrute(BaseModel):
    """
    Modèle recomposé à partir des extractions parallèles de la Couche 1.
    """
    first_name: str
    poste_vise_header: Optional[str] = None
    poste_vise_confidence: float = 0.0
    introduction: Optional[str] = None
    liens_externes: List[LienExterne] = Field(default_factory=list)
    grammaire_orthographe: Optional[GrammaireOrthographe] = None
    skills: Skills
    experiences: List[Experience] = Field(default_factory=list)
    projets: Optional[List[ProjetBrut]] = Field(default=None)
    formations: List[Formation] = Field(default_factory=list)
    langues: List[Langue] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)


# --- Modèles de Sortie Pydantic (Couche 3 CrewAI) ---




class OutputAuditeurProjets(BaseModel):
    projets: List[EvaluationAgentiqueProjet] = Field(default_factory=list, description="Liste des projets analysés")

class OutputProfileur(BaseModel):
    is_reconversion: bool = Field(description="Vrai si le candidat est en reconversion stricte")
    is_etudiant: bool = Field(description="Vrai si le candidat est actuellement étudiant")
    is_disponible: bool = Field(description="Vrai si le candidat est disponible (fin d'études ou fin de contrat)")
    transferables_par_skill: Dict[str, CompetenceTransferable] = Field(description="Dictionnaire où la clé est le nom exact du soft skill, et la valeur son évaluation transférable.")

class OutputRoni(BaseModel):
    qualite_cv: QualiteCV
    evaluations_experiences: Dict[str, EvaluationExperienceRoni] = Field(description="Dictionnaire où la clé est l'intitulé exact du poste.")

class OutputStratege(BaseModel):
    contexte_recruteur: str
    axes_entretien_simulateur: List[AxeEntretien]

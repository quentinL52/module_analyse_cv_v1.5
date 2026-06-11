import os
import json
import yaml
import asyncio
import logging
import litellm
from crewai import Agent, Task, Crew, Process, LLM
from src.core.config import settings
from src.core.token_tracker import litellm_success_callback

# Configuration anti-Rate Limit pour Mistral
litellm.num_retries = 4
litellm.retry_policy = "exponential_backoff"
litellm.suppress_logs = True
# Comptage des tokens de chaque appel LiteLLM (agents CrewAI inclus)
litellm.success_callback = [litellm_success_callback]

from src.schemas.cv_schema import (
    OutputAuditeurProjets,
    OutputProfileur,
    OutputRoni,
    OutputStratege,
    AnalyseMatchingOffre
)

logger = logging.getLogger(__name__)

# Load YAML configs
BASE_DIR = os.path.dirname(__file__)
AGENTS_CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'agents.yaml')
TASKS_CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'tasks.yaml')

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_crewai_llm():
    """Instancie le LLM pour CrewAI via la classe LLM native (qui wrap LiteLLM)."""
    provider = settings.DEFAULT_LLM_PROVIDER.lower()
    model = settings.DEFAULT_LLM_MODEL
    
    if provider == "openai":
        return LLM(model=model, api_key=settings.OPENAI_API_KEY, max_retries=4, temperature=0.1)
    elif provider == "mistral":
        # Pour LiteLLM, il faut préfixer les modèles mistral
        return LLM(model=f"mistral/{model}", api_key=settings.MISTRAL_API_KEY, max_retries=4, temperature=0.1)
    else:
        raise ValueError(f"Provider LLM non supporté par CrewAI : {provider}")

api_semaphore = asyncio.Semaphore(4)


def _compact_json(data) -> str:
    """Sérialisation compacte pour les prompts (pas de repr() Python, pas de champs vides)."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _slim(d: dict, keys: tuple) -> dict:
    """Ne conserve que les clés utiles et non vides d'un dict."""
    return {k: d[k] for k in keys if d.get(k) not in (None, [], "")}


# --- Payloads ciblés par agent : chaque agent ne reçoit que les données dont sa
# tâche a besoin, au lieu du CV complet (principal poste de coût de la Couche 3). ---

def _payload_auditeur(cv: dict) -> dict:
    return {
        "poste_vise_header": cv.get("poste_vise_header"),
        "projets": cv.get("projets") or [],
    }


def _payload_profileur(cv: dict) -> dict:
    return {
        "formations": cv.get("formations", []),
        "experiences": [
            _slim(e, ("poste", "entreprise", "start_date", "end_date", "type", "statut_temporel"))
            for e in cv.get("experiences", [])
        ],
        "soft_skills": (cv.get("skills") or {}).get("soft_skills", []),
    }


def _payload_ats(cv: dict) -> dict:
    return {
        "poste_vise_header": cv.get("poste_vise_header"),
        "skills": cv.get("skills"),
        "experiences": [
            _slim(e, ("poste", "type", "start_date", "end_date", "metriques_identifiees"))
            for e in cv.get("experiences", [])
        ],
        "projets": [_slim(p, ("title", "technologies")) for p in (cv.get("projets") or [])],
        "formations": [_slim(f, ("titre_diplome", "etablissement")) for f in cv.get("formations", [])],
    }


def _payload_stratege(cv: dict) -> dict:
    # La synthèse reçoit déjà les rapports des autres agents via `context=` :
    # on n'y rajoute que le minimum factuel (poste visé + statuts temporels).
    return {
        "poste_vise_header": cv.get("poste_vise_header"),
        "experiences": [_slim(e, ("poste", "statut_temporel")) for e in cv.get("experiences", [])],
    }

async def _execute_task_safely(task: Task):
    """Exécute une tâche CrewAI dans un thread pour ne pas bloquer l'Event Loop, 
    tout en la protégeant avec le sémaphore global."""
    async with api_semaphore:
        # Exécution synchrone de CrewAI déportée dans un thread
        result = await asyncio.to_thread(task.execute_sync)
        # Délai renforcé pour espacer les requêtes (Rate Limit)
        await asyncio.sleep(2)
        return result

class CVCrewOrchestrator:
    def __init__(self):
        self.agents_config = load_yaml(AGENTS_CONFIG_PATH)
        self.tasks_config = load_yaml(TASKS_CONFIG_PATH)
        self.llm = get_crewai_llm()

    def create_agents(self) -> dict:
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        time_context = f"\n\nINFORMATION CRITIQUE : Nous sommes aujourd'hui le {current_date}. Toutes les dates antérieures à aujourd'hui dans le CV sont des expériences passées ou en cours. Ne les considère JAMAIS comme étant dans le futur."

        agents = {}
        for agent_name, config in self.agents_config.items():
            backstory = config['backstory']
            backstory += time_context
                
            agents[agent_name] = Agent(
                role=config['role'],
                goal=config['goal'],
                backstory=backstory,
                llm=self.llm,
                verbose=False,
                allow_delegation=False,
                max_rpm=20
            )
        return agents

    def create_tasks(self, agents: dict, json_cv: dict, offre_emploi_texte: str = None) -> list:
        tasks = []

        def cv_context(payload: dict) -> str:
            return f"Voici les données extraites du CV :\n{_compact_json(payload)}"

        # Tâches asynchrones
        self.task_audit_projets = Task(
            description=self.tasks_config['audit_projets']['description'] + "\n" + cv_context(_payload_auditeur(json_cv)),
            expected_output=self.tasks_config['audit_projets']['expected_output'],
            agent=agents['auditeur'],

            output_pydantic=OutputAuditeurProjets
        )
        tasks.append(self.task_audit_projets)


        self.task_parcours = Task(
            description=self.tasks_config['analyse_parcours']['description'] + "\n" + cv_context(_payload_profileur(json_cv)),
            expected_output=self.tasks_config['analyse_parcours']['expected_output'],
            agent=agents['profileur'],

            output_pydantic=OutputProfileur
        )
        tasks.append(self.task_parcours)

        # Roni évalue le CV dans sa globalité : il reçoit le CV complet (épuré des champs vides)
        self.task_roni = Task(
            description=self.tasks_config['critique_globale']['description'] + "\n" + cv_context(json_cv),
            expected_output=self.tasks_config['critique_globale']['expected_output'],
            agent=agents['roni'],

            output_pydantic=OutputRoni
        )
        tasks.append(self.task_roni)

        # Conditionnel (Agent 3)
        self.task_ats = None
        if offre_emploi_texte:
            desc = self.tasks_config['matching_offre']['description'].replace('{offre_emploi_texte}', offre_emploi_texte)
            self.task_ats = Task(
                description=desc + "\n" + cv_context(_payload_ats(json_cv)),
                expected_output=self.tasks_config['matching_offre']['expected_output'],
                agent=agents['ats_matcher'],

                output_pydantic=AnalyseMatchingOffre
            )
            tasks.append(self.task_ats)

        # Tâche finale (Agent 4)
        context_tasks = [self.task_audit_projets, self.task_parcours, self.task_roni]
        if self.task_ats:
            context_tasks.append(self.task_ats)

        self.task_synthese = Task(
            description=self.tasks_config['synthese_finale']['description'] + "\n" + cv_context(_payload_stratege(json_cv)),
            expected_output=self.tasks_config['synthese_finale']['expected_output'],
            agent=agents['stratege'],
            context=context_tasks,
            output_pydantic=OutputStratege
        )
        tasks.append(self.task_synthese)

        return tasks

    async def run_async(self, json_cv: dict, offre_emploi_texte: str = None) -> dict:
        import copy
        safe_json_cv = copy.deepcopy(json_cv)
        if 'first_name' in safe_json_cv:
            safe_json_cv['first_name'] = 'ANONYMISÉ'
        if 'liens_externes' in safe_json_cv:
            safe_json_cv['liens_externes'] = []
            
        agents = self.create_agents()
        # On crée les tâches (qui sont liées à leurs agents)
        self.create_tasks(agents, safe_json_cv, offre_emploi_texte)

        # Exécution parallèle protégée par le sémaphore
        parallel_tasks = [self.task_audit_projets, self.task_parcours, self.task_roni]
        if self.task_ats:
            parallel_tasks.append(self.task_ats)
            
        await asyncio.gather(*[_execute_task_safely(t) for t in parallel_tasks])
        
        # Exécution de la tâche de synthèse (qui a besoin du contexte des précédentes)
        await _execute_task_safely(self.task_synthese)
        
        # Récupération des outputs pydantic
        outputs = {
            "audit_projets": self.task_audit_projets.output.pydantic,
            "profileur": self.task_parcours.output.pydantic,
            "roni": self.task_roni.output.pydantic,
            "stratege": self.task_synthese.output.pydantic,
        }
        if self.task_ats:
            outputs["ats"] = self.task_ats.output.pydantic

        return outputs

async def run_agentic_eval_async(json_cv: dict, offre_emploi_texte: str = None) -> dict:
    """
    Couche 3 : Exécution asynchrone sécurisée par Sémaphore avec un timeout strict.
    Passe en mode dégradé si dépassement.
    """
    orchestrator = CVCrewOrchestrator()
    try:
        result = await asyncio.wait_for(
            orchestrator.run_async(json_cv, offre_emploi_texte),
            timeout=settings.LAYER3_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Timeout Couche 3 atteint ({settings.LAYER3_TIMEOUT_SECONDS}s). Mode Dégradé activé.")
        return {"error": "MODE_DEGRADE", "message": "Les agents n'ont pas répondu à temps."}
    except Exception as e:
        logger.error(f"Erreur Couche 3 (CrewAI): {e}")
        return {"error": "FAILED", "message": str(e)}

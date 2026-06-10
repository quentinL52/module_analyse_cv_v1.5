import json
import os
import asyncio
from typing import Tuple, List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class JobProfileMatcher:
    def __init__(self):
        self.metiers_data = []
        self.metiers_names = []
        self.metiers_skills_corpus = []
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.tfidf_matrix = None
        self._load_data()

    def _load_data(self):
        """Charge le metiers.json en mémoire et précalcule la matrice TF-IDF."""
        file_path = os.path.join(os.path.dirname(__file__), "../../data/metiers.json")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for cat in data.get("metiers", []):
                for metier in cat.get("metiers", []):
                    self.metiers_data.append(metier)
                    self.metiers_names.append(metier.get("nom", "Inconnu"))
                    skills = " ".join(metier.get("skills_atomic", []))
                    self.metiers_skills_corpus.append(skills)
            
            if self.metiers_skills_corpus:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.metiers_skills_corpus)
                logger.info(f"Loaded {len(self.metiers_names)} job profiles into Knowledge Graph.")
            else:
                logger.warning("metiers.json is empty or invalid format.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de metiers.json: {e}")

    def find_dominant_profile(self, candidate_skills: list[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Calcule la similarité cosinus entre les skills du candidat et les profils métiers.
        Retourne le top K des profils.
        """
        if self.tfidf_matrix is None or not candidate_skills:
            return [{"nom": "Générique", "confidence": 0.0}]

        query_doc = " ".join(candidate_skills)
        query_vec = self.vectorizer.transform([query_doc])
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        scores = []
        for idx in best_indices:
            score = float(similarities[idx])
            if score >= 0.05:
                results.append({"nom": self.metiers_names[idx]})
                scores.append(score)
                
        if not results:
            return [{"nom": "Générique", "confidence": 0.0}]
            
        # Application du Softmax avec T=0.1
        scores_arr = np.array(scores)
        T = 0.1
        exp_scores = np.exp(scores_arr / T)
        softmax_scores = exp_scores / np.sum(exp_scores)
        
        for i, res in enumerate(results):
            res["confidence"] = float(softmax_scores[i])
            
        return results

# Instance globale (chargée au boot)
matcher = JobProfileMatcher()

async def get_dominant_profile_async(candidate_skills: list[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Couche 2 : Calcul avec timeout de sécurité de 2s.
    """
    try:
        # Exécution dans un thread pour ne pas bloquer l'Event Loop
        result = await asyncio.wait_for(
            asyncio.to_thread(matcher.find_dominant_profile, candidate_skills, top_k),
            timeout=settings.LAYER2_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Timeout Couche 2 atteint ({settings.LAYER2_TIMEOUT_SECONDS}s). Fallback sur Générique.")
        return [{"nom": "Générique", "confidence": 0.0}]
    except Exception as e:
        logger.error(f"Erreur Couche 2: {e}")
        return [{"nom": "Générique", "confidence": 0.0}]

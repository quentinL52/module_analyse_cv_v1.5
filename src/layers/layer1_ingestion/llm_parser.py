import asyncio
import instructor
from src.schemas.cv_schema import CVExtractionBrute, ExtractIdentiteSkills, ExtractExperiences, ExtractProjets, ExtractHeaderVision, ExtractFormationsLangues
from src.prompts.extraction_prompts import EXTRACTOR_SYSTEM_PROMPT
from src.core.config import settings
from src.core.llm_factory import get_instructor_client, get_default_model_name, get_vision_client_and_model
import logging
from src.layers.layer1_ingestion.extractor import extractor

logger = logging.getLogger(__name__)

async def extract_header_with_vision(base64_img: str) -> ExtractHeaderVision:
    client, model_name = get_vision_client_and_model()
    
    def _sync_call():
        return client.chat.completions.create(
            model=model_name,
            response_model=ExtractHeaderVision,
            messages=[
                {"role": "system", "content": "Tu es un expert RH. Extrais les informations d'identité depuis l'image de la première page de ce CV."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Voici la première page du CV. Trouve le prénom, le nom, le poste visé, l'introduction et les liens externes."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}
            ],
            max_retries=3
        )
    return await asyncio.to_thread(_sync_call)

async def parse_cv_brute(texte_cv: str, pdf_bytes: bytes = None) -> CVExtractionBrute:
    """
    Couche 1 : Appelle le LLM via Instructor pour extraire le JSON strict depuis le texte.
    Lancement en parallèle de 3 requêtes (Identité, Expériences, Projets) avec gpt-4o-mini.
    Intègre un timeout de sécurité (20s).
    """
    client = get_instructor_client()
    model = get_default_model_name()
    
    async def _extract_identite():
        def _sync_call():
            return client.chat.completions.create(
                model=model,
                response_model=ExtractIdentiteSkills,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Voici le texte du CV à extraire :\n\n{texte_cv}"}
                ],
                max_retries=3
            )
        return await asyncio.to_thread(_sync_call)

    async def _extract_experiences():
        def _sync_call():
            return client.chat.completions.create(
                model=model,
                response_model=ExtractExperiences,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Voici le texte du CV à extraire :\n\n{texte_cv}"}
                ],
                max_retries=3
            )
        return await asyncio.to_thread(_sync_call)

    async def _extract_projets():
        def _sync_call():
            return client.chat.completions.create(
                model=model,
                response_model=ExtractProjets,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Voici le texte du CV à extraire :\n\n{texte_cv}"}
                ],
                max_retries=3
            )
        return await asyncio.to_thread(_sync_call)

    async def _extract_formations_langues():
        def _sync_call():
            return client.chat.completions.create(
                model=model,
                response_model=ExtractFormationsLangues,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Voici le texte du CV à extraire :\n\n{texte_cv}"}
                ],
                max_retries=3
            )
        return await asyncio.to_thread(_sync_call)

    async def _run_parallel():
        return await asyncio.gather(
            _extract_identite(),
            _extract_experiences(),
            _extract_projets(),
            _extract_formations_langues()
        )

    try:
        # Timeout strict de la Couche 1
        res_identite, res_experiences, res_projets, res_formations = await asyncio.wait_for(_run_parallel(), timeout=settings.LAYER1_TIMEOUT_SECONDS)
        
        # Self-healing : Fallback Vision si identité échoue ou manque de champs cruciaux
        if pdf_bytes and (not res_identite.first_name or res_identite.first_name.lower() in ["n/a", "non précisé", "inconnu", "null", ""] or not res_identite.poste_vise_header):
            logger.info("Identité incomplète depuis le texte, déclenchement du Fallback Vision...")
            base64_img = extractor.get_first_page_as_base64(pdf_bytes)
            if base64_img:
                try:
                    vision_res = await asyncio.wait_for(extract_header_with_vision(base64_img), timeout=20.0)
                    # Merge vision results
                    if vision_res.first_name and vision_res.first_name.lower() not in ["n/a", ""]:
                        res_identite.first_name = vision_res.first_name
                    if vision_res.poste_vise_header:
                        res_identite.poste_vise_header = vision_res.poste_vise_header
                    if vision_res.introduction:
                        res_identite.introduction = vision_res.introduction
                    if vision_res.liens_externes:
                        existing_urls = [l.url for l in res_identite.liens_externes]
                        for link in vision_res.liens_externes:
                            if link.url not in existing_urls:
                                res_identite.liens_externes.append(link)
                except Exception as e:
                    logger.warning(f"Échec du Fallback Vision: {e}")

        # Recompose CVExtractionBrute
        return CVExtractionBrute(
            first_name=res_identite.first_name,
            poste_vise_header=res_identite.poste_vise_header,
            poste_vise_confidence=res_identite.poste_vise_confidence,
            introduction=res_identite.introduction,
            liens_externes=res_identite.liens_externes,
            grammaire_orthographe=res_identite.grammaire_orthographe,
            skills=res_identite.skills,
            experiences=res_experiences.experiences,
            projets=res_projets.projets,
            formations=res_formations.formations,
            langues=res_formations.langues,
            certifications=res_formations.certifications
        )

    except asyncio.TimeoutError:
        logger.error(f"Timeout de {settings.LAYER1_TIMEOUT_SECONDS}s atteint lors de l'extraction Couche 1.")
        raise
    except Exception as e:
        logger.error(f"Erreur d'extraction LLM: {str(e)}")
        raise

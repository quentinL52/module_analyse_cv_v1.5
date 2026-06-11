import asyncio
from src.schemas.cv_schema import CVExtractionBrute, ExtractHeaderVision
from src.prompts.extraction_prompts import EXTRACTOR_SYSTEM_PROMPT
from src.core.config import settings
from src.core.llm_factory import get_layer1_client_and_model, get_vision_client_and_model
from src.core.token_tracker import track_completion_usage
import logging
from src.layers.layer1_ingestion.extractor import extractor

logger = logging.getLogger(__name__)

async def extract_header_with_vision(base64_img: str) -> ExtractHeaderVision:
    # Client Instructor séparé : provider et modèle Vision pilotés par le .env
    client, model_name = get_vision_client_and_model()

    def _sync_call():
        result, completion = client.chat.completions.create_with_completion(
            model=model_name,
            response_model=ExtractHeaderVision,
            messages=[
                {"role": "system", "content": "Tu es un expert RH. Extrais les informations d'identité depuis l'image de la première page de ce CV."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Voici la première page du CV. Trouve le prénom, le nom, le poste visé, l'introduction et les liens externes."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]},
            ],
            max_retries=1
        )
        track_completion_usage(completion)
        return result
    return await asyncio.to_thread(_sync_call)

async def parse_cv_brute(texte_cv: str, pdf_bytes: bytes = None) -> CVExtractionBrute:
    """
    Couche 1 : Appelle le LLM via Instructor pour extraire le JSON strict depuis le texte.
    Un seul appel pour l'intégralité du CV (le texte n'est payé qu'une fois),
    sur un modèle dédié à l'extraction (LAYER1_LLM_MODEL).
    Intègre un timeout de sécurité.
    """
    client, model = get_layer1_client_and_model()

    def _sync_call():
        result, completion = client.chat.completions.create_with_completion(
            model=model,
            response_model=CVExtractionBrute,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"Voici le texte du CV à extraire :\n\n{texte_cv}"}
            ],
            max_retries=1
        )
        track_completion_usage(completion)
        return result

    try:
        # Timeout strict de la Couche 1
        brute_data = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=settings.LAYER1_TIMEOUT_SECONDS)

        # Self-healing : Fallback Vision si identité échoue ou manque de champs cruciaux
        if pdf_bytes and (not brute_data.first_name or brute_data.first_name.lower() in ["n/a", "non précisé", "inconnu", "null", ""] or not brute_data.poste_vise_header):
            logger.info("Identité incomplète depuis le texte, déclenchement du Fallback Vision...")
            base64_img = extractor.get_first_page_as_base64(pdf_bytes)
            if base64_img:
                try:
                    vision_res = await asyncio.wait_for(extract_header_with_vision(base64_img), timeout=20.0)
                    # Merge vision results
                    if vision_res.first_name and vision_res.first_name.lower() not in ["n/a", ""]:
                        brute_data.first_name = vision_res.first_name
                    if vision_res.poste_vise_header:
                        brute_data.poste_vise_header = vision_res.poste_vise_header
                    if vision_res.introduction:
                        brute_data.introduction = vision_res.introduction
                    if vision_res.liens_externes:
                        existing_urls = [l.url for l in brute_data.liens_externes]
                        for link in vision_res.liens_externes:
                            if link.url not in existing_urls:
                                brute_data.liens_externes.append(link)
                except Exception as e:
                    logger.warning(f"Échec du Fallback Vision: {e}")

        return brute_data

    except asyncio.TimeoutError:
        logger.error(f"Timeout de {settings.LAYER1_TIMEOUT_SECONDS}s atteint lors de l'extraction Couche 1.")
        raise
    except Exception as e:
        logger.error(f"Erreur d'extraction LLM: {str(e)}")
        raise

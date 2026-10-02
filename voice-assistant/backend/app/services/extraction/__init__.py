from app.services.extraction.base import BaseExtractor
from app.services.extraction.llm_extractor import LLMExtractor
from app.core.config import settings
from app.services.interfaces.extractor import ExtractorInterface
from app.core.exceptions import ExtractionException


def get_extractor() -> ExtractorInterface:
    """
    Factory function to get the appropriate extractor based on config.
    """
    gemini_key = settings.effective_gemini_api_key
    if gemini_key:
        return LLMExtractor(
            api_key=gemini_key,
            model=settings.extraction_model or "gemini-3.1-flash-lite",
            temperature=settings.extraction_temperature,
            provider="google"
        )
    
    if settings.openai_api_key:
        return LLMExtractor(
            api_key=settings.openai_api_key,
            model="gpt-4o",
            temperature=settings.extraction_temperature,
            provider="openai"
        )
    
    raise ExtractionException("Neither Gemini (GEMINI_API_KEY/GOOGLE_API_KEY) nor OpenAI API key configured for extraction")

__all__ = ["get_extractor", "BaseExtractor", "LLMExtractor"]


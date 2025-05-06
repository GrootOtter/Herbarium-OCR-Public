from .openai_client import OpenAICompatibleClient
from .xfyun_http_ocr_client import XFYunHttpOcrClient
from .surya_ocr_client import SuryaOcrClient # Add the new client

__all__ = [
    "OpenAICompatibleClient",
    "XFYunHttpOcrClient",
    "SuryaOcrClient", # Export the new client
]
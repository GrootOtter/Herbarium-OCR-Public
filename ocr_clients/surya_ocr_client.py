# ocr_clients/surya_ocr_client.py

import logging
import time
from typing import Optional, List, Dict, Any
from PIL import Image
import io
# Import Surya
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import torch # Surya requires torch


logger = logging.getLogger(__name__)

class SuryaOcrClient:
    """Client for local Surya OCR."""

    def __init__(self, device: str = "cpu"):
        """
        Initializes Surya OCR predictors.

        :param device: Device to run inference on ('cpu', 'cuda:0', etc.).
        """
        if RecognitionPredictor is None or DetectionPredictor is None or torch is None:
            logger.critical("Surya OCR or PyTorch not found. Please install them (`pip install surya-ocr torch torchvision`) to use Surya models.")
            raise ImportError("Surya OCR or PyTorch library is required for SuryaOcrClient.")

        self.device = torch.device(device) # Surya uses torch device object
        self.det_predictor = None
        self.rec_predictor = None
        self._load_models()

    def _load_models(self):
        """Loads the Surya detection and recognition models."""
        try:
            logger.info(f"Loading Surya OCR models onto device '{self.device}'...")
            start_time = time.time()
            # Initialize predictors (models download automatically on first run)
            self.det_predictor = DetectionPredictor(device=self.device)
            self.rec_predictor = RecognitionPredictor(device=self.device)
            load_time = time.time() - start_time
            logger.info(f"Surya OCR models loaded successfully in {load_time:.2f} seconds.")
        except Exception as e:
            logger.critical(f"Failed to load Surya OCR models on device '{self.device}': {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize Surya OCR models: {e}") from e

    def transcribe_image(
        self,
        image_bytes: bytes,
        languages: List[str], # Can be list of lang codes or None
        block_info: str = "image block",
        shared_limiter_state: Optional[Dict] = None, # Not used by local model
        limiter_lock: Optional[Any] = None           # Not used by local model
    ) -> Optional[str]:
        """
        Performs OCR on the image bytes using Surya OCR.
        Runs detection and recognition internally on the provided block.

        :param image_bytes: Bytes of the image block (PNG expected).
        :param languages: List of language codes (e.g., ['en', 'hi']) or empty list/None for auto-detection.
        :param block_info: Logging identifier.
        :return: Concatenated text lines or None on failure.
        """
        if self.rec_predictor is None or self.det_predictor is None:
            logger.error(f"Surya OCR models not loaded. Cannot transcribe {block_info}.")
            return "<!-- ERROR: SURYA OCR MODELS NOT LOADED -->"

        # Language handling: Surya prefers None for auto-detection or a list of codes.
        langs_for_surya = languages if languages else None # Pass None if list is empty
        logger.debug(f"Using languages for Surya: {langs_for_surya}")

        try:
            # Convert bytes to PIL Image
            img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            logger.debug(f"Running Surya OCR prediction for {block_info}...")
            start_time = time.time()

            # Call Surya recognition, which internally uses detection
            # Input is a list of images, list of language lists (one per image)
            predictions = self.rec_predictor([img_pil], [langs_for_surya], self.det_predictor)

            pred_time = time.time() - start_time
            logger.debug(f"Surya OCR prediction took {pred_time:.3f}s for {block_info}.")

            # Process results
            if predictions and isinstance(predictions, list) and len(predictions) > 0:
                page_result = predictions[0] # Get result for the single image block
                if page_result and hasattr(page_result, 'text_lines'):
                    # Concatenate text from all detected lines within the block
                    full_text = "\n".join([line.text for line in page_result.text_lines if line.text])
                    logger.debug(f"Surya OCR Result for {block_info}: Found {len(page_result.text_lines)} lines. Concatenated text starts: '{full_text[:50]}...'")
                    return full_text if full_text else None # Return None if no text detected
                else:
                    logger.warning(f"Surya result object for {block_info} invalid or missing 'text_lines'. Result: {page_result}")
                    return "<!-- ERROR: SURYA UNEXPECTED OUTPUT -->"
            else:
                logger.warning(f"Surya model returned empty or invalid prediction list for {block_info}. Output: {predictions}")
                return None # Indicate failure

        except Exception as e:
            logger.error(f"Error during Surya OCR prediction for {block_info}: {e}", exc_info=True)
            return f"<!-- ERROR: SURYA PREDICTION FAILED: {e} -->"
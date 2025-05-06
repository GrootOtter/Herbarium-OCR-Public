# ocr_clients/openai_client.py

import os
import logging
import time
import base64
import io
from typing import Optional, List, Dict, Any
from openai import ( OpenAI, RateLimitError, APIError, APIConnectionError,
                   APIStatusError ) # Removed BadRequestError as APIStatusError covers 4xx
from PIL import Image, UnidentifiedImageError
from multiprocessing.managers import SyncManager # Keep for type hinting if needed
from multiprocessing.synchronize import Lock as LockType # Keep for type hinting

logger = logging.getLogger(__name__)

# --- Default Constants (Import or define if not accessible) ---
DEFAULT_RESIZE_MAX_DIMENSION = 1500
DEFAULT_JPEG_QUALITY = 90

class OpenAICompatibleClient:
    """Client for OpenAI compatible APIs with image processing and rate limiting."""

    def __init__(self, api_key: str, base_url: Optional[str], model_id: str, rpm_limit: int,
                 # Use standardized parameter name
                 max_dimension: Optional[int] = None,
                 jpeg_quality: Optional[int] = None):
        """
        Initializes the client.

        :param api_key: API key for the service.
        :param base_url: Base URL for the API endpoint.
        :param model_id: Specific model ID to use.
        :param rpm_limit: Requests Per Minute limit (for rate limiting).
        :param max_dimension: Max dimension (longest side) for image resizing. Uses default if None.
        :param jpeg_quality: Quality factor (1-100) for JPEG conversion. Uses default if None.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.rpm_limit = rpm_limit if rpm_limit > 0 else 60 # Ensure positive RPM

        # Store image processing parameters, using defaults if None provided
        self.max_dimension = max_dimension if max_dimension is not None else DEFAULT_RESIZE_MAX_DIMENSION
        self.jpeg_quality = jpeg_quality if jpeg_quality is not None else DEFAULT_JPEG_QUALITY

        logger.debug(f"OpenAI Client init: model='{self.model_id}', "
                     f"resize_max_dim={self.max_dimension}, jpeg_q={self.jpeg_quality}")

    def _process_image_to_jpeg_bytes(self, image_bytes: bytes, block_info: str) -> Optional[bytes]:
        """
        Optionally resizes image based on max_dimension and converts to JPEG bytes.
        Returns original bytes if processing fails or isn't needed/configured.
        """
        try:
            # Load image from bytes
            img_pil = Image.open(io.BytesIO(image_bytes))
            original_width, original_height = img_pil.size
            resized = False

            # Resize only if max_dimension is set (> 0) and image exceeds it
            if self.max_dimension is not None and self.max_dimension > 0 and \
               (original_width > self.max_dimension or original_height > self.max_dimension):
                logger.debug(f"Block '{block_info}': Resizing ({original_width}x{original_height}) to max_dim {self.max_dimension}.")
                # Use thumbnail for in-place resizing maintaining aspect ratio
                img_pil.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                resized_width, resized_height = img_pil.size
                logger.debug(f"Block '{block_info}': Resized to {resized_width}x{resized_height}.")
                resized = True
            else:
                 logger.debug(f"Block '{block_info}': No resize needed or configured (max_dim={self.max_dimension}).")

            # Always convert to JPEG using configured quality
            if img_pil.mode not in ['RGB', 'L']: # Ensure mode is compatible with JPEG
                logger.debug(f"Block '{block_info}': Converting mode {img_pil.mode} to RGB for JPEG.")
                img_pil = img_pil.convert('RGB')

            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG', quality=self.jpeg_quality)
            processed_bytes = img_byte_arr.getvalue()

            logger.debug(f"Block '{block_info}': Processed (Resized={resized}, JPEG Q={self.jpeg_quality}). Bytes: {len(image_bytes)}->{len(processed_bytes)}")
            return processed_bytes

        except UnidentifiedImageError:
             logger.error(f"Cannot identify image data for '{block_info}' during processing.")
             return None # Return None on failure
        except Exception as img_err:
            logger.error(f"Error processing image to JPEG for '{block_info}': {img_err}", exc_info=True)
            return None # Return None on failure

    def transcribe_image(
        self,
        image_bytes: bytes,                 # Input image block bytes (expected PNG)
        languages: List[str],               # Language hints
        block_info: str = "image block",    # Logging identifier
        max_retries: int = 3,               # Max API call retries
        initial_delay: int = 5,             # Initial retry delay (seconds)
        shared_limiter_state: Optional[Dict] = None, # For shared rate limiting
        limiter_lock: Optional[LockType] = None      # Lock for shared state
    ) -> Optional[str]:
        """
        Transcribes text from image bytes using OpenAI compatible API.
        Includes image processing (resize/jpeg), rate limiting, and retries.
        """

        # 1. Rate Limiting (uses shared state if provided)
        if shared_limiter_state is not None and limiter_lock is not None:
            acquired_lock = False
            try:
                limiter_lock.acquire(); acquired_lock = True
                # Use model_id as key for shared state
                last_request_time = shared_limiter_state.get(self.model_id, 0.0)
                current_time = time.time(); time_since_last = current_time - last_request_time
                # Add small buffer (1.05x) to required interval
                required_interval = (60.0 / self.rpm_limit) * 1.05
                if time_since_last < required_interval:
                    sleep_duration = required_interval - time_since_last
                    logger.debug(f"Shared Rate Limiting ({self.model_id}): Sleeping {sleep_duration:.2f}s for {block_info}.")
                    # Release lock BEFORE sleeping
                    limiter_lock.release(); acquired_lock = False
                    time.sleep(sleep_duration)
                    # Re-acquire lock AFTER sleeping
                    limiter_lock.acquire(); acquired_lock = True
                # Update timestamp WHILE holding lock
                shared_limiter_state[self.model_id] = time.time()
            finally:
                if acquired_lock: limiter_lock.release() # Ensure lock is released
        else:
            # Fallback or single-threaded: Use instance variable (less precise)
            # Note: This part is less critical if always run in batch mode with shared state
            pass # Or implement simple instance-based delay if needed

        # 2. Process Image (Resize and Convert to JPEG)
        processed_jpeg_bytes = self._process_image_to_jpeg_bytes(image_bytes, block_info)
        if processed_jpeg_bytes is None:
            return "<!-- ERROR: IMAGE PROCESSING FAILED -->" # Return error string

        # 3. Prepare Base64 Data URI
        try:
            base64_image = base64.b64encode(processed_jpeg_bytes).decode('utf-8')
            # Use image/jpeg mime type now
            data_url = f"data:image/jpeg;base64,{base64_image}"
            logger.debug(f"Block '{block_info}': Sending JPEG base64 size: {len(base64_image)} bytes.")
        except Exception as e:
            logger.error(f"Error base64 encoding processed image for {block_info}: {e}", exc_info=True)
            return "<!-- ERROR: FAILED BASE64 ENCODING -->"

        # 4. Construct Prompt and Messages for OpenAI API
        if languages:
            language_str = f"the original language(s) (language codes/hints: {', '.join(languages)})"
        else:
            language_str = "the original language"
        prompt = f"You're an OCR expert. Perform OCR on this image. Accurately transcribe the text content you see. Preserve {language_str}. Focus only on transcription."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}} # Send JPEG data URI
                ]
            }
        ]

        # 5. API Call with Retry Logic
        current_delay = initial_delay
        for attempt in range(max_retries):
            try:
                logger.debug(f"Calling API ({self.model_id}) for {block_info} (attempt {attempt + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=2048, # Consider making configurable?
                    temperature=0.1  # Low temp for deterministic OCR
                )
                # Check for valid response content
                if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                    logger.debug(f"OCR successful for {block_info} using {self.model_id}.")
                    return response.choices[0].message.content.strip() # SUCCESS
                else:
                    logger.warning(f"No valid response content received for {block_info} (attempt {attempt + 1}). Response: {response}")
                    # Consider retrying even on empty content? For now, let retry logic handle it.

            # --- Specific Exception Handling ---
            except RateLimitError as e:
                logger.warning(f"Rate limit error (attempt {attempt + 1}/{max_retries}) for {block_info}: {e}. Retrying in {current_delay}s...")
            except APIConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}) for {block_info}: {e}. Retrying in {current_delay}s...")
            except APIStatusError as e:
                # Handle errors based on status code (e.g., 4xx client errors, 5xx server errors)
                status_code = e.status_code
                # Log client errors as ERROR, server errors as WARNING (might be temporary)
                log_level_exc = logging.ERROR if 400 <= status_code < 500 else logging.WARNING
                error_message = str(e)
                # Try to get more detail from response body if available
                try: response_body = e.response.json(); error_detail = response_body.get("error", {}).get("message"); error_message = error_detail or error_message
                except: pass # Ignore parsing errors for the error message itself
                logger.log(log_level_exc, f"API Status error {status_code} (attempt {attempt + 1}/{max_retries}) for {block_info}: {error_message}.")
                # Stop retrying immediately on non-retryable client errors (e.g., bad request, auth, not found)
                # 429 is RateLimitError, handled separately.
                if status_code in [400, 401, 403, 404, 422]:
                     logger.error(f"Non-retryable status {status_code}. Stopping retries for {block_info}.")
                     error_detail = f"HTTP {status_code}: {error_message}"
                     return f"<!-- ERROR: API Call Failed ({error_detail}) -->" # Return error string
                # Otherwise (e.g., 5xx errors), proceed to retry logic
                logger.warning(f"Retrying in {current_delay}s...")
            except APIError as e: # Catch other generic OpenAI API errors
                status_code_str = f"Status={e.status_code}" if hasattr(e, 'status_code') else "Status=N/A"
                logger.warning(f"Generic API error (attempt {attempt + 1}/{max_retries}) for {block_info}: {status_code_str}, Error={e}. Retrying in {current_delay}s...")
            except Exception as e:
                # Catch unexpected errors during the API call
                logger.error(f"Unexpected error during API call (attempt {attempt + 1}/{max_retries}) for {block_info}: {e}", exc_info=True)
                # Optionally break retry loop on unexpected errors? For now, retry.

            # --- Retry Delay ---
            if attempt < max_retries - 1:
                logger.debug(f"Waiting {current_delay}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, 60) # Exponential backoff up to 60s
            else:
                 logger.error(f"OCR failed after {max_retries} retries for {block_info} using {self.model_id}")

        return None # Return None if all retries fail
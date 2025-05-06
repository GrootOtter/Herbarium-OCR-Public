# ocr_clients/xfyun_http_ocr_client.py

import requests
import json
import base64
import hashlib
import hmac
import io
import logging
import time
from urllib.parse import urlparse, urlencode
from datetime import datetime, timezone
from time import mktime, sleep
from wsgiref.handlers import format_date_time
from typing import Optional, List, Dict, Tuple
from PIL import Image, UnidentifiedImageError
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Lock as LockType

logger = logging.getLogger(__name__)

# --- Constants ---
MAX_SIZE_GENERAL_B64 = 4 * 1024 * 1024
MAX_SIZE_MULTI_B64 = 13 * 1024 * 1024
SUPPORTED_ENCODINGS_GENERAL = {"jpg", "jpeg", "png", "bmp"}
SUPPORTED_ENCODINGS_MULTI = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}

class Url:
    """Helper class to parse URL components."""
    def __init__(self, host, path, schema):
        self.host = host
        self.path = path
        self.schema = schema

def parse_url(request_url):
    """Parses URL string into Url object."""
    try:
        stidx = request_url.find("://")
        schema = request_url[:stidx + 3] if stidx != -1 else "http://"
        host_path = request_url[stidx + 3:] if stidx != -1 else request_url
        edidx = host_path.find("/")
        host = host_path[:edidx] if edidx > 0 else host_path
        path = host_path[edidx:] if edidx > 0 else "/"
        if not host or not path:
            raise ValueError("Invalid URL structure")
        return Url(host, path, schema)
    except Exception as e:
        logger.error(f"Failed parse URL '{request_url}': {e}")
        raise ValueError(f"Invalid request URL: {request_url}") from e

class XFYunHttpOcrClient:
    """Client for XFYun HTTP OCR APIs with API-specific handling and shared rate limiting."""
    def __init__(self, app_id: str, api_key: str, api_secret: str, api_url: str,
                 service_key: str, param_type: str, param_value: str, rpm_limit: int,
                 max_dimension: int, jpeg_quality: int):
        """Initializes the client with parameters from config."""
        self.app_id, self.api_key, self.api_secret = app_id, api_key, api_secret
        self.api_url, self.service_key = api_url, service_key
        self.param_type, self.param_value = param_type, param_value
        self.rpm_limit = rpm_limit # Advisory limit
        # self._last_request_time = 0 # Removed
        self.max_dimension, self.jpeg_quality = max_dimension, jpeg_quality

        # Determine API-specific limits and keys
        if self.service_key == 'sf8e6aca1':
            self.max_b64_size, self.supported_encodings = MAX_SIZE_GENERAL_B64, SUPPORTED_ENCODINGS_GENERAL
            self.parameter_key, self.payload_data_key, self.response_payload_key = 'sf8e6aca1', 'sf8e6aca1_data_1', 'result'
        elif self.service_key == 'ocr':
            self.max_b64_size, self.supported_encodings = MAX_SIZE_MULTI_B64, SUPPORTED_ENCODINGS_MULTI
            self.parameter_key, self.payload_data_key, self.response_payload_key = 'ocr', 'image', 'ocr_output_text'
        else: raise ValueError(f"Unsupported service_key '{self.service_key}'.")
        logger.debug(f"XFYun client init: service='{self.service_key}', max_b64={self.max_b64_size}")

    def _assemble_auth_url(self) -> str:
        """Assembles the authorization URL with HMAC-SHA256 signature."""
        try:
            u = parse_url(self.api_url); host, path = u.host, u.path; now_utc = datetime.now(timezone.utc); timestamp = now_utc.timestamp(); date = format_date_time(timestamp)
            signature_origin = f"host: {host}\ndate: {date}\nPOST {path} HTTP/1.1"; logger.debug(f"Signature origin:\n{signature_origin}")
            signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
            signature_sha_base64 = base64.b64encode(signature_sha).decode('utf-8')
            authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
            v = {"authorization": authorization, "date": date, "host": host}; request_url = self.api_url + "?" + urlencode(v)
            logger.debug(f"Assembled XFYun Auth URL (start): {request_url[:100]}...")
            return request_url
        except Exception as e: logger.error(f"Error assembling auth URL: {e}", exc_info=True); raise

    def _parse_general_ocr_response(self, response_text_base64: str, block_info: str) -> Optional[str]:
        """Parses the base64 decoded JSON response from the General OCR API."""
        decoded_json_str = ""
        try:
            decoded_bytes=base64.b64decode(response_text_base64); decoded_json_str=decoded_bytes.decode('utf-8'); result_data=json.loads(decoded_json_str); full_text=[];
            if 'pages' in result_data and isinstance(result_data['pages'], list):
                for page in result_data['pages']:
                    if 'lines' in page and isinstance(page['lines'], list):
                        for line in page['lines']:
                            line_text = "".join([word.get('content','') for word in line.get('words',[]) if isinstance(word,dict)])
                            if line_text: full_text.append(line_text.strip())
            else: logger.warning(f"No 'pages' in General OCR response for {block_info}.")
            return "\n".join(full_text) if full_text else None
        except base64.B64DecodeError as e: logger.error(f"B64 decode failed (General OCR) for {block_info}: {e}"); return f"<!-- ERROR: B64 DECODE FAILED ({self.service_key}) -->"
        except json.JSONDecodeError as e: logger.error(f"JSON decode failed (General OCR) for {block_info}: {e}. String(start): {decoded_json_str[:100]}..."); return f"<!-- ERROR: JSON DECODE FAILED ({self.service_key}) -->"
        except Exception as e: logger.error(f"Unexpected error parsing General OCR for {block_info}: {e}", exc_info=True); return f"<!-- ERROR: UNEXPECTED PARSE FAIL ({self.service_key}) -->"

    def _parse_multi_ocr_response(self, response_text_base64: str, block_info: str) -> Optional[str]:
        """Parses the base64 decoded JSON response from the Multilingual Printed OCR API."""
        decoded_json_str = ""
        try:
            decoded_bytes=base64.b64decode(response_text_base64); decoded_json_str=decoded_bytes.decode('utf-8'); result_data=json.loads(decoded_json_str); full_text=[];
            if 'pages' in result_data and isinstance(result_data['pages'], list):
                for page in result_data['pages']:
                    if 'lines' in page and isinstance(page['lines'], list):
                        for line in page['lines']:
                            line_text=line.get('content','')
                            if line_text: full_text.append(line_text.strip())
            else: logger.warning(f"No 'pages' in Multi OCR response for {block_info}.")
            return "\n".join(full_text) if full_text else None
        except base64.B64DecodeError as e: logger.error(f"B64 decode failed (Multi OCR) for {block_info}: {e}"); return f"<!-- ERROR: B64 DECODE FAILED ({self.service_key}) -->"
        except json.JSONDecodeError as e: logger.error(f"JSON decode failed (Multi OCR) for {block_info}: {e}. String(start): {decoded_json_str[:100]}..."); return f"<!-- ERROR: JSON DECODE FAILED ({self.service_key}) -->"
        except Exception as e: logger.error(f"Unexpected error parsing Multi OCR for {block_info}: {e}", exc_info=True); return f"<!-- ERROR: UNEXPECTED PARSE FAIL ({self.service_key}) -->"

    def _process_image_to_jpeg(self, image_bytes: bytes, block_info: str) -> Optional[bytes]:
        """Resizes image if dimensions exceed max_dimension AND converts to JPEG bytes."""
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)); original_width, original_height = img_pil.size; resized = False;
            if original_width > self.max_dimension or original_height > self.max_dimension:
                logger.debug(f"Resizing block '{block_info}' ({original_width}x{original_height}) to max_dim {self.max_dimension}.")
                img_pil.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                resized_width, resized_height = img_pil.size; logger.debug(f"Resized to {resized_width}x{resized_height}"); resized = True;
            else:
                logger.debug(f"Block '{block_info}': Dims within max_dimension. No resize.")
            if img_pil.mode not in ['RGB', 'L']:
                img_pil = img_pil.convert('RGB')
            img_byte_arr = io.BytesIO(); img_pil.save(img_byte_arr, format='JPEG', quality=self.jpeg_quality); processed_bytes = img_byte_arr.getvalue();
            logger.debug(f"Block '{block_info}': Processed (Resized={resized}, JPEG Q={self.jpeg_quality}). Bytes: {len(image_bytes)}->{len(processed_bytes)}")
            return processed_bytes
        except Exception as img_err:
            logger.error(f"Error processing image to JPEG for '{block_info}': {img_err}", exc_info=True); return None

    def _prepare_image_for_api(self, image_bytes: bytes, block_info: str) -> Tuple[Optional[str], Optional[str]]:
        """Checks image size, determines encoding, processes if needed."""
        processed_image_bytes = image_bytes; img_format = "png"
        try:
            initial_base64 = base64.b64encode(image_bytes).decode('utf-8'); initial_b64_size = len(initial_base64); logger.debug(f"Block '{block_info}': Initial PNG b64 size={initial_b64_size} bytes. Limit={self.max_b64_size}.");
            if initial_b64_size <= self.max_b64_size:
                if 'png' not in self.supported_encodings:
                    logger.warning(f"Block '{block_info}': PNG not supported by {self.service_key}. Forcing JPEG.")
                    processed_image_bytes = self._process_image_to_jpeg(image_bytes, block_info)
                    if processed_image_bytes:
                        img_format = "jpg"
                    else:
                        return None, None
                else:
                    return initial_base64, "png"
            else: # Size exceeded
                logger.warning(f"Block '{block_info}': Initial size ({initial_b64_size}) exceeds limit ({self.max_b64_size}). Processing to JPEG.")
                processed_image_bytes = self._process_image_to_jpeg(image_bytes, block_info)
                if processed_image_bytes is None:
                    return None, None
                img_format = "jpg"; final_base64 = base64.b64encode(processed_image_bytes).decode('utf-8'); final_b64_size = len(final_base64); logger.debug(f"Block '{block_info}': Processed JPEG b64 size={final_b64_size}.");
                if final_b64_size > self.max_b64_size:
                    logger.error(f"Block '{block_info}': STILL exceeds limit ({final_b64_size}) after processing.")
                    return None, None
                else:
                    return final_base64, img_format
        except Exception as img_prep_err:
            logger.error(f"Error preparing image for API '{block_info}': {img_prep_err}", exc_info=True); return None, None

    def transcribe_image(
        self,
        image_bytes: bytes,
        languages: List[str],
        block_info: str = "image block",
        max_retries: int = 3,
        initial_delay: int = 5,
        shared_limiter_state: Optional[Dict] = None,
        limiter_lock: Optional[LockType] = None
    ) -> Optional[str]:
        """Transcribes image using XFYun API, with shared rate limiting."""

        # Rate Limiting using Shared State
        if shared_limiter_state is not None and limiter_lock is not None:
            acquired_lock = False
            try:
                limiter_lock.acquire()
                acquired_lock = True
                last_request_time = shared_limiter_state.get(self.service_key, 0.0)
                current_time = time.time()
                time_since_last = current_time - last_request_time
                required_interval = (60.0 / self.rpm_limit) * 1.05

                if time_since_last < required_interval:
                    sleep_duration = required_interval - time_since_last
                    logger.debug(f"Shared Rate Limiting ({self.service_key}): Sleeping {sleep_duration:.2f}s for {block_info}.")
                    limiter_lock.release() # Release before sleep
                    acquired_lock = False
                    sleep(sleep_duration)
                    limiter_lock.acquire() # Re-acquire after sleep
                    acquired_lock = True

                shared_limiter_state[self.service_key] = time.time() # Update while holding lock
            finally:
                if acquired_lock:
                    limiter_lock.release() # Ensure release
        else:
            logger.debug("Rate limiting shared state not provided.")

        # Prepare Image
        image_base64, img_format = self._prepare_image_for_api(image_bytes, block_info)
        if image_base64 is None: return f"<!-- ERROR: IMAGE PREPARATION FAILED ({self.service_key}) -->"

        # Assemble Auth URL
        try:
            request_url = self._assemble_auth_url()
        except Exception as auth_err:
            logger.error(f"Failed auth URL assembly: {auth_err}"); return f"<!-- ERROR: AUTH URL FAILED ({self.service_key}) -->"

        # Construct Request Body
        headers = {'Content-Type': 'application/json', 'host': parse_url(self.api_url).host}
        body = None; current_param_value = self.param_value

        # Use passed languages list for 'ocr' service
        if self.service_key == 'ocr' and self.param_type == "language":
             if languages: current_param_value = languages[0].lower(); logger.debug(f"Using lang '{current_param_value}' for Multi-OCR {block_info}.")
             else: logger.debug(f"Using default lang '{current_param_value}' for Multi-OCR.")

        try: # Build body
            request_header = {"app_id": self.app_id, "status": 3}; result_format = {"encoding": "utf8", "compress": "raw", "format": "json"};
            if self.service_key == 'sf8e6aca1':
                parameter_content = {"category": current_param_value, "result": result_format}; payload_content = {"encoding": img_format, "image": image_base64, "status": 3};
            elif self.service_key == 'ocr':
                parameter_content = {"language": current_param_value, "ocr_output_text": result_format}; payload_content = {"encoding": img_format, "image": image_base64, "status": 3};
            else: raise ValueError(f"Unknown service key {self.service_key}")
            body = {"header": request_header, "parameter": {self.parameter_key: parameter_content}, "payload": {self.payload_data_key: payload_content}}
        except Exception as body_err:
            logger.error(f"Failed request body build ({self.service_key}): {body_err}", exc_info=True); return f"<!-- ERROR: FAILED BUILD BODY ({self.service_key}) -->"

        # Send Request with Retry Logic
        current_delay = initial_delay
        for attempt in range(max_retries):
            logger.debug(f"Sending POST to XFYun ({self.service_key}) for {block_info} (attempt {attempt + 1}/{max_retries})...")
            response = None
            try:
                response = requests.post(request_url, headers=headers, data=json.dumps(body), timeout=60)
                logger.debug(f"Received response: Status={response.status_code}, Reason='{response.reason}'")
                response.raise_for_status()
                response_data = response.json(); resp_header = response_data.get("header", {}); resp_code = resp_header.get("code"); resp_sid = resp_header.get("sid", "N/A");
                if resp_code == 0: # Success
                    result_payload_part = response_data.get("payload", {}).get(self.response_payload_key, {}); result_text_base64 = result_payload_part.get("text");
                    if result_text_base64:
                        parser = self._parse_general_ocr_response if self.service_key == 'sf8e6aca1' else self._parse_multi_ocr_response
                        parsed_text = parser(result_text_base64, block_info)
                        if parsed_text is not None and not parsed_text.startswith("<!-- ERROR"): logger.debug(f"XFYun OCR success for {block_info} [SID: {resp_sid}]."); return parsed_text
                        else: logger.error(f"XFYun failed parse result for {block_info} [SID: {resp_sid}]. Parser output: {parsed_text}"); return parsed_text or f"<!-- ERROR: PARSE FAIL ({self.service_key}) -->"
                    else: logger.error(f"XFYun OK but missing text field for {block_info} [SID: {resp_sid}]. Resp: {response_data}"); return f"<!-- ERROR: XFYun MISSING TEXT ({self.service_key}) -->"
                else: # API Error Code
                    error_message = resp_header.get("message", "Unknown XFYun API error"); logger.error(f"XFYun API error for {block_info} [SID: {resp_sid}]: Code {resp_code}, Msg: {error_message}");
                    if resp_code in [10106, 10107]: logger.error("Auth error. Stopping retries."); return f"<!-- ERROR: XFYun AUTH FAILED ({resp_code}) -->"
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout (attempt {attempt+1}/{max_retries}) for {block_info}: {e}. Retrying in {current_delay}s...")
            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response is not None else "N/A"; logger.error(f"Request failed (attempt {attempt+1}/{max_retries}) for {block_info}: Status={status_code}, Error={e}. Retrying in {current_delay}s...");
                if status_code == 401 or status_code == 403: logger.error(f"Stopping retries due to {status_code}."); return f"<!-- ERROR: HTTP {status_code} GATEWAY -->"
            except json.JSONDecodeError as e:
                response_text_snippet = response.text[:100] if response and response.text else "N/A"; logger.error(f"Invalid JSON response (attempt {attempt+1}/{max_retries}) for {block_info}: {e}. Resp text: {response_text_snippet}...");
                if attempt >= 1: logger.error("Stopping retries on persistent JSON error."); return f"<!-- ERROR: INVALID JSON RESP ({self.service_key}) -->"
            except Exception as e:
                logger.error(f"Unexpected error during XFYun request (attempt {attempt+1}/{max_retries}) for {block_info}: {e}", exc_info=True)

            if attempt < max_retries - 1:
                logger.debug(f"Waiting {current_delay}s before retry {attempt + 2}/{max_retries}...")
                sleep(current_delay); current_delay = min(current_delay * 2, 60);
            else:
                logger.error(f"XFYun OCR failed after {max_retries} retries for {block_info}")

        return None # Failure
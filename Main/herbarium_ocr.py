# Main/herbarium_ocr.py

import os
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import io
import argparse
import sys
import fitz  # PyMuPDF for handling PDF and DjVu files
from PIL import Image, UnidentifiedImageError, ImageOps # Pillow for image manipulation
from tqdm import tqdm # Progress bar for loops
from doclayout_yolo import YOLOv10 # Layout detection model
import torch # PyTorch for model loading and device management
import concurrent.futures # For parallel processing in batch mode
import multiprocessing # For process management and shared state

# --- Relative Imports ---
# Import default config values and image processing utilities from the same package
from .config import (load_configuration, OCR_CONFIG as DEFAULT_OCR_CONFIG,
                     DEFAULT_RESIZE_MAX_DIMENSION, DEFAULT_JPEG_QUALITY)
from .image_processer import ImageProcessor, _initialize_tesseract, TESSERACT_AVAILABLE

# --- Import OCR Clients ---
# Import available client classes for different OCR engines/APIs
from ocr_clients import (OpenAICompatibleClient, XFYunHttpOcrClient,
                         SuryaOcrClient) # Import all available clients

# --- Import Output Generators ---
# Import functions responsible for creating different output file formats
from output_generators import (generate_markdown_output, generate_full_json_output,
                               generate_filtered_json_output, generate_xml_output,
                               generate_html_output)

# --- Setup Logging ---
# Configure basic logging format and level for the entire application
logging.basicConfig(
    level=logging.INFO, # Default level, overridden by -v flag
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(process)d] - %(message)s'
)
logger = logging.getLogger(__name__) # Logger specific to this main script module

# --- Global State ---
# These are set during runtime in the main() function or workers
MODEL_CHOICE: Optional[str] = None        # Holds the user-selected model name
active_ocr_client: Optional[Any] = None   # Holds the initialized OCR client instance

# --- Initialize Image Processor ---
# Create a single instance, as its methods are mostly stateless
image_processor = ImageProcessor()

# --- Unified OCR Call ---
def call_ocr(
    active_ocr_client: Any,                 # The initialized client instance
    image_bytes: bytes,                     # Bytes of the cropped (and potentially enhanced) image block
    languages: List[str],                   # List of language hints/codes
    block_info: str = "image block",        # Descriptive info for logging
    shared_limiter_state: Optional[Dict] = None, # Shared dict for rate limiting in batch mode
    limiter_lock: Optional[Any] = None      # Lock for accessing shared state in batch mode
    ) -> Optional[str]:
    """Calls the transcribe_image method of the active OCR client."""
    if active_ocr_client is None:
        logger.error("OCR client is not initialized. Cannot call OCR.")
        return None
    # Delegate the call to the specific client's implementation
    return active_ocr_client.transcribe_image(
        image_bytes=image_bytes,
        languages=languages,
        block_info=block_info,
        shared_limiter_state=shared_limiter_state,
        limiter_lock=limiter_lock
    )

# --- Core Processing Logic per Page ---
def process_pdf_page(
    page: fitz.Page,                # fitz Page object
    page_num: int,                  # Page number (1-based)
    doclayout_model: YOLOv10,       # Initialized layout model instance
    languages: List[str],           # Language hints for OCR
    apply_enhancements: bool,       # Flag: apply enhancements to cropped blocks?
    active_ocr_client: Any,         # Initialized OCR client instance
    app_config: Dict,               # Loaded application configuration
    pre_rotated_image: Image.Image, # PIL Image of the page (already Tesseract-rotated if enabled)
    target_device: str,             # Device ('cpu', 'cuda', etc.) for layout model
    shared_limiter_state: Optional[Dict] = None, # Shared state for rate limiting
    limiter_lock: Optional[Any] = None           # Lock for shared state
) -> List[Dict]:
    """
    Processes a single PDF page.
    Steps: 1. Layout detection. 2. Filter relevant blocks. 3. For each block: Crop, Enhance (optional), OCR.
    Returns a list of result dictionaries for the page.
    """
    logger.debug(f"Starting processing page {page_num}...")
    page_results = []
    # Get relevant sub-configs for easier access
    ocr_cfg = app_config.get('OCR_CONFIG', {})
    doc_cfg = app_config.get('DOCLAYOUT_CONFIG', {})
    relevant_classes = set(doc_cfg.get("RELEVANT_TEXT_CLASSES", []))
    # Check which enhancement steps are enabled based on flags and config
    do_contrast = apply_enhancements and ocr_cfg.get('enhance_contrast', True)
    do_denoise = apply_enhancements and ocr_cfg.get('denoise', True)
    do_sharpen = apply_enhancements and ocr_cfg.get('sharpen', True)
    denoise_mode = ocr_cfg.get("denoise_mode", "gray")

    try:
        # 1. Layout detection on the potentially pre-rotated full page image
        det_res = doclayout_model.predict(
            pre_rotated_image,
            imgsz=doc_cfg.get("DOCLAYOUT_IMG_SIZE", 1024),
            conf=doc_cfg.get("DOCLAYOUT_CONF_THRESHOLD", 0.3),
            device=target_device # Use specified device
        )
        # Check result structure before accessing boxes
        num_boxes = len(det_res[0].boxes) if det_res and len(det_res) > 0 and det_res[0].boxes is not None else 0
        logger.debug(f"Page {page_num}: Layout found {num_boxes} boxes.")

        # 2. Filter and sort relevant text blocks based on class name
        text_blocks = []
        if num_boxes > 0:
            for box_data in det_res[0].boxes.data:
                try:
                    x1, y1, x2, y2, conf, class_id_tensor = box_data
                    class_id = int(class_id_tensor.item())
                    # Ensure class_id is valid before accessing names
                    if 0 <= class_id < len(doclayout_model.model.names):
                        class_name = doclayout_model.model.names[class_id]
                        # Keep only blocks whose class is in the relevant list
                        if class_name in relevant_classes:
                            text_blocks.append({
                                "box": (int(x1), int(y1), int(x2), int(y2)),
                                "class": class_name,
                                "confidence": float(conf.item()) # Store layout confidence
                            })
                    else:
                        logger.warning(f"Page {page_num}: Invalid class_id {class_id} detected by layout model.")
                except Exception as box_err:
                    # Log error if processing a specific box fails
                    logger.error(f"Page {page_num}: Error processing layout box data: {box_data} -> {box_err}", exc_info=True)
            text_blocks.sort(key=lambda b: b['box'][1]) # Sort top-to-bottom
        logger.debug(f"Page {page_num}: Found {len(text_blocks)} relevant blocks.")

        # 3. Process each relevant block
        for i, block in enumerate(text_blocks):
            box = block['box']
            block_info = f"page {page_num}, block {i+1}/{len(text_blocks)} (Class: {block['class']}, Conf: {block.get('confidence', -1):.2f})"
            logger.debug(f"  Processing block: {block_info} @ {box}")
            img_bytes = None # Initialize image bytes for OCR

            try:
                # Validate box coordinates
                if not (isinstance(box, (tuple, list)) and len(box) == 4 and box[0] < box[2] and box[1] < box[3]):
                    raise ValueError(f"Invalid box coordinates: {box}")

                # Crop block from the potentially pre-rotated full page image
                cropped_img = pre_rotated_image.crop(box)
                processed_block_img = cropped_img # Start with cropped image
                enhancements_applied_list = []

                # Apply enhancements to the *cropped* block if enabled
                if apply_enhancements:
                    img_before_step = processed_block_img # Track if image changes
                    # Apply contrast if enabled
                    if do_contrast:
                        processed_block_img = image_processor.enhance_contrast(processed_block_img)
                        if processed_block_img is not img_before_step: enhancements_applied_list.append("contrast"); img_before_step = processed_block_img
                    # Apply denoising if enabled
                    if do_denoise:
                        denoise_func = image_processor.denoise_image_color if denoise_mode == "color" else image_processor.denoise_image_gray
                        if denoise_func: processed_block_img = denoise_func(processed_block_img)
                        else: logger.warning(f"Unknown denoise_mode '{denoise_mode}'. Skipping.")
                        if processed_block_img is not img_before_step: enhancements_applied_list.append(f"denoise_{denoise_mode}"); img_before_step = processed_block_img
                    # Apply sharpening if enabled
                    if do_sharpen:
                        processed_block_img = image_processor.sharpen_image(processed_block_img)
                        if processed_block_img is not img_before_step: enhancements_applied_list.append("sharpen")
                    # Log applied enhancements if any
                    if enhancements_applied_list: logger.debug(f"    Applied block enhancements: {enhancements_applied_list}")

                # Convert final processed block to PNG bytes for OCR client
                img_byte_arr = io.BytesIO()
                processed_block_img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

            except Exception as proc_err:
                # Log error during cropping or enhancement, add error result
                logger.error(f"    Error during crop/enhance for block {block_info}: {proc_err}", exc_info=logger.isEnabledFor(logging.DEBUG))
                page_results.append({
                    "page": page_num, "class": block.get('class','unknown'),
                    "box": box, "confidence": block.get('confidence'),
                    "text": f"<!-- ERROR PROCESSING BLOCK: {proc_err} -->"
                })
                continue # Skip OCR for this block

            # 4. Call OCR if block processed successfully
            if img_bytes:
                ocr_text = call_ocr(
                    active_ocr_client, img_bytes, languages, block_info,
                    shared_limiter_state, limiter_lock
                )
                # 5. Store result
                page_results.append({
                    "page": page_num, "class": block['class'], "box": box,
                    "confidence": block.get('confidence'), # Include layout confidence
                    "text": ocr_text if ocr_text else "<!-- OCR FAILED -->"
                })
            # If img_bytes is None, error result was already added

        logger.debug(f"Finished processing page {page_num}.")
        return page_results

    except Exception as e:
        # Catch unexpected errors during page processing
        logger.error(f"Critical error processing page {page_num}: {e}", exc_info=True)
        # Return an error entry for this page
        return [{"page": page_num, "error": f"Failed page processing: {e}"}]

# --- Core Processing Logic per Image File ---
def process_single_image(
    image_path_str: str,            # Path to the image file
    doclayout_model: YOLOv10,       # Initialized layout model
    languages: List[str],           # Language hints
    apply_enhancements: bool,       # Flag: apply enhancements to cropped blocks?
    attempt_rotation: bool,         # Flag: attempt Tesseract rotation on full image?
    active_ocr_client: Any,         # Initialized OCR client instance
    app_config: Dict,               # Loaded application configuration
    target_device: str,             # Device for layout model
    shared_limiter_state: Optional[Dict] = None, # Shared state for rate limiting
    limiter_lock: Optional[Any] = None           # Lock for shared state
) -> List[Dict]: # Return list of result dictionaries
    """Processes single image: EXIF rot -> Tesseract rot -> Layout -> Enhance/OCR Cropped Blocks."""
    results = []
    # Get config values
    ocr_cfg = app_config.get('OCR_CONFIG', {}); doc_cfg = app_config.get('DOCLAYOUT_CONFIG', {});
    relevant_classes = set(doc_cfg.get("RELEVANT_TEXT_CLASSES", []));
    do_contrast = apply_enhancements and ocr_cfg.get('enhance_contrast', True)
    do_denoise = apply_enhancements and ocr_cfg.get('denoise', True)
    do_sharpen = apply_enhancements and ocr_cfg.get('sharpen', True)
    denoise_mode = ocr_cfg.get("denoise_mode", "gray")

    try:
        with Image.open(image_path_str) as img:
            logger.debug(f"Opened image: {image_path_str}")

            # 1. Apply EXIF Orientation Correction FIRST
            try:
                img_oriented = ImageOps.exif_transpose(img)
                logger.debug("Applied EXIF correction (if needed).")
            except Exception as exif_err:
                logger.warning(f"Could not apply EXIF orientation from {image_path_str}: {exif_err}. Using original.")
                img_oriented = img

            # 2. Convert to RGB
            try:
                img_rgb = img_oriented.convert("RGB")
            except Exception as convert_err:
                logger.error(f"Failed to convert image {image_path_str} to RGB: {convert_err}")
                raise # Propagate error, cannot proceed

            # 3. Optional FULL IMAGE Auto-Rotation (Tesseract)
            image_for_layout = img_rgb # Start with EXIF corrected RGB image
            if attempt_rotation:
                logger.debug("Attempting Tesseract auto-rotation...")
                image_for_layout = image_processor.auto_rotate_image(img_rgb, app_config)
            else:
                 logger.debug("Skipping Tesseract auto-rotation (disabled in config).")

            # 4. Layout Analysis
            logger.debug(f"Running layout detection on device {target_device}...")
            det_res = doclayout_model.predict(
                image_for_layout,
                imgsz=doc_cfg.get("DOCLAYOUT_IMG_SIZE", 1024),
                conf=doc_cfg.get("DOCLAYOUT_CONF_THRESHOLD", 0.3),
                device=target_device # Use specified device
            )
            num_boxes = len(det_res[0].boxes) if det_res and len(det_res) > 0 and det_res[0].boxes is not None else 0
            logger.debug(f"Layout found {num_boxes} boxes.")

            # 5. Filter and sort relevant text blocks
            text_blocks = []
            if num_boxes > 0:
                for box_data in det_res[0].boxes.data:
                    try:
                        x1, y1, x2, y2, conf, class_id_tensor = box_data; class_id = int(class_id_tensor.item())
                        if 0 <= class_id < len(doclayout_model.model.names):
                            class_name = doclayout_model.model.names[class_id]
                            if class_name in relevant_classes:
                                text_blocks.append({"box":(int(x1),int(y1),int(x2),int(y2)), "class":class_name, "confidence":float(conf.item())})
                        else: logger.warning(f"Invalid class_id {class_id} detected in image.")
                    except Exception as box_err: logger.error(f"Error processing box data in image: {box_data} -> {box_err}", exc_info=True); continue
                text_blocks.sort(key=lambda b: b['box'][1])
            logger.debug(f"Found {len(text_blocks)} relevant blocks.")

            # 6. Process each relevant block
            for i, block in enumerate(text_blocks):
                box = block['box']
                block_info = f"image '{os.path.basename(image_path_str)}', block {i+1}/{len(text_blocks)} (Class: {block['class']}, Conf: {block.get('confidence', -1):.2f})"
                logger.debug(f"  Processing block: {block_info} @ {box}")
                img_bytes = None # Initialize
                try:
                    # Validate & Crop
                    if not (isinstance(box, (tuple, list)) and len(box) == 4 and box[0] < box[2] and box[1] < box[3]): raise ValueError(f"Invalid box: {box}")
                    cropped_img = image_for_layout.crop(box)
                    processed_block_img = cropped_img; enhancements_applied_list = []
                    # Enhance cropped block if enabled
                    if apply_enhancements:
                        img_before_step = processed_block_img
                        if do_contrast: processed_block_img = image_processor.enhance_contrast(processed_block_img);
                        if processed_block_img is not img_before_step: enhancements_applied_list.append("contrast"); img_before_step = processed_block_img
                        if do_denoise:
                            denoise_func = image_processor.denoise_image_color if denoise_mode == "color" else image_processor.denoise_image_gray
                            if denoise_func: processed_block_img = denoise_func(processed_block_img)
                            else: logger.warning(f"Unknown denoise_mode '{denoise_mode}'. Skipping.")
                            if processed_block_img is not img_before_step: enhancements_applied_list.append(f"denoise_{denoise_mode}"); img_before_step = processed_block_img
                        if do_sharpen: processed_block_img = image_processor.sharpen_image(processed_block_img);
                        if processed_block_img is not img_before_step: enhancements_applied_list.append("sharpen")
                        if enhancements_applied_list: logger.debug(f"    Applied block enhancements: {enhancements_applied_list}")
                    # Convert final block to PNG bytes
                    img_byte_arr = io.BytesIO(); processed_block_img.save(img_byte_arr, format='PNG'); img_bytes = img_byte_arr.getvalue()
                except Exception as proc_err:
                    logger.error(f"    Error crop/enhance block {block_info}: {proc_err}", exc_info=logger.isEnabledFor(logging.DEBUG))
                    results.append({"class": block.get('class', 'unknown'), "box": box, "confidence": block.get('confidence'), "error": f"Failed to process block: {proc_err}"}); continue

                # 7. Call OCR
                if img_bytes:
                    ocr_text = call_ocr(active_ocr_client, img_bytes, languages, block_info, shared_limiter_state, limiter_lock)
                    results.append({"class": block['class'], "box": box, "confidence": block.get('confidence'), "text": ocr_text if ocr_text else "<!-- OCR FAILED -->"})

    except FileNotFoundError: logger.error(f"Image file not found: {image_path_str}"); raise
    except UnidentifiedImageError: logger.error(f"Cannot identify image file (maybe corrupt?): {image_path_str}"); raise
    except Exception as e: logger.error(f"Failed processing image {image_path_str}: {e}", exc_info=True); raise # Re-raise

    return results # Return the list of results for this image

# --- Worker Function (for Batch Mode) ---
def process_single_file_worker(
    input_path_str: str,            # Path to the input file for this worker
    final_output_base_str: str,     # Base path for output files (worker adds suffix)
    output_format: Optional[str],   # Requested conversion format (or None)
    languages: List[str],           # Language hints
    attempt_rotation: bool,         # Global flag: try Tesseract rotation?
    app_config: Dict,               # Full application configuration (read-only copy)
    target_device: str,             # Determined device for this worker
    shared_limiter_state: Optional[Dict] = None, # Shared dict for rate limiting
    limiter_lock: Optional[Any] = None           # Lock for shared dict
) -> bool: # Return True on success, False on failure
    """
    Worker function executed by each process in the pool for batch mode.
    Initializes its own resources (models, clients) and processes one file.
    Handles output file generation for its assigned file.
    """
    # Setup logging for this specific worker process
    worker_pid = os.getpid()
    worker_logger = logging.getLogger(f"Worker_{worker_pid}")
    log_level_worker = app_config.get('_log_level', logging.INFO)
    # Configure logging if the root logger wasn't configured (can happen with spawn)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=log_level_worker, format='%(asctime)s-%(levelname)s-[%(name)s:%(funcName)s:%(process)d]- %(message)s')
    worker_logger.setLevel(log_level_worker)

    worker_logger.info(f"Worker started for: {os.path.basename(input_path_str)} on device {target_device}")
    success = False # Flag to track if processing succeeded
    active_ocr_client = None # Worker-specific client instance
    doclayout_model = None   # Worker-specific layout model instance
    all_results = [] # Results for this specific file

    try:
        # Retrieve necessary configurations within the worker
        ocr_cfg = app_config.get('OCR_CONFIG', {})
        doc_cfg = app_config.get('DOCLAYOUT_CONFIG', {})
        model_configs_final = app_config.get('MODEL_CONFIGS', {})
        model_choice = app_config.get('_model_choice', '')
        apply_enhancements_worker = app_config.get('_apply_enhancements_run', False) # Get enhancement flag

        # Initialize Tesseract within Worker if needed
        if attempt_rotation:
            worker_logger.debug("Worker initializing Tesseract...")
            _initialize_tesseract(app_config) # Needs the config for path

        # Load DocLayout Model within Worker
        doclayout_model_path = doc_cfg.get("DOCLAYOUT_MODEL_PATH")
        if not doclayout_model_path or not Path(doclayout_model_path).is_file():
            raise FileNotFoundError(f"Layout model path invalid or not found: '{doclayout_model_path}'")
        worker_logger.debug(f"Worker loading layout model...")
        doclayout_model = YOLOv10(str(doclayout_model_path)) # Let YOLO handle device in predict call
        if not hasattr(doclayout_model, 'model'):
            raise RuntimeError("Loaded layout model object seems invalid.")
        worker_logger.debug("Worker layout model loaded.")

        # Initialize OCR Client within Worker
        model_config = model_configs_final.get(model_choice)
        if not model_config: raise ValueError(f"Model config '{model_choice}' not found")
        model_type = model_config.get("type")

        if model_type == "openai_compatible":
            api_key = os.getenv(model_config.get("api_key_env",""));
            if not api_key: raise ValueError(f"Env var {model_config.get('api_key_env')} not set for {model_choice}")
            resize_dim = model_config.get("max_dimension", DEFAULT_RESIZE_MAX_DIMENSION)
            jpeg_q = model_config.get("jpeg_quality", DEFAULT_JPEG_QUALITY)
            active_ocr_client = OpenAICompatibleClient(api_key=api_key, base_url=model_config.get("base_url"), model_id=model_config.get("model_id"), rpm_limit=model_config.get("rpm_limit"), max_dimension=resize_dim, jpeg_quality=jpeg_q)
        elif model_type == "xfyun_http_ocr":
            app_id = os.getenv(model_config.get("app_id_env","")); api_key = os.getenv(model_config.get("api_key_env","")); api_secret = os.getenv(model_config.get("api_secret_env",""));
            if not all([app_id, api_key, api_secret]): raise ValueError(f"XFyun env vars missing for {model_choice}")
            active_ocr_client = XFYunHttpOcrClient(app_id=app_id, api_key=api_key, api_secret=api_secret, api_url=model_config.get("api_url"), service_key=model_config.get("service_key"), param_type=model_config.get("param_type"), param_value=model_config.get("param_value"), rpm_limit=model_config.get("rpm_limit", 60), max_dimension=model_config.get("max_dimension", 1500), jpeg_quality=model_config.get("jpeg_quality", 85))
        elif model_type == "surya_ocr":
            if SuryaOcrClient is None: raise ImportError("Surya OCR client requested but library not found.")
            surya_device_cfg = model_config.get("device", target_device) # Use worker's device
            active_ocr_client = SuryaOcrClient(device=surya_device_cfg)
        else: raise ValueError(f"Unsupported model type '{model_type}'")
        worker_logger.debug(f"Worker OCR client for '{model_choice}' initialized.")

        # Process the file assigned to this worker
        input_path_obj = Path(input_path_str)
        file_ext = input_path_obj.suffix.lower()
        if file_ext in ['.pdf', '.djvu']:
             doc = fitz.open(input_path_str)
             worker_logger.debug(f"Processing PDF '{os.path.basename(input_path_str)}' ({len(doc)} pages).")
             for page_num in range(len(doc)):
                 page = None
                 try:
                     page = doc.load_page(page_num)
                     dpi = ocr_cfg.get("dpi", 300); mat = fitz.Matrix(dpi/72, dpi/72); pix = page.get_pixmap(matrix=mat, alpha=False); img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                     rotated_img = image_processor.auto_rotate_image(img, app_config) if attempt_rotation else img
                     # Pass target_device to page processor
                     all_results.extend(process_pdf_page(page, page_num + 1, doclayout_model, languages, apply_enhancements_worker, active_ocr_client, app_config, rotated_img, target_device, shared_limiter_state, limiter_lock))
                 except Exception as page_err: worker_logger.error(f"Error processing page {page_num+1}: {page_err}", exc_info=True); all_results.append({"page": page_num + 1, "error": f"Page fail: {page_err}"})
                 finally: del page
             doc.close()
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
             # Pass target_device to image processor
             all_results = process_single_image(input_path_str, doclayout_model, languages, apply_enhancements_worker, attempt_rotation, active_ocr_client, app_config, target_device, shared_limiter_state, limiter_lock)
        else:
             worker_logger.warning(f"Unsupported file type '{file_ext}'. Skipping {input_path_str}.")
             return False # Indicate failure

        # Generate outputs within the worker for this file
        if all_results:
            # Always generate the full JSON output
            full_json_path = Path(f"{final_output_base_str}_full.json")
            generate_full_json_output(all_results, str(full_json_path), source_filename=input_path_str)
            # Generate the requested format if specified
            if output_format and output_format != "full_json":
                requested_format_path = Path(f"{final_output_base_str}.{output_format}")
                generator_map = {
                    "markdown": generate_markdown_output,
                    "json": generate_filtered_json_output, # 'json' means filtered
                    "xml": generate_xml_output,
                    "html": generate_html_output
                }
                gen_func = generator_map.get(output_format)
                if gen_func:
                    gen_func(all_results, str(requested_format_path), source_filename=input_path_str)
                else:
                    worker_logger.error(f"Invalid output format '{output_format}' specified in worker.")
        else:
            worker_logger.warning(f"No results generated for {input_path_str}, skipping output file generation.")

        success = True # Mark success if processing reached here without critical error

    except Exception as worker_err:
         # Catch any exception during worker execution
         worker_logger.error(f"Error processing {os.path.basename(input_path_str)} in worker: {worker_err}", exc_info=True)
         success = False
    finally:
        # Ensure resources are released (important in multiprocessing)
        del doclayout_model
        del active_ocr_client
        worker_logger.info(f"Worker finished for: {os.path.basename(input_path_str)} (Success: {success})")
        return success


# --- Batch Processing Orchestration ---
def process_batch_directory(
    input_dir: str,                 # Directory containing input files
    file_extensions: Tuple[str],    # Tuple of valid file extensions
    output_format: Optional[str],   # Requested conversion format (or None)
    languages: List[str],           # Language hints
    app_config: Dict,               # Loaded application configuration
    target_device: str              # Device determined in main()
):
    """Finds files and distributes them to worker processes using ProcessPoolExecutor."""
    logger.info(f"Starting batch processing for directory: {input_dir}")
    input_dir_path = Path(input_dir)
    if not input_dir_path.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return

    ocr_cfg = app_config.get('OCR_CONFIG', {})
    # Determine number of workers based on config or CPU count
    max_workers = ocr_cfg.get("max_workers", 1)
    if max_workers <= 0:
        cpu_count = os.cpu_count()
        max_workers = cpu_count if cpu_count else 1
        logger.info(f"max_workers <= 0, using system CPU count: {max_workers}")
    else:
        logger.info(f"Using up to {max_workers} worker processes.")

    # Determine and create output subdirectory
    output_subdir_name = ocr_cfg.get("batch_output_subdir_name", "herbariumOCR_output")
    output_dir_for_batch = input_dir_path / output_subdir_name
    try:
        output_dir_for_batch.mkdir(parents=True, exist_ok=True)
        logger.info(f"Batch output directory: {output_dir_for_batch}")
    except Exception as e:
        logger.error(f"Failed create batch output dir '{output_dir_for_batch}': {e}")
        return

    # Find files matching extensions
    files_to_process = []
    try:
        logger.debug(f"Scanning {input_dir} for files with extensions: {file_extensions}")
        for item in input_dir_path.iterdir():
            if item.is_file() and item.suffix.lower() in file_extensions:
                 files_to_process.append(item.name)
    except OSError as e:
        logger.error(f"Error reading input directory {input_dir}: {e}")
        return

    if not files_to_process:
        logger.warning(f"No files with extensions {file_extensions} found in directory: {input_dir}")
        return
    logger.info(f"Found {len(files_to_process)} files to process.")

    # Prepare tasks for the process pool
    model_choice = app_config.get('_model_choice', '')
    model_suffix = f"_{model_choice}" if model_choice else ""
    attempt_rotation_global = ocr_cfg.get("attempt_auto_rotation", False)

    # Use a Manager for shared state (rate limiter) across processes
    with multiprocessing.Manager() as manager:
        shared_limiter_state = manager.dict() # Shared dictionary for {client_id: last_request_timestamp}
        limiter_lock = manager.Lock()         # Lock to synchronize access to the dict
        logger.debug("Created multiprocessing Manager and Lock for shared rate limiting.")

        tasks = []
        for filename in files_to_process:
            full_input_path = input_dir_path / filename
            base_filename = full_input_path.stem
            # Base path for outputs (worker adds suffixes like _full.json, .md)
            final_output_base = output_dir_for_batch / f"{base_filename}{model_suffix}"
            # Package arguments for the worker function
            tasks.append({
                "input_path_str": str(full_input_path),
                "final_output_base_str": str(final_output_base), # Pass base path
                "output_format": output_format, # Pass requested format (or None)
                "languages": languages,
                "attempt_rotation": attempt_rotation_global,
                "app_config": app_config, # Pass the full config
                "target_device": target_device, # Pass determined device
                "shared_limiter_state": shared_limiter_state, # Pass shared dict
                "limiter_lock": limiter_lock             # Pass shared lock
            })

        processed_count = 0
        failed_count = 0
        logger.info(f"Submitting {len(tasks)} tasks to process pool...")

        # Execute tasks using ProcessPoolExecutor
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map futures to input paths for better logging on error/completion
                futures = { executor.submit(process_single_file_worker, **task): task["input_path_str"] for task in tasks }

                # Process results as they complete using tqdm for progress bar
                # Disable progress bar if logging level is DEBUG or higher
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing batch", unit="file", disable=logger.level <= logging.DEBUG):
                    input_path_f = futures[future] # Get input path associated with this future
                    try:
                        success = future.result() # Get return value (True/False) from worker
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                            # Warning should have been logged by worker
                    except Exception as exc:
                        # Catch exceptions raised within the worker process itself
                        failed_count += 1
                        logger.error(f"Worker process for {os.path.basename(input_path_f)} raised exception: {exc}", exc_info=True)
        except Exception as pool_err:
            # Catch errors related to the pool setup or management
            logger.error(f"Critical error occurred during process pool execution: {pool_err}", exc_info=True)

    logger.info(f"Batch processing finished. Succeeded: {processed_count}, Failed/Skipped: {failed_count}")


# --- Main Function ---
def main():
    """Parses arguments, loads config, initializes resources, and orchestrates processing."""
    parser = argparse.ArgumentParser(description="Herbarium-OCR: Layout analysis and OCR.")
    # Configuration file
    parser.add_argument("-c", "--config", metavar="PATH", help="Path to custom TOML configuration file.")
    # Core processing arguments
    parser.add_argument("--mode", choices=["pdf", "pdf_batch", "image", "image_batch"], required=True, help="Processing mode")
    parser.add_argument("--input", required=True, help="Path to input file or directory.")
    parser.add_argument("--model", required=True, help="Specify the OCR model (defined in config).") # Choices added dynamically
    # Optional parameter overrides
    parser.add_argument("--languages", default=None, help="Language hints/codes (e.g., en,ru).")
    parser.add_argument("--output_format", choices=["markdown", "md", "json", "xml", "html", "htm"], default=None, help="Generate additional output format ('json' = filtered).")
    parser.add_argument("--preprocess_images", action="store_true", default=None, help="Enable enhancement steps on cropped blocks.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    # Configure Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level); logger.setLevel(log_level)
    logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")

    # Load Configuration
    try:
        app_config = load_configuration(custom_config_path=args.config)
        app_config['_model_choice'] = args.model; app_config['_log_level'] = log_level;
    except Exception as cfg_err: logger.critical(f"Failed load config: {cfg_err}", exc_info=True); sys.exit(1)

    # Determine Runtime Settings
    ocr_cfg = app_config.get('OCR_CONFIG', {}); doc_cfg = app_config.get('DOCLAYOUT_CONFIG', {}); model_configs_final = app_config.get('MODEL_CONFIGS', {});
    # Update parser choices dynamically based on loaded config
    available_models = list(model_configs_final.keys())
    parser.set_defaults(model=args.model) # Keep user's choice
    for action in parser._actions:
        if action.dest == 'model': action.choices = available_models; break

    output_format_requested = args.output_format;
    if output_format_requested == "md": output_format_requested = "markdown";
    if output_format_requested == "htm": output_format_requested = "html"
    languages_str = args.languages if args.languages is not None else ocr_cfg.get("languages", ""); languages = [lang.strip() for lang in languages_str.split(',') if lang.strip()];
    attempt_rotation_config = ocr_cfg.get("attempt_auto_rotation", False);
    apply_enhancements_final = args.preprocess_images if args.preprocess_images is not None else ocr_cfg.get("preprocess_images", False);
    app_config['_apply_enhancements_run'] = apply_enhancements_final

    # Determine Target Device (No command-line override)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Auto-detected device for local models: '{target_device}'")
    app_config['_target_device'] = target_device # Store for workers

    # Validate Model Choice & Language Settings
    model_choice = args.model;
    if model_choice not in model_configs_final: logger.critical(f"Model '{model_choice}' not found in config."); sys.exit(1)
    model_config = model_configs_final[model_choice]; model_type = model_config.get("type");
    # Language validation is now implicitly handled by client initialization or usage
    logger.debug(f"Languages passed to client: {languages}")

    # Check API Keys
    required_env_vars = [];
    if model_type == "openai_compatible": required_env_vars.append(model_config.get("api_key_env"))
    elif model_type == "xfyun_http_ocr": required_env_vars.extend([model_config.get("app_id_env"), model_config.get("api_key_env"), model_config.get("api_secret_env")])
    required_env_vars = [v for v in required_env_vars if v]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars: logger.critical(f"Missing env var(s) for '{model_choice}': {', '.join(missing_vars)}"); sys.exit(1)
    if required_env_vars: logger.info(f"Env var check passed for '{model_choice}'.")
    else: logger.info(f"No specific env vars required for model type '{model_type}'.")

    # Initialize Tesseract (Main Process Check)
    if attempt_rotation_config:
        logger.info("Auto-rotation enabled. Initializing Tesseract...")
        _initialize_tesseract(app_config) # Pass config for path finding
        if not TESSERACT_AVAILABLE: logger.warning("Tesseract init failed. Rotation will be skipped.")
        else: logger.info("Tesseract initialized successfully.")
    else: logger.info("Auto-rotation disabled.")

    # --- Execute Processing ---
    start_time = time.time()
    logger.info(f"--- Starting Herbarium-OCR ---")
    logger.info(f"Mode: {args.mode}, Input: {args.input}")
    # Retrieve model_id from the loaded config for the chosen model
    model_id_to_log = model_config.get("model_id", "N/A")
    logger.info(f"OCR Model: {model_choice} ({model_id_to_log})")
    logger.info(f"Device (Auto-detected): {target_device}")
    logger.info(f"Output Format(s): _full.json" + (f" + .{output_format_requested}" if output_format_requested else ""))
    logger.info(f"Languages: {languages if languages else 'Default/None'}, Enhance Blocks: {apply_enhancements_final}, Attempt Full Rotation: {attempt_rotation_config}")

    try:
        input_path = Path(args.input).resolve()

        # --- Single File Mode ---
        if args.mode in ["pdf", "image"]:
            if not input_path.is_file(): raise FileNotFoundError(f"Input file not found: {input_path}")
            base_filename = input_path.stem; model_suffix = f"_{model_choice}" if model_choice else ""
            output_dir = input_path.parent # Output next to input
            output_base_path = output_dir / f"{base_filename}{model_suffix}"
            logger.info(f"Output base path: {output_base_path}")

            active_ocr_client = None; doclayout_model = None; all_results = []
            try:
                # Initialize client (pass target_device to local models)
                if model_type == "openai_compatible":
                    resize_dim = model_config.get("max_dimension", DEFAULT_RESIZE_MAX_DIMENSION)
                    jpeg_q = model_config.get("jpeg_quality", DEFAULT_JPEG_QUALITY)
                    active_ocr_client = OpenAICompatibleClient(api_key=os.getenv(model_config["api_key_env"]), base_url=model_config.get("base_url"), model_id=model_config.get("model_id"), rpm_limit=model_config.get("rpm_limit"), max_dimension=resize_dim, jpeg_quality=jpeg_q)
                elif model_type == "xfyun_http_ocr":
                     active_ocr_client = XFYunHttpOcrClient(app_id=os.getenv(model_config["app_id_env"]), api_key=os.getenv(model_config["api_key_env"]), api_secret=os.getenv(model_config["api_secret_env"]), api_url=model_config.get("api_url"), service_key=model_config.get("service_key"), param_type=model_config.get("param_type"), param_value=model_config.get("param_value"), rpm_limit=model_config.get("rpm_limit", 60), max_dimension=model_config.get("max_dimension", 1500), jpeg_quality=model_config.get("jpeg_quality", 85))
                elif model_type == "surya_ocr":
                    if 'SuryaOcrClient' not in globals() or SuryaOcrClient is None: raise ImportError("Surya OCR client requested but library not found.")
                    surya_device_cfg = model_config.get("device", target_device)
                    active_ocr_client = SuryaOcrClient(device=surya_device_cfg)
                else: raise ValueError(f"Unsupported type {model_type}")
                logger.info(f"Single mode: OCR client '{model_choice}' initialized.")

                # Load Layout Model
                doclayout_model_path = doc_cfg.get("DOCLAYOUT_MODEL_PATH");
                if not doclayout_model_path or not Path(doclayout_model_path).is_file(): raise FileNotFoundError("Layout model path invalid")
                doclayout_model = YOLOv10(str(doclayout_model_path));
                if not hasattr(doclayout_model, 'model'): raise RuntimeError("Invalid model structure.")
                logger.info(f"Single mode: Layout model loaded successfully.")

                # Process file
                if args.mode == "pdf":
                    doc = fitz.open(str(input_path))
                    logger.debug(f"Processing PDF '{input_path.name}' ({len(doc)} pages).")
                    for page_num in tqdm(range(len(doc)), desc=f"Processing {input_path.name}", unit="page"):
                        page = None
                        try:
                            page = doc.load_page(page_num)
                            dpi = ocr_cfg.get("dpi", 300); mat = fitz.Matrix(dpi/72, dpi/72); pix = page.get_pixmap(matrix=mat, alpha=False); img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            rotated_img = image_processor.auto_rotate_image(img, app_config) if attempt_rotation_config else img
                            all_results.extend(process_pdf_page(page, page_num + 1, doclayout_model, languages, apply_enhancements_final, active_ocr_client, app_config, rotated_img, target_device, None, None)) # Pass target_device
                        except Exception as page_err: logger.error(f"Error on page {page_num+1}: {page_err}", exc_info=True); all_results.append({"page": page_num + 1, "error": f"Page fail: {page_err}"})
                        finally: del page
                    doc.close()
                else: # args.mode == "image"
                    all_results = process_single_image(str(input_path), doclayout_model, languages, apply_enhancements_final, attempt_rotation_config, active_ocr_client, app_config, target_device, None, None) # Pass target_device

                # Generate outputs
                if all_results:
                     full_json_path = Path(f"{str(output_base_path)}_full.json")
                     generate_full_json_output(all_results, str(full_json_path), source_filename=str(input_path))
                     if output_format_requested:
                         requested_format_path = Path(f"{str(output_base_path)}.{output_format_requested}")
                         generator_map = {"markdown": generate_markdown_output, "json": generate_filtered_json_output, "xml": generate_xml_output, "html": generate_html_output}
                         gen_func = generator_map.get(output_format_requested)
                         if gen_func: gen_func(all_results, str(requested_format_path), source_filename=str(input_path))
                         else: logger.error(f"Invalid output format: '{output_format_requested}'")
                else: logger.warning(f"No results for {input_path}.")

            except Exception as single_err: logger.error(f"Failed processing single file {input_path}: {single_err}", exc_info=True)
            finally: del active_ocr_client; del doclayout_model # Cleanup

        # --- Batch Mode ---
        elif args.mode in ["pdf_batch", "image_batch"]:
            if not input_path.is_dir(): raise NotADirectoryError(f"Input directory not found: {input_path}")
            extensions = ('.pdf', '.djvu') if args.mode == "pdf_batch" else ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            process_batch_directory(str(input_path), extensions, output_format_requested, languages, app_config, target_device) # Pass target_device
        else: logger.critical(f"Invalid mode: {args.mode}")

    except (FileNotFoundError, NotADirectoryError) as e: logger.critical(str(e)); sys.exit(1)
    except KeyboardInterrupt: logger.warning("--- Processing interrupted by user ---")
    except Exception as main_err: logger.error(f"--- An unexpected error occurred: {main_err} ---", exc_info=True)
    finally: end_time = time.time(); logger.info(f"--- Herbarium-OCR finished in {end_time - start_time:.2f} seconds ---")

# --- Standard Python Entry Point ---
if __name__ == "__main__":
    # Set start method for multiprocessing BEFORE creating pool
    # Crucial for CUDA compatibility on Linux/macOS
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
           try: multiprocessing.set_start_method('spawn', force=True); logger.debug("Set multiprocessing start method to 'spawn'.")
           except (RuntimeError, ValueError) as e: logger.warning(f"Could not force start_method to 'spawn': {e}. Using default '{current_method}'. CUDA in batch mode might fail.")
    main()
# Main/image_processer.py
import copy
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps # Ensure ImageOps is imported
import os
import logging
from tqdm import tqdm
import argparse
import fitz
import io
from pathlib import Path
from typing import Dict, List
import time
import sys
from .config import load_configuration, OCR_CONFIG as DEFAULT_OCR_CONFIG

logger = logging.getLogger(__name__)

TESSERACT_AVAILABLE = False
pytesseract = None

def _initialize_tesseract(app_config: Dict):
    """Initializes Tesseract OCR using path from loaded config."""
    global TESSERACT_AVAILABLE, pytesseract
    if TESSERACT_AVAILABLE:
        return
    # --- Corrected Try/Except for Imports ---
    try:
        import pytesseract as pt_import
        pytesseract = pt_import
    except ImportError:
        logger.error("`pytesseract` not found (`pip install pytesseract`). Auto-rotation unavailable.")
        TESSERACT_AVAILABLE = False
        pytesseract = None
        return # Exit if import fails

    # --- Proceed only if import succeeded ---
    ocr_cfg = app_config.get('OCR_CONFIG', {})
    tesseract_path = ocr_cfg.get("tesseract_cmd_path", DEFAULT_OCR_CONFIG.get("tesseract_cmd_path"))

    if tesseract_path and os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logger.info(f"Using Tesseract: {tesseract_path}")
    elif tesseract_path:
         logger.warning(f"Configured Tesseract path not found: '{tesseract_path}'. Relying on system PATH.")

    # --- Corrected Try/Except for version check ---
    try:
        version = pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
        logger.info(f"Tesseract available (Version: {version}).")
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract not found in PATH or configured path.")
        TESSERACT_AVAILABLE = False
        pytesseract = None
    except Exception as e: # Catch other potential errors during init
        logger.error(f"Error during Tesseract initialization: {e}")
        TESSERACT_AVAILABLE = False
        pytesseract = None


class ImageProcessor:
    """Contains methods for image rotation and enhancement."""

    def auto_rotate_image(self, image: Image.Image, app_config: Dict) -> Image.Image:
        """Attempts to auto-rotate full image using Tesseract OSD."""
        if not TESSERACT_AVAILABLE or pytesseract is None:
            logger.warning("Skipping auto-rotation: Tesseract unavailable.")
            return image
        ocr_cfg = app_config.get('OCR_CONFIG', {})
        min_confidence = ocr_cfg.get("min_rotation_confidence", 60)
        logger.debug(f"Attempting rotation (min_conf: {min_confidence})")
        try:
            start_rot_time = time.time()
            open_cv_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(open_cv_bgr, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < 50 or gray.shape[1] < 50:
                logger.debug("Image too small for OSD.")
                return image
            osd_data = pytesseract.image_to_osd(gray, config='--psm 0', output_type=pytesseract.Output.DICT)
            angle = osd_data.get('rotate', 0)
            confidence = osd_data.get('orientation_conf', 0.0)
            if angle != 0 and confidence >= min_confidence:
                logger.info(f"Auto-rotating by {angle} degrees (Conf: {confidence:.2f})")
                (h, w) = gray.shape[:2]; center = (w // 2, h // 2); M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                rotated_bgr = cv2.warpAffine(open_cv_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                final_pil_image = Image.fromarray(cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB))
            else:
                logger.debug(f"No rotation applied (Angle: {angle}, Conf: {confidence:.2f})")
                final_pil_image = image
            # --- FIX: Separated log and return ---
            logger.debug(f"Rotation check took {time.time() - start_rot_time:.3f}s.")
            return final_pil_image
            # --- END FIX ---
        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract OSD failed: {e}")
            return image
        except Exception as e:
            logger.error(f"Error during rotation: {e}", exc_info=True)
            return image
    def enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Applies contrast enhancement (CLAHE)."""
        try:
            start_time = time.time()
            logger.debug("Applying contrast enhancement (CLAHE)...")
            # Convert PIL RGB to OpenCV BGR
            open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Convert to LAB color space
            lab = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            # Merge channels and convert back to BGR
            lab_clahe = cv2.merge([l_clahe, a, b])
            enhanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            # Convert back to PIL RGB
            processed_pil_image = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
            logger.debug(f"Contrast enhancement took {time.time() - start_time:.3f}s.")
            return processed_pil_image
        except Exception as e:
            logger.error(f"Error during contrast enhancement: {e}", exc_info=True)
            return image # Return original on error

    def denoise_image_color(self, image: Image.Image) -> Image.Image:
        """Applies fast Non-Local Means denoising to COLOR image."""
        try:
            start_time = time.time()
            logger.debug("Applying color denoising (fastNlMeansDenoisingColored)...")
            # Convert PIL RGB to OpenCV BGR
            open_cv_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Apply colored denoising
            denoised_bgr = cv2.fastNlMeansDenoisingColored(
                open_cv_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            )
            # Convert back to PIL RGB
            processed_pil_image = Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))
            logger.debug(f"Color Denoising took {time.time() - start_time:.3f}s.")
            return processed_pil_image
        except Exception as e:
            logger.error(f"Error during color denoising: {e}", exc_info=True)
            return image # Return original on error

    def denoise_image_gray(self, image: Image.Image) -> Image.Image:
        """Applies fast Non-Local Means denoising to GRAYSCALE image."""
        try:
            start_time = time.time()
            logger.debug("Applying grayscale denoising (fastNlMeansDenoising)...")
            # Convert PIL RGB to OpenCV Grayscale
            open_cv_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            # Apply grayscale denoising
            h_gray = 10 # Denoising strength parameter
            denoised_gray = cv2.fastNlMeansDenoising(
                open_cv_gray, None, h=h_gray, templateWindowSize=7, searchWindowSize=21
            )
            # Convert back to RGB PIL Image for consistency
            processed_pil_image = Image.fromarray(denoised_gray).convert('RGB')
            logger.debug(f"Grayscale Denoising took {time.time() - start_time:.3f}s.")
            return processed_pil_image
        except Exception as e:
            logger.error(f"Error during grayscale denoising: {e}", exc_info=True)
            return image # Return original on error

    def sharpen_image(self, image: Image.Image) -> Image.Image:
        """Applies a basic sharpening filter."""
        try:
            start_time = time.time()
            logger.debug("Applying sharpening...")
            # Convert PIL RGB to OpenCV BGR
            open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Simple sharpening using Gaussian blur subtraction (Unsharp Masking principle)
            blurred = cv2.GaussianBlur(open_cv_image, (0, 0), 3)
            sharpened = cv2.addWeighted(open_cv_image, 1.5, blurred, -0.5, 0)
            # Convert back to PIL RGB
            processed_pil_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
            logger.debug(f"Sharpening took {time.time() - start_time:.3f}s.")
            return processed_pil_image
        except Exception as e:
            logger.error(f"Error during sharpening: {e}", exc_info=True)
            return image # Return original on error

# --- Helper function for main_test to process a single file ---
def _process_single_file(
    input_file_path: Path,
    output_dir_for_file: Path,
    processor: ImageProcessor,
    app_config: Dict,
    is_batch_run: bool
) -> bool:
    """
    Processes a single image or PDF file for the test script.
    Applies rotation attempt and all configured enhancements.
    Saves output to output_dir_for_file.
    """
    base_filename = input_file_path.stem
    output_filename_stem = base_filename if is_batch_run else f"{base_filename}_preprocessed_test"
    file_ext = input_file_path.suffix.lower()
    processed_output_path_str = ""
    file_processed_successfully = False
    start_time_file = time.time()

    logger.info(f"Test processing: {input_file_path} -> {output_dir_for_file / output_filename_stem}.<ext>")

    ocr_cfg_for_test = app_config.get('OCR_CONFIG', {})
    # Respect denoise_mode from the loaded config for the test
    denoise_mode_for_test = ocr_cfg_for_test.get("denoise_mode", DEFAULT_OCR_CONFIG.get("denoise_mode", "gray"))
    do_contrast_test = ocr_cfg_for_test.get('enhance_contrast', True) # Assume test applies if key exists and True
    do_denoise_test = ocr_cfg_for_test.get('denoise', True)
    do_sharpen_test = ocr_cfg_for_test.get('sharpen', True)


    if file_ext in ['.pdf', '.djvu']:
        output_path = output_dir_for_file / (output_filename_stem + ".pdf")
        output_doc = fitz.open(); doc = None
        try:
            doc = fitz.open(str(input_file_path))
            dpi = ocr_cfg_for_test.get("dpi", DEFAULT_OCR_CONFIG.get("dpi",300))
            mat = fitz.Matrix(dpi/72, dpi/72)
            for page_num in range(len(doc)):
                logger.debug(f"  Processing test page {page_num + 1}/{len(doc)} of {input_file_path.name}...")
                page = doc.load_page(page_num); pix = page.get_pixmap(matrix=mat, alpha=False); img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                current_img = processor.auto_rotate_image(img, app_config) # Attempt rotation
                logger.debug(f"  Applying enhancements for test page (Contrast:{do_contrast_test}, Denoise:{do_denoise_test} (mode:{denoise_mode_for_test}), Sharpen:{do_sharpen_test})...")
                if do_contrast_test: current_img = processor.enhance_contrast(current_img)
                if do_denoise_test:
                    if denoise_mode_for_test == "color": current_img = processor.denoise_image_color(current_img)
                    elif denoise_mode_for_test == "gray": current_img = processor.denoise_image_gray(current_img)
                    else: logger.warning(f"Unknown denoise_mode '{denoise_mode_for_test}' in test. Skipping denoise.")
                if do_sharpen_test: current_img = processor.sharpen_image(current_img)

                img_bytes_io = io.BytesIO(); current_img.save(img_bytes_io, format='PNG'); img_bytes_io.seek(0)
                page_rect = fitz.Rect(0, 0, current_img.width, current_img.height)
                new_page = output_doc.new_page(width=page_rect.width, height=page_rect.height)
                new_page.insert_image(page_rect, stream=img_bytes_io.read())
            if len(output_doc) > 0: output_doc.save(str(output_path)); processed_output_path_str = str(output_path)
        except Exception as e: logger.error(f"Error processing PDF/DjVu {input_file_path.name}: {e}", exc_info=True)
        finally:
            if doc: doc.close()
            if output_doc: output_doc.close()
    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        output_path = output_dir_for_file / (output_filename_stem + ".png")
        try:
            img = Image.open(input_file_path); img = ImageOps.exif_transpose(img); img_rgb = img.convert("RGB")
            current_img = processor.auto_rotate_image(img_rgb, app_config) # Attempt rotation
            logger.info(f"Applying enhancements for test image (Contrast:{do_contrast_test}, Denoise:{do_denoise_test} (mode:{denoise_mode_for_test}), Sharpen:{do_sharpen_test})...")
            if do_contrast_test: current_img = processor.enhance_contrast(current_img)
            if do_denoise_test:
                if denoise_mode_for_test == "color": current_img = processor.denoise_image_color(current_img)
                elif denoise_mode_for_test == "gray": current_img = processor.denoise_image_gray(current_img)
                else: logger.warning(f"Unknown denoise_mode '{denoise_mode_for_test}' in test. Skipping denoise.")
            if do_sharpen_test: current_img = processor.sharpen_image(current_img)
            current_img.save(str(output_path), format="PNG")
            processed_output_path_str = str(output_path)
        except Exception as e: logger.error(f"Failed processing image {input_file_path.name}: {e}", exc_info=True)
    else:
        logger.warning(f"Unsupported file type for testing: {input_file_path.name}")
        return False

    if processed_output_path_str:
        logger.info(f"Test output for {input_file_path.name} saved to: {processed_output_path_str} (took {time.time() - start_time_file:.2f}s)")
        file_processed_successfully = True
    else:
        logger.warning(f"Test did not produce an output for {input_file_path.name}")
    return file_processed_successfully

# --- Testing Function ---
def main_test():
    """Entry point for testing the image preprocessing pipeline via command line."""
    parser = argparse.ArgumentParser(
        description="Test Image Preprocessing Pipeline (Rotation + All Enhancements)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to input image or PDF.")
    # Add config argument consistent with other scripts
    parser.add_argument("-c", "--config", metavar="PATH", help="Path to custom TOML configuration file (used for Tesseract path).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    # Setup logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if not logging.getLogger().hasHandlers(): # Configure root logger if not done yet
         logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
    else: # Set level if already configured
         logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level) 

    input_p = Path(args.input).resolve()

    logger.info("--- Starting Preprocessing Test/Utility ---")
    try:
        app_config = load_configuration(custom_config_path=args.config)
    except Exception as cfg_err:
        logger.error(f"Failed to load config: {cfg_err}. Using defaults for OCR_CONFIG.", exc_info=True)
        app_config = {'OCR_CONFIG': copy.deepcopy(DEFAULT_OCR_CONFIG)}

    logger.info("NOTE: This utility always attempts Tesseract initialization for rotation capability check.")
    _initialize_tesseract(app_config)
    if not TESSERACT_AVAILABLE:
        logger.warning("Tesseract init failed/skipped. Rotation will not be applied in outputs.")

    processor = ImageProcessor()
    files_to_process: List[Path] = []
    output_directory_base: Path
    # Hardcoded output subdirectory name for batch mode
    batch_output_subdir_name = "herbariumOCR_preprocessed"

    if input_p.is_file():
        files_to_process.append(input_p)
        output_directory_base = input_p.parent # Output next to single input file
        logger.info(f"Single file mode. Output directory: {output_directory_base}")
    elif input_p.is_dir():
        output_directory_base = input_p / batch_output_subdir_name # Use fixed name
        logger.info(f"Batch mode. Scanning directory: {input_p}")
        logger.info(f"Batch output will be in: {output_directory_base}")
        try:
            output_directory_base.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Failed to create batch output directory '{output_directory_base}': {e}")
            sys.exit(1)
        supported_extensions = {'.pdf', '.djvu', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        for item in input_p.iterdir():
            if item.is_file() and item.suffix.lower() in supported_extensions:
                files_to_process.append(item)
        if not files_to_process:
            logger.warning(f"No supported files found in directory: {input_p}"); sys.exit(0)
        logger.info(f"Found {len(files_to_process)} files for batch preprocessing.")
    else:
        logger.critical(f"Input path is not a valid file or directory: {args.input}"); sys.exit(1)

    overall_start_time = time.time()
    success_count = 0; fail_count = 0

    for file_path in tqdm(files_to_process, desc="Preprocessing Files", unit="file", disable=len(files_to_process) <= 1 or logger.level > logging.INFO) :
        if _process_single_file(
            file_path,
            output_directory_base,
            processor,
            app_config,
            is_batch_run=input_p.is_dir()
        ):
            success_count +=1
        else:
            fail_count +=1

    logger.info(f"--- Preprocessing Test/Utility Finished in {time.time() - overall_start_time:.2f}s ---")
    logger.info(f"Total files processed: {len(files_to_process)}. Succeeded: {success_count}. Failed: {fail_count}.")
    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main_test()
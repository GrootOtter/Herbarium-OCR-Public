# Main/image_processer.py

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps # Ensure ImageOps is imported
import os
import logging
import argparse
import fitz
import io
from pathlib import Path
from typing import Dict
from .config import OCR_CONFIG
import time
import sys
from . import config as default_config_module
from .herbarium_ocr import load_configuration

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
    tesseract_path = ocr_cfg.get("tesseract_cmd_path", OCR_CONFIG.get("tesseract_cmd_path"))

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

# --- Testing Function ---
def main_test(input_path: str):
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

    input_p = Path(args.input).resolve() # Resolve input path
    if not input_p.is_file():
        logger.critical(f"Input file not found: {args.input}")
        sys.exit(1)

    # Determine output path (next to input file)
    output_dir = input_p.parent
    base_filename = input_p.stem
    output_base_name = f"{base_filename}_preprocessed_test"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Failed ensure output dir '{output_dir}': {e}")
        sys.exit(1)

    logger.info("--- Starting Preprocessing Test ---")
    # Load config primarily to get Tesseract path if set by user
    try:
        app_config = load_configuration(default_config_module, custom_config_path=args.config)
    except Exception as cfg_err:
        logger.error(f"Failed to load config: {cfg_err}. Tesseract path might be incorrect if not in PATH.", exc_info=True)
        # Use default OCR_CONFIG as fallback if loading fails
        app_config = {'OCR_CONFIG': getattr(default_config_module, 'OCR_CONFIG', {})}

    # Initialize Tesseract (always attempted in test)
    logger.info("NOTE: Test function always attempts Tesseract initialization.")
    _initialize_tesseract(app_config)
    if not TESSERACT_AVAILABLE:
        logger.warning("Tesseract initialization failed/skipped. Rotation will not be applied in test output.")

    processor = ImageProcessor()
    file_ext = input_p.suffix.lower()

    try:
        processed_output = None
        start_time = time.time()

        if file_ext in ['.pdf', '.djvu']:
            output_path = output_dir / (output_base_name + ".pdf")
            logger.info(f"Processing document '{args.input}' -> '{output_path}'")
            # ... (Keep PDF processing logic from previous version, using app_config for DPI/Tesseract) ...
            output_doc = fitz.open(); doc = None
            try:
                doc = fitz.open(str(input_p)); dpi = app_config.get('OCR_CONFIG',{}).get("dpi", 300); mat = fitz.Matrix(dpi/72, dpi/72)
                for page_num in range(len(doc)):
                    logger.debug(f"  Processing test page {page_num + 1}/{len(doc)}...")
                    page = doc.load_page(page_num); pix = page.get_pixmap(matrix=mat, alpha=False); img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    rotated_img = processor.auto_rotate_image(img, app_config) # Attempt rotation
                    processed_page_img = rotated_img
                    logger.debug("  Applying ALL enhancement steps (default gray denoise) for test page...")
                    processed_page_img = processor.enhance_contrast(processed_page_img)
                    processed_page_img = processor.denoise_image_gray(processed_page_img) # Use gray for speed test
                    processed_page_img = processor.sharpen_image(processed_page_img)
                    try:
                        img_bytes = io.BytesIO(); processed_page_img.save(img_bytes, format='PNG'); img_bytes.seek(0)
                        page_rect = fitz.Rect(0, 0, processed_page_img.width, processed_page_img.height)
                        new_page = output_doc.new_page(width=page_rect.width, height=page_rect.height)
                        new_page.insert_image(page_rect, stream=img_bytes.read())
                    except Exception as insert_err: logger.error(f"Error inserting test page {page_num + 1}: {insert_err}", exc_info=True)
            finally:
                 if doc: doc.close()
            if len(output_doc) > 0: output_doc.save(str(output_path)); processed_output = output_path; logger.info(f"Test PDF with {len(output_doc)} pages saved.")
            else: logger.warning("No pages added to test PDF.")
            output_doc.close()

        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            output_path = output_dir / (output_base_name + ".png")
            logger.info(f"Processing image '{args.input}' -> '{output_path}'")
            try:
                img = Image.open(str(input_p))
                img = ImageOps.exif_transpose(img) # Apply EXIF
                img_rgb = img.convert("RGB")
                rotated_img = processor.auto_rotate_image(img_rgb, app_config) # Attempt rotation
                processed_img = rotated_img
                logger.info("Applying ALL enhancement steps (default gray denoise) for test...")
                processed_img = processor.enhance_contrast(processed_img)
                processed_img = processor.denoise_image_gray(processed_img) # Use gray for speed test
                processed_img = processor.sharpen_image(processed_img)
                processed_img.save(str(output_path), format="PNG")
                processed_output = output_path
            except Exception as e: logger.error(f"Failed processing test image '{args.input}': {e}", exc_info=True)
        else:
            logger.warning(f"Unsupported file type for testing: {file_ext}")

        if processed_output: logger.info(f"Test finished in {time.time() - start_time:.3f}s. Output: {processed_output}")
        else: logger.warning(f"Test did not produce output for {args.input}.")
    except Exception as e:
        logger.error(f"Error during preprocess_test '{args.input}': {e}", exc_info=True)
    finally:
        logger.info("--- Finished Preprocessing Test ---")

# --- Standard Python Entry Point ---
# This now calls the main_test function if the script is run directly
if __name__ == "__main__":
    main_test()
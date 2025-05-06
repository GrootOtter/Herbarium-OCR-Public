# Main/check_layout_model.py

import logging
import os
import sys
import torch
from doclayout_yolo import YOLOv10
import argparse 
import toml 
from pathlib import Path 
import copy 

# --- Relative Imports ---
# Import default config values and configuration loading logic
from . import config as default_config_module
from .herbarium_ocr import load_configuration # Reuse the loader from main script

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__) # Use named logger

# --- Main Execution Function ---
def main_check():
    """Loads config, checks layout model, and prints supported classes."""
    parser = argparse.ArgumentParser(description="Check DocLayout-YOLO model's supported classes.")
    # Add config argument, consistent with main script
    parser.add_argument("-c", "--config", metavar="PATH", help="Path to custom TOML configuration file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Load configuration using the shared function
    try:
        app_config = load_configuration(default_config_module, custom_config_path=args.config)
        doc_cfg = app_config.get("DOCLAYOUT_CONFIG", {})
        doclayout_model_path = doc_cfg.get("DOCLAYOUT_MODEL_PATH")
        if not doclayout_model_path:
             raise ValueError("DOCLAYOUT_MODEL_PATH not found in configuration.")
    except Exception as cfg_err:
        logger.critical(f"Failed to load configuration or find model path: {cfg_err}", exc_info=True)
        sys.exit(1)

    # Determine device (simple auto-detect for this script)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {target_device}")

    # Check if model file exists
    if not os.path.exists(doclayout_model_path):
        logger.error(f"Error: DocLayout-YOLO model file not found at {doclayout_model_path}")
        logger.error("Please check the 'DOCLAYOUT_MODEL_PATH' in your configuration.")
        exit(1)

    try:
        # Load the layout model
        logger.info(f"Loading layout model: {doclayout_model_path}")
        model = YOLOv10(doclayout_model_path) # YOLOv10 should handle device internally or use predict arg
        logger.info("Layout model loaded successfully.")

        # Access and display supported class names
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            class_names_dict = model.model.names
            class_names_list = list(class_names_dict.values())
            logger.info(f"Model-supported category names: {class_names_list}")
            logger.info(f"Full class mapping (ID: Name): {class_names_dict}")
            # Provide guidance for configuration
            print("\nPlease update the `RELEVANT_TEXT_CLASSES` list in your configuration")
            print("(Main/config.py or herbarium_ocr_config.toml) based on the names above.")
            print("Example: RELEVANT_TEXT_CLASSES = ['title', 'plain text', 'figure']")
        else:
            logger.warning("Failed to access model's class names attribute (expected model.model.names).")

    except Exception as e:
        logger.error(f"Error occurred during model loading or inspection: {e}", exc_info=True)
        exit(1)

# --- Standard Python Entry Point ---
# Ensures the main logic runs only when the script is executed directly
if __name__ == "__main__":
    main_check() # Call the main checking function
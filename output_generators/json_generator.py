# output_generators/json_generator.py

import json
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path # Use pathlib

logger = logging.getLogger(__name__)

# Renamed to reflect its purpose
def generate_full_json_output(results: List[Dict], output_path: str, source_filename: Optional[str] = None):
    """
    Generates a JSON file containing the complete, unfiltered OCR results.

    :param results: List of dictionaries with all extracted data.
    :param output_path: Path to save the generated JSON file.
    :param source_filename: Optional path to the original source file.
    """
    logger.info(f"Generating full JSON output to: {output_path}")
    output_p = Path(output_path)

    output_data = {
        "source_file": Path(source_filename).name if source_filename else "Not Provided",
        "results": results # Store the complete results
    }

    try:
        output_p.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with output_p.open('w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Full JSON file generated successfully: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write full JSON to {output_path}: {e}", exc_info=True)

# --- New Function for Filtered JSON ---
def generate_filtered_json_output(results: List[Dict], output_path: str, source_filename: Optional[str] = None):
    """
    Generates a JSON file excluding box and confidence keys for simpler use cases.

    :param results: List of dictionaries with all extracted data.
    :param output_path: Path to save the generated filtered JSON file.
    :param source_filename: Optional path to the original source file.
    """
    logger.info(f"Generating filtered JSON output to: {output_path}")
    output_p = Path(output_path)
    filtered_results = []

    for result in results:
        # Create a new dict omitting specific keys
        filtered_item = {
            key: value for key, value in result.items()
            if key not in ['box', 'confidence'] # Keys to exclude
        }
        # Keep item if it still has meaningful content (text or error)
        if 'text' in filtered_item or 'error' in filtered_item or \
           'file_error' in filtered_item or 'image_error' in filtered_item:
             filtered_results.append(filtered_item)
        else:
             logger.debug(f"Skipping empty result item after filtering: {result}")

    output_data = {
        "source_file": Path(source_filename).name if source_filename else "Not Provided",
        "results": filtered_results
    }
    try:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with output_p.open('w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Filtered JSON file generated successfully: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write filtered JSON to {output_path}: {e}", exc_info=True)
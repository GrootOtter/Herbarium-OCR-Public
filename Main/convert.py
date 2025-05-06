# Main/convert.py

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Import generator functions from the centralized location
from output_generators import (generate_markdown_output, generate_xml_output,
                               generate_html_output, generate_filtered_json_output)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("convert_format")

def process_single_json_file(input_json_path: Path, formats_to_generate: set):
    """Loads a single _full.json file and converts it to requested formats."""
    logger.info(f"Processing input file: {input_json_path}")

    # Determine output directory and base name
    output_dir = input_json_path.parent
    output_base_name = input_json_path.stem.replace("_full", "") # Remove suffix

    # Load the full JSON data
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "results" not in data or not isinstance(data.get("results"), list):
             logger.error(f"Invalid format in {input_json_path}: Missing 'results' list.")
             return False # Indicate failure
        results = data.get("results", [])
        source_file = data.get("source_file", output_base_name) # Use original source or derived name
        logger.debug(f"Loaded {len(results)} results from {input_json_path} (Source: {source_file})")
    except Exception as e:
        logger.error(f"Failed read/decode {input_json_path}: {e}")
        return False # Indicate failure

    if not results:
        logger.warning(f"Input JSON file {input_json_path} contains no results. Skipping conversion.")
        return True # Considered success as there's nothing to convert

    # --- Generate requested formats ---
    generation_map = {
        'markdown': (generate_markdown_output, ".md"),
        'html': (generate_html_output, ".html"),
        'xml': (generate_xml_output, ".xml"),
        'filtered_json': (generate_filtered_json_output, ".json") # Filtered output uses .json extension
    }
    success = True
    for fmt_key, (gen_func, ext) in generation_map.items():
        if fmt_key in formats_to_generate:
            output_file_path = output_dir / f"{output_base_name}{ext}"
            try:
                # Call the appropriate generator function
                gen_func(results, str(output_file_path), source_filename=source_file)
            except Exception as e:
                logger.error(f"Failed to generate {fmt_key.upper()} output to {output_file_path}: {e}", exc_info=True)
                success = False # Mark overall success as False if any conversion fails

    return success


def main():
    """Main function for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert Herbarium-OCR full JSON output (*_full.json) to other formats (Markdown, HTML, XML, filtered JSON)."
    )
    # Input argument: Can be a single file or a directory
    parser.add_argument(
        "input_path",
        help="Path to a single *_full.json file OR a directory containing *_full.json files."
    )
    # Output format selection (removed 'all')
    parser.add_argument(
        "--to",
        nargs='+', # One or more formats
        choices=['markdown', 'md', 'html', 'htm', 'xml', 'json'], # Excluded 'all'
        required=True,
        help="Output format(s) to generate. 'json' creates a filtered version."
    )
    # Optional output directory (REMOVED - output is always alongside input)
    # parser.add_argument("-o", "--output_dir", ...)
    # Verbosity flag
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging."
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    logger.debug(f"Arguments received: {args}")

    input_path = Path(args.input_path).resolve() # Resolve to absolute path

    # Determine which files to process
    files_to_process = []
    if input_path.is_file():
        if input_path.name.endswith("_full.json"):
            files_to_process.append(input_path)
        else:
            logger.critical(f"Input file must end with '_full.json'. Provided: {input_path}")
            sys.exit(1)
    elif input_path.is_dir():
        logger.info(f"Scanning directory for *_full.json files: {input_path}")
        found_files = list(input_path.glob("*_full.json"))
        if not found_files:
            logger.warning(f"No '*_full.json' files found in directory: {input_path}")
            sys.exit(0) # Exit cleanly if no files found
        files_to_process.extend(found_files)
        logger.info(f"Found {len(files_to_process)} files to convert.")
    else:
        logger.critical(f"Input path is not a valid file or directory: {input_path}")
        sys.exit(1)

    # Determine formats to generate (handle aliases)
    formats_to_generate = set()
    for fmt in args.to:
        if fmt in ['markdown', 'md']: formats_to_generate.add('markdown')
        elif fmt in ['html', 'htm']: formats_to_generate.add('html')
        elif fmt == 'xml': formats_to_generate.add('xml')
        elif fmt == 'json': formats_to_generate.add('filtered_json') # Map 'json' arg to filtered

    if not formats_to_generate:
        logger.error("No valid output formats specified with --to argument.")
        sys.exit(1)

    logger.info(f"Target conversion formats: {', '.join(formats_to_generate)}")

    # Process each file
    success_count = 0
    fail_count = 0
    for json_file_path in files_to_process:
        if process_single_json_file(json_file_path, formats_to_generate):
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"Conversion process finished. Succeeded: {success_count}, Failed: {fail_count}")
    if fail_count > 0:
        sys.exit(1) # Exit with error code if any conversion failed

# Standard entry point guard
if __name__ == "__main__":
    main()
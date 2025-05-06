# output_generators/markdown_generator.py

import logging
import os
from typing import List, Dict, Optional

# Get logger instance. The calling script (herbarium_ocr.py) should configure the root logger.
logger = logging.getLogger(__name__)

def generate_markdown_output(results: List[Dict], output_path: str, source_filename: Optional[str] = None):
    """
    Generates a structured Markdown file from OCR results, applying formatting based on block types.
    Includes page markers and source file information for traceability.
    Detailed block info (like coordinates) is logged at DEBUG level.

    :param results: List of dictionaries, each containing OCR results for a block (with 'page', 'class', 'box', 'text' or 'error').
    :param output_path: Path to save the generated Markdown file.
    :param source_filename: Optional path to the original source file for inclusion in the output header.
    """
    logger.info(f"Generating structured Markdown output to: {output_path}")
    page_groups = {}
    markdown_content = []

    # Add source file comment at the beginning
    if source_filename:
        # Use os.path.basename to just get the filename, not the full path
        markdown_content.append(f"<!-- Source File: {os.path.basename(source_filename)} -->\n")
    else:
        markdown_content.append("<!-- Source File: Not Provided -->\n")

    # Group results by page first
    for result in results:
        page_num = result.get("page", "N/A") # Use "N/A" for results without a page number (e.g., single image errors)
        page_groups.setdefault(page_num, []).append(result)

    # Sort pages numerically, placing "N/A" or non-integer keys last
    page_keys = sorted([k for k in page_groups if isinstance(k, int)])
    page_keys.extend([k for k in page_groups if not isinstance(k, int)]) # Add non-integer keys like "N/A"

    for page_num in page_keys:
        # Add page marker
        if isinstance(page_num, int):
             markdown_content.append(f"<!-- Page {page_num} -->\n")
        else:
             markdown_content.append(f"<!-- Content Section / Errors -->\n") # Handle non-page sections

        formatted_page_content = []
        for result in page_groups[page_num]:
            # Log detailed info for debugging before generating potentially simplified output
            # Include box info in the debug log
            logger.debug(f"Processing block for Markdown: Class='{result.get('class', 'N/A')}', Box={result.get('box', 'N/A')}, Page={page_num}")

            formatted_text = None # Initialize for clarity

            if "error" in result:
                # Keep error messages clear, include context if available
                formatted_text = f"<!-- ERROR: {result['error']} (Class: {result.get('class', 'N/A')}, Box: {result.get('box', 'N/A')}) -->"
            elif "text" in result and result['text'] and result['text'] != "<!-- OCR FAILED -->":
                block_class = result.get('class', 'unknown')
                text = result['text'].strip() # Ensure text is stripped

                # Apply formatting based on class, keeping comments simple
                if block_class == 'title':
                    formatted_text = f"## {text}"
                elif block_class == 'plain text':
                    formatted_text = text
                elif block_class == 'table':
                    # Comment notes detection, raw text follows
                    formatted_text = f"<!-- Table Block Detected -->\n\n{text}"
                elif block_class == 'figure':
                     # Comment notes detection, raw text follows (important for labels)
                    formatted_text = f"<!-- Figure Content/Text Detected -->\n\n{text}"
                elif block_class == 'figure_caption':
                    formatted_text = f"*Figure Caption: {text}*"
                elif block_class == 'table_caption':
                    formatted_text = f"*Table Caption: {text}*"
                elif block_class == 'table_footnote':
                    formatted_text = f"*Table Footnote: {text}*"
                elif block_class == 'isolate_formula':
                    # Using code block is reasonable for formulas
                    formatted_text = f"<!-- Isolated Formula Detected -->\n\n```\n{text}\n```"
                elif block_class == 'formula_caption':
                    formatted_text = f"*Formula Caption: {text}*"
                # Add elif conditions for other specific classes if needed in the future
                else: # Default for unknown or unhandled classes
                    formatted_text = f"<!-- Block Type: {block_class} -->\n\n{text}"

            elif result.get('text') == "<!-- OCR FAILED -->":
                 # Provide context for failed OCR attempts
                 formatted_text = f"<!-- OCR FAILED (Class: {result.get('class', 'N/A')}, Box: {result.get('box', 'N/A')}) -->"

            # Append the formatted text if it was generated
            if formatted_text is not None:
                formatted_page_content.append(formatted_text)

        # Join formatted blocks for the current page with double newlines
        markdown_content.append("\n\n".join(formatted_page_content))

        # Add page break marker between pages/sections, but not after the very last one
        if page_num != page_keys[-1]: # Check if it's not the last key
             if isinstance(page_num, int):
                 markdown_content.append("\n\n<!-- === Page Break === -->\n")
             else:
                 markdown_content.append("\n\n<!-- === Section Break === -->\n") # Differentiate if needed


    # Join all parts (header, pages, breaks) into the final string
    final_markdown = "\n".join(markdown_content).strip()

    # Write to file
    try:
        # Ensure output directory exists just in case
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path: # Avoid error if output_path is just a filename in current dir
             os.makedirs(output_dir_path, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        logger.info(f"Structured Markdown file generated successfully: {output_path}")
    except IOError as e:
        logger.error(f"Failed to write structured Markdown to {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Markdown file writing: {e}", exc_info=True)
# output_generators/html_generator.py

import logging
import os
from typing import List, Dict, Optional
import html # For escaping text

logger = logging.getLogger(__name__)

def generate_html_output(results: List[Dict], output_path: str, source_filename: Optional[str] = None):
    """
    Generates an HTML file from OCR results, applying basic formatting.

    :param results: List of dictionaries containing OCR results.
    :param output_path: Path to save the generated HTML file.
    :param source_filename: Optional path to the original source file.
    """
    logger.info(f"Generating HTML output to: {output_path}")

    source_base_name = os.path.basename(source_filename) if source_filename else "OCR Results"

    # Basic HTML structure
    html_content = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        f'  <title>{html.escape(source_base_name)}</title>',
        '  <style>',
        '    body { font-family: sans-serif; line-height: 1.6; margin: 2em; }',
        '    h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }',
        '    .page-marker, .error-marker, .block-comment { color: #888; font-style: italic; margin-top: 1.5em; margin-bottom: 0.5em; font-size: 0.9em;}',
        '    .block { margin-bottom: 1em; padding: 0.5em; border-left: 3px solid #eee; }',
        '    .block-title { font-weight: bold; font-size: 1.2em; margin-bottom: 0.5em; }', # Style for h2
        '    .block-caption { font-style: italic; color: #555; }', # Style for captions
        '    .block-code { background-color: #f5f5f5; padding: 0.5em; border: 1px solid #ddd; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; }', # Style for formulas
        '    .block-table-figure { background-color: #f9f9f9; padding: 0.5em; border: 1px dashed #ccc; }', # Style for tables/figures
        '    .error-text { color: red; font-weight: bold; }',
        '    hr { border: 0; height: 1px; background: #ccc; margin: 2em 0; }', # Page break style
        '  </style>',
        "</head>",
        "<body>",
        f'  <h1>OCR Results for: {html.escape(source_base_name)}</h1>'
    ]

    current_page = None
    first_page = True

    for result in results:
        page = result.get("page", "N/A")

        # Add page break if page number changes
        if page != current_page:
            if not first_page:
                 html_content.append('<hr>') # Page break
            html_content.append(f'<p class="page-marker">Page {page}</p>')
            current_page = page
            first_page = False

        # Start block container (optional, for styling)
        html_content.append('<div class="block">')

        block_class = result.get('class', 'unknown')
        escaped_text = html.escape(result.get("text", "")).strip()
        error_text = html.escape(result.get("error", result.get("image_error", result.get("file_error", ""))))

        # Apply formatting based on class
        if error_text:
            html_content.append(f'  <p class="error-marker">Error (Class: {html.escape(block_class)})</p>')
            html_content.append(f'  <p class="error-text">{error_text}</p>')
        elif escaped_text == "<!-- OCR FAILED -->": # Check escaped version
             html_content.append(f'  <p class="error-marker">OCR Failed (Class: {html.escape(block_class)})</p>')
        elif escaped_text:
            if block_class == 'title':
                html_content.append(f'  <h2 class="block-title">{escaped_text}</h2>')
            elif block_class == 'plain text':
                html_content.append(f'  <p>{escaped_text}</p>')
            elif block_class in ['figure_caption', 'table_caption', 'formula_caption', 'table_footnote']:
                 caption_type = block_class.replace('_', ' ').title()
                 html_content.append(f'  <p class="block-caption"><i>{html.escape(caption_type)}: {escaped_text}</i></p>')
            elif block_class == 'isolate_formula':
                  html_content.append(f'  <p class="block-comment"><!-- Isolated Formula Detected --></p>')
                  html_content.append(f'  <pre class="block-code"><code>{escaped_text}</code></pre>')
            elif block_class == 'table':
                  html_content.append(f'  <p class="block-comment"><!-- Table Block Detected --></p>')
                  # Output table text preformatted for basic structure preservation
                  html_content.append(f'  <pre class="block-table-figure"><code>{escaped_text}</code></pre>')
            elif block_class == 'figure':
                  html_content.append(f'  <p class="block-comment"><!-- Figure Content/Text Detected --></p>')
                  # Output figure text preformatted
                  html_content.append(f'  <pre class="block-table-figure"><code>{escaped_text}</code></pre>')
            else: # Default for unknown or other classes
                html_content.append(f'  <p class="block-comment"><!-- Block Type: {html.escape(block_class)} --></p>')
                html_content.append(f'  <p>{escaped_text}</p>')

        # End block container
        html_content.append('</div>')


    html_content.append("</body>")
    html_content.append("</html>")

    # Write to file
    try:
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path:
            os.makedirs(output_dir_path, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        logger.info(f"HTML file generated successfully: {output_path}")

    except IOError as e:
        logger.error(f"Failed to write HTML to {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during HTML generation/writing: {e}", exc_info=True)
# output_generators/__init__.py
import logging

from .markdown_generator import generate_markdown_output
# Rename import and add new one
from .json_generator import generate_full_json_output, generate_filtered_json_output
from .xml_generator import generate_xml_output
from .html_generator import generate_html_output

# Update exports
__all__ = [
    "generate_markdown_output",
    "generate_full_json_output",    # Renamed
    "generate_filtered_json_output",# Added
    "generate_xml_output",
    "generate_html_output",
]
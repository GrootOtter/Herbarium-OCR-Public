# output_generators/xml_generator.py

import logging
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom # For pretty printing
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def _dict_to_xml_str(data_dict: Dict) -> str:
    """Helper to convert simple dict (like box coords) to string for XML."""
    # Example: {'x1': 10, 'y1': 20, ...} -> "x1=10, y1=20, ..."
    # Or just return str(data_dict) for simplicity
    return str(data_dict)

def generate_xml_output(results: List[Dict], output_path: str, source_filename: Optional[str] = None):
    """
    Generates an XML file from OCR results.

    :param results: List of dictionaries containing OCR results.
    :param output_path: Path to save the generated XML file.
    :param source_filename: Optional path to the original source file.
    """
    logger.info(f"Generating XML output to: {output_path}")

    # Create root element
    root = ET.Element("Document")
    if source_filename:
        root.set("source_file", os.path.basename(source_filename))
    else:
        root.set("source_file", "Not Provided")

    # Add block elements from results
    for i, result in enumerate(results):
        block_elem = ET.SubElement(root, "Block", id=str(i))

        # Add common attributes/elements
        if "page" in result:
            ET.SubElement(block_elem, "Page").text = str(result["page"])
        if "class" in result:
            block_elem.set("class", result["class"]) # Use attribute for class
        if "confidence" in result and result["confidence"] is not None:
             block_elem.set("layout_confidence", f"{result['confidence']:.4f}") # Add confidence

        # Add box coordinates as sub-elements for clarity
        if "box" in result and isinstance(result["box"], (tuple, list)) and len(result["box"]) == 4:
            box_elem = ET.SubElement(block_elem, "Box")
            ET.SubElement(box_elem, "x1").text = str(result["box"][0])
            ET.SubElement(box_elem, "y1").text = str(result["box"][1])
            ET.SubElement(box_elem, "x2").text = str(result["box"][2])
            ET.SubElement(box_elem, "y2").text = str(result["box"][3])
        elif "box" in result: # Handle non-standard box format if needed
             ET.SubElement(block_elem, "BoxInfo").text = _dict_to_xml_str(result["box"])

        # Add text or error
        if "text" in result and result["text"]:
            text_elem = ET.SubElement(block_elem, "Text")
            # Use CDATA section if text might contain XML-like characters,
            # otherwise just assign to text. For simplicity now, just assign.
            # Consider adding CDATA later if needed: text_elem.text = f"<![CDATA[{result['text']}]]>"
            text_elem.text = result["text"]
        elif "error" in result:
            error_elem = ET.SubElement(block_elem, "Error")
            error_elem.text = result["error"]
        elif "image_error" in result: # Handle file-level errors
             error_elem = ET.SubElement(block_elem, "FileError")
             error_elem.text = result["image_error"]
        elif "file_error" in result:
             error_elem = ET.SubElement(block_elem, "FileError")
             error_elem.text = result["file_error"]


    # Convert ElementTree to string with pretty printing
    try:
        # Use minidom for pretty printing (adds indentation and newlines)
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml_as_string = reparsed.toprettyxml(indent="  ", encoding='utf-8') # Use 2 spaces for indent

        # Write to file
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path:
            os.makedirs(output_dir_path, exist_ok=True)

        with open(output_path, 'wb') as f: # Write bytes
            f.write(pretty_xml_as_string)
        logger.info(f"XML file generated successfully: {output_path}")

    except IOError as e:
        logger.error(f"Failed to write XML to {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during XML generation/writing: {e}", exc_info=True)
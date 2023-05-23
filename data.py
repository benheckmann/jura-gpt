import re
from typing import Union, Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

from haystack import Document

def parse_element(element):
    """ Recursively parses an XML element and returns a dictionary of its structure """
    parsed_element = {}
    for child in element:
        if len(child) > 0:  # element has child elements
            parsed_element[child.tag] = parse_element(child)
        else:
            parsed_element[child.tag] = child.text
    return parsed_element

def parse_content(element):
    """ Recursively parses an XML content structure and returns a string of the contents """
    content = []
    for child in element:
        # If there is text directly within this child, append it
        if child.text is not None and child.text.strip() != '':
            content.append(child.text.strip())
        # If this child is a BR tag, append a newline
        if child.tag == 'BR':
            content.append('\n')
        # If there are further child elements, recursively parse those
        elif len(child) > 0:
            content.append(parse_content(child))
        # If there is text after this child element (tail), append that
        if child.tail is not None and child.tail.strip() != '':
            content.append(child.tail.strip())
    content_string = " ".join(content)
    # replace all instances of newline followed by multiple spaces with a single newline
    content_string = re.sub("\n\s+", "\n", content_string)
    return content_string


def parse_gg_xml_to_documents(xml_file: str) -> List[Document]:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    documents = []
    for norm in root.iter("norm"):
        doknr = norm.attrib.get("doknr")
        metadaten = norm.find("metadaten")
        textdaten = norm.find("textdaten")

        # Extract metadata
        meta = {}
        if metadaten is not None:
            meta = parse_element(metadaten)

        # Extract content
        content = ""
        if textdaten is not None:
            text = textdaten.find("text")
            if text is not None:
                content_element = text.find("Content")
                if content_element is not None:
                    content = parse_content(content_element)

        # Create document
        document = Document(
            content=content,
            content_type="text",
            id=doknr,
            meta=meta,
        )
        documents.append(document)

    return documents

_xml_file = "./data/grundgesetz.xml"
documents = parse_gg_xml_to_documents(_xml_file)

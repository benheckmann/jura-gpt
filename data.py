from typing import Union, Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

from haystack import Document

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
            for child in metadaten:
                meta[child.tag] = child.text

        # Extract content
        content = ""
        if textdaten is not None:
            text = textdaten.find("text")
            if text is not None:
                content = ET.tostring(text, encoding="utf-8", method="text").decode(
                    "utf-8"
                )

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

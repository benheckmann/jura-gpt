import logging
import os

import dotenv
from haystack import Pipeline
from haystack.nodes import PromptNode, PromptTemplate, BM25Retriever

from data import grundgesetz_in_memory_document_store, lfqa_prompt

dotenv.load_dotenv()

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)

retriever = BM25Retriever(document_store=grundgesetz_in_memory_document_store, top_k=5)

prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    default_prompt_template=lfqa_prompt,
    model_kwargs={"stream": True})

pipe = Pipeline()
pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

output = pipe.run(query="Wann wurde das Grundgesetz angenommen?")

print(output["results"])

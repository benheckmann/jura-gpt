import logging
import os

import dotenv
from haystack.nodes import OpenAIAnswerGenerator, BM25Retriever
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers

from data import grundgesetz_in_memory_document_store

dotenv.load_dotenv()

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)

retriever = BM25Retriever(document_store=grundgesetz_in_memory_document_store, top_k=2)

generator = OpenAIAnswerGenerator(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-davinci-003"
)

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)

QUESTIONS = [
    "wann wurde das Grundgesetz angenommen"
]

for question in QUESTIONS:
    res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
    print_answers(res, details="medium")

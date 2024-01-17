from typing import Annotated
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import OpenAI
# from textwrap import dedent
from llm_wrapper import llm_func

# Initialize Ollama
def initialize_ollama(model='openhermes', verbose=False):
    print(f"Initializing LLM model: {model}")
    if model == 'openai':
        llm = OpenAI()
    else:
        llm = Ollama(model=model, verbose=verbose)
    return llm


llm_openai=OpenAI()
llm_openhermes=initialize_ollama(model='openhermes')
llm_openhermes = llm_openai


@llm_func
def generate_base64_enc_svg_file() -> str:
    """Return just the base64 encoded svg content.
    """
    pass

import base64
# Example usage
query="draw a red circle."
print(f"Query: {query}")
svg_file = generate_base64_enc_svg_file(llm=llm_openhermes,query=query)
print(svg_file)
if ',' in svg_file:
    svg_file = svg_file.split(',')[1]
with open("data/test.svg", "wb") as fh:
    fh.write(base64.b64decode(svg_file))
    


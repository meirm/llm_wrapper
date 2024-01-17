import inspect
from typing import Union
from pydantic import BaseModel
from functools import wraps
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
# from textwrap import dedent


def llm_func(func)->Union[int,float,str,bool,list,dict,set, BaseModel]:
    """Given a query, run the LLM and return the output as a basic type (int, float, str, etc.).

    Args:
        llm (object): The LLM model to run.
        func (object): The wrapped function to simulate.
        query (str): The query to run the LLM on.
    """
    
    return_type = get_return_annotation(func)
    return_type = get_return_annotation(func)
    doc_string = get_docstring(func)
    if issubclass(return_type, BaseModel):
        parser = PydanticOutputParser(pydantic_object=return_type)
        prompt = PromptTemplate(
            template="{doc_string}\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions(), "doc_string": doc_string}
        )
    
        @wraps(func)
        def wrapper(*args, **kwargs):
            llm = kwargs.get('llm')
            if llm is None:
                raise Exception("LLM model not initialized.")
            query = kwargs.get('query')
            # print(f"Prompt: {prompt.format_prompt(query=query)}")
            chain = prompt | llm | parser
            response = chain.invoke({"query": f"{query}"})
            try:
                response.model_dump_json()
                return response
            except Exception as e:
                print(f"Error: {e}")
                print(e.__traceback__)
                return None    
        return wrapper
    else:
        parser = BasicTypeOutputParser(return_type=return_type)
        prompt = PromptTemplate(
            template="{doc_string}\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions(), "doc_string": doc_string}
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            llm = kwargs.get('llm')
            if llm is None:
                raise Exception("LLM model not initialized.")
            return_type = get_return_annotation(func)
            parser = BasicTypeOutputParser(return_type=return_type)
            query = kwargs.get('query')
            # print(f"Prompt: {prompt.format_prompt(query=query)}")
            chain = prompt | llm # | parser
            response = chain.invoke({"query": f"{query}"})
            try:
                return parser.parse(response)
            except Exception as e:
                print(f"Error: {e}")
                print(e.__traceback__)
                return None    
        return wrapper
        
    
        
    
def get_return_annotation(func):
    # Get the signature of the function
    signature = inspect.signature(func)
    # Get the return annotation of the function
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        raise Exception("Return type must be explicitly annotated for the wrapped function.")
    return return_annotation

def get_docstring(func):
    # Extract the docstring
    docstring = inspect.cleandoc(func.__doc__)
    return docstring


class BasicTypeOutputParser:
    def __init__(self, return_type):
        self.return_type = return_type
    
    def parse(self, response):
        """Parses the response from the LLM based on the specified return type."""
        if not response:
            raise ValueError("No response provided for parsing.")
        
        # Attempt to convert the response to the specified return type
        # print(f"Response: {response}")
        try:
            if self.return_type == int:
                return int(response)
            elif self.return_type == float:
                return float(response)
            elif self.return_type == str:
                return str(response)
            elif self.return_type == bool:
                return bool(response.lower() == "true")
            elif self.return_type == list:
                return list(response)
            elif self.return_type == dict:
                return dict(response)
            elif self.return_type == set:
                return set(response)
            else:
                raise ValueError(f"Unsupported return type: {self.return_type}")
        except ValueError as e:
            # Handle the case where conversion is not possible
            raise ValueError(f"Error parsing response: {e}")
    
    def get_format_instructions(self):
        """Provides format instructions based on the return type for the LLM's templating system."""
        if self.return_type == int:
            return "Please provide the response as an integer."
        elif self.return_type == float:
            return "Please provide the response as a floating-point number."
        elif self.return_type == str:
            return "Please provide the response as text."
        else:
            return "Unsupported return type."


    
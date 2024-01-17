import inspect
from typing import Union
from pydantic import BaseModel
from functools import wraps
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import json
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
                if response is None:
                    print("Response is None.")
                    return None
                print(f"Response:{response}")
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
                if response is None:
                    print("Response is None.")
                    return None
                print(f"Response:{response}")
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
            if response is None:
                return None
            if self.return_type == int:
                return int(response)
            elif self.return_type == float:
                return float(response)
            elif self.return_type == str:
                return str(response)
            elif self.return_type == bool:
                return bool(response.lower() == "true")
            elif self.return_type == list:
                return list(json.loads(response))
            elif self.return_type == dict:
                return dict(json.loads(response))
            elif self.return_type == set:
                # Convert set syntax to list syntax
                llm_output_as_list = response.replace('{', '[').replace('}', ']')
                # Parse the string into a list
                parsed_list = json.loads(llm_output_as_list)
                # Convert the list to a set
                parsed_set = set(parsed_list)
                return parsed_set
            else:
                raise ValueError(f"Unsupported return type: {self.return_type}")
        except ValueError as e:
            # Handle the case where conversion is not possible
            raise ValueError(f"Error parsing response: {e}")
    
    def get_format_instructions(self):
        """Provides format instructions based on the return type for the LLM's templating system."""
        if self.return_type == int:
            return "Please provide the response as an integer. "
        elif self.return_type == float:
            return "Please provide the response as a floating-point number. "
        elif self.return_type == str:
            return "Please provide the response as text. "
        elif self.return_type == bool:
            return "Please provide the response as 'True' or 'False'. "
        elif self.return_type == dict:
            return "Please provide the response as a json dictionary. "
        elif self.return_type == list:
            return "Please provide the response as a json list. "
        elif self.return_type == set:
            return "Please provide the response as a set. "
        else:
            return "Unsupported return type. "


    
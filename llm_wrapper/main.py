import inspect
from typing import Union
from pydantic import BaseModel
from functools import wraps
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import re
import json
import logging

logger = logging.getLogger(__name__)

def correct_json_format(response: str, llm: any) -> str:
    """Corrects the JSON formatting, handling mixed quotes, escape issues, extra commas, and non-standard wrappers."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    try:
        # Trim leading/trailing non-JSON characters (like ```)
        trimmed_response = re.sub(r'^[^{]*', '', response)
        trimmed_response = re.sub(r'[^}]*$', '', trimmed_response)

        # Attempt to load the response as JSON first in case it's already correctly formatted
        return json.loads(trimmed_response)
    except json.JSONDecodeError:
        pass
    try: 
        # Attempt to load the response as JSON first in case it's already correctly formatted
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    try:
        # Process to correct malformed JSON
        # Correct single quotes around keys and values
        corrected_response = re.sub(r"(\W)'([^']*)'", r'\1"\2"', trimmed_response)

        # Escape double quotes inside strings
        corrected_response = re.sub(r'(?<!\\)"', r'\"', corrected_response)

        # Convert it back to standard JSON format
        corrected_response = corrected_response.replace('\'', '\"')

        return json.loads(corrected_response)
    except json.JSONDecodeError:
        pass
    try:
        prompt = PromptTemplate(
            template="## ERROR\nerror parsing json instance:\n{response}\n\n"
            "## OBJECTIVE\nfix the json instance\n\n"
            "## RESPONSE\nvalid json\n\n",
            input_variables=["response"],
            # partial_variables={}
        )
        chain = prompt | llm
        ai_response = chain.invoke({"query": f"{response}"})
        print(f"Original Response: {response}")
        print(f"AI Response: {ai_response}")
        return json.loads(ai_response)
    except json.JSONDecodeError:
        raise ValueError("Error parsing response.")

# from textwrap import dedent


def llm_func(func)->Union[int,float,str,bool,list,dict,set, BaseModel]:
    """Given a query, run the LLM and return the output as a basic type (int, float, str, etc.).

    Args:
        llm (object): The LLM model to run.
        func (object): The wrapped function to simulate.
        query (str): The query to run the LLM on.
        Optional[on_error_retry] (int): The number of times to retry the LLM if it fails to parse the response. Defaults to 0.
    """
    
    return_type = get_return_annotation(func)
    doc_string = get_docstring(func)
    if issubclass(return_type, BaseModel):
        parser = PydanticOutputParser(pydantic_object=return_type)
        prompt = PromptTemplate(
            template="## CONTEXT\n{doc_string}\n\n##FORMAT\n{format_instructions}\n\n## DATA\n{query}\n\n"
            "## RESPONSE\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions(), "doc_string": doc_string}
        )
        

        @wraps(func)
        def wrapper(*args, **kwargs):
            llm = kwargs.get('llm')
            on_error_retry:int = kwargs.get('on_error_retry',0)
            if llm is None:
                raise Exception("LLM model not initialized.")
            query = kwargs.get('query')
            # print(f"Prompt: {prompt.format_prompt(query=query)}")
            try:
                chain = prompt | llm | parser
                response = chain.invoke({"query": f"{query}"})
                response.model_dump_json()
                return response
            except Exception as e:
                logger.error(f"Error: {e}")
                if on_error_retry==0:
                    raise e
                else:
                    pass
            response = None
            try:
                chain = prompt | llm
                llm_response = chain.invoke({"query": f"{query}"})
                validated_response = correct_json_format(response=llm_response, llm= llm)
                response = parser.parse(json.dumps(validated_response))
                response.model_dump_json()
                return response
            except Exception as e:
                logger.error(f"Error: {e}")
                if on_error_retry>0 and response is None and llm_response is not None:  # Error parsing response, retry
                    kwargs['on_error_retry'] = on_error_retry-1
                    return wrapper(*args, **kwargs)
                else:
                    logger.warning("Response is None.")
                    if llm_response is not None:
                        logger.warning(f"LLM Response:{llm_response}")
                        logger.warning(f"LLM Response Type:{type(llm_response)}")
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
                    logger.warning("Response is None.")
                    return None
                logger.warning(f"Response:{response}")
                return parser.parse(response)
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.error(e.__traceback__)
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


    
import inspect
from typing import Union
from pydantic import BaseModel
from functools import wraps
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import re
import json
import logging
from textwrap import dedent

# template_COSTAR = "## DATA\n{query}\n\n" \
#             "## CONTEXT\n{doc_string}\n\n" \
#             "## FORMAT\n{format_instructions}\n\n" \
#             "## OBJECTIVE\nRespond with the appropiate JSON instance only.\n\n" \
#             "## RESPONSE\n"
            
template_minimal = "user request:{query}\n\naction:\n{doc_string}\n\nformatting:\n{format_instructions}\n\nnote:\nrespond exactly as instructed without comments\n\nresponse:\n"
            
# template_verbose = """You are a developer working on a new API. You are tasked with mimicking the output of a function that returns a JSON instance.
# {doc_string}\n\n
# Respond to the user with a JSON instance that meets their criteria. Do not respond with anything else but the json instance.
# {format_instructions}
    
# REQUEST:    
# {query}
    
# JSON RESPONSE:
# """

# template_pydantic = """Answer the user query.\n{doc_string}\n{format_instructions}\n{query}\n"""
template_default = template_minimal
logger = logging.getLogger(__name__)


def extract_json(response: str) -> str:
    json_match = re.search(r'json\s*({.+?})\s*', response, re.DOTALL)
    json_string = '{}'
    if json_match:
        json_string = json_match.group(1)
    return json_string


def llm_func(func):
    """
    Decorator to run a function simulation using a Language Model.
    Args:
        func (callable): The function to simulate.
    Returns:
        callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        llm = kwargs.get('llm')
        if llm is None:
            raise Exception("LLM model not initialized.")

        query = kwargs.get('query')
        return_type = get_return_annotation(func)
        doc_string = get_docstring(func)
        parser = determine_parser(return_type)
        prompt = construct_prompt(doc_string, parser.get_format_instructions())

        try:
            response = run_llm_query(llm, query, prompt)
            return parse_response(response, parser)
        except Exception as e:
            handle_exception(e, query)

    return wrapper

def determine_parser(return_type):
    """
    Determines the appropriate parser based on the return type.
    """
    if issubclass(return_type, BaseModel):
        return PydanticOutputParser(pydantic_object=return_type)
    else:
        return BasicTypeOutputParser(return_type=return_type)

def construct_prompt(doc_string, format_instructions):
    """
    Constructs the LLM prompt from the docstring and format instructions.
    """
    return PromptTemplate(
        template=template_default,
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions, "doc_string": doc_string}
    )

def run_llm_query(llm, query, prompt):
    """
    Runs the query through the LLM and returns the response.
    """
    chain = prompt | llm
    return chain.invoke({"query": f"{query}"})

def parse_response(response, parser):
    """
    Parses the LLM response using the given parser.
    """
    if response is None:
        logger.warning("Response is None.")
        return None
    return parser.parse(response)

def handle_exception(exception, query):
    """
    Handles exceptions that occur during LLM query processing.
    """
    if exception.__class__.__name__ == "JSONDecodeError":
        logger.error(f"Error parsing response for '{query}': {exception}")
    elif exception.__class__.__name__ == "ValueError":
        logger.error(f"Error parsing response for '{query}': {exception}")
    else:
        logger.error(f"Error during LLM query for '{query}': {exception}")
    # Additional error handling logic can be added here.
    return None

def keep_first_json_object(json_string: str) -> str:
    """
    Keeps only the first complete JSON object or array from the input string.
    """
    brace_count = 0
    last_index = 0

    for i, char in enumerate(json_string):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                last_index = i + 1
                break

    first_json_object = json_string[:last_index].strip()
    
    try:
        # Validate if the extracted string is a proper JSON object
        json.loads(first_json_object)
        return first_json_object
    except json.JSONDecodeError:
        raise ValueError("No valid JSON object found.")
    
def correct_json_format(response: str, llm: any) -> str:
    """Corrects the JSON formatting, handling mixed quotes, escape issues, extra commas, and non-standard wrappers."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    try: 
        return keep_first_json_object(response)
    except ValueError:
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
        llm_prompt = PromptTemplate(
            template="## ERROR\nerror parsing json instance:\n{query}\n\n"
            "## OBJECTIVE\nfix the json instance\n\n"
            "## RESPONSE\nvalid json\n\n",
            input_variables=["response"],
            # partial_variables={}
        )
        llm_chain = llm_prompt | llm
        ai_response = llm_chain.invoke({"query": f"{response}"})
        logger.debug(f"Original Response: {response}")
        logger.debug(f"AI Response: {ai_response}")
        return json.loads(ai_response)
    except json.JSONDecodeError:
        raise ValueError("Error parsing response.")

# 


# def llm_func_old(func)->Union[int,float,str,bool,list,dict,set, BaseModel]:
#     """Given a query, run the LLM and return the output as a basic type (int, float, str, etc.).

#     Args:
#         llm (object): The LLM model to run.
#         func (object): The wrapped function to simulate.
#         query (str): The query to run the LLM on.
#         Optional[on_error_retry] (int): The number of times to retry the LLM if it fails to parse the response. Defaults to 0.
#     """
    
#     return_type = get_return_annotation(func)
#     doc_string = get_docstring(func)
#     if issubclass(return_type, BaseModel):
#         parser = PydanticOutputParser(pydantic_object=return_type)
        
        

#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             prompt = PromptTemplate(
#             template= template_default,
#             input_variables=["query"],
#             partial_variables={"format_instructions": parser.get_format_instructions(), "doc_string": doc_string}
#         )
#             llm = kwargs.get('llm')
#             on_error_retry:int = kwargs.get('on_error_retry',0)
#             if llm is None:
#                 raise Exception("LLM model not initialized.")
#             query = kwargs.get('query')
#             # print(f"Prompt: {prompt.format_prompt(query=query)}")
#             try:
#                 if kwargs.get('error',False):
#                     prompt = PromptTemplate(
#                         template= "## ERROR\n{error}\n{doc_string}\n{format_instructions}\n{query}\n",
#                         input_variables=["query"],
#                         partial_variables={"format_instructions": parser.get_format_instructions(), "doc_string": doc_string, "error": kwargs.get('error')}
#                     )
#                 chain = prompt | llm | parser
#                 response = chain.invoke({"query": f"{query}"})
#                 response.model_dump_json()
#                 return response
#             except Exception as e:
#                 logger.error(f"Error: {e}")
#                 pass
#             response = None
#             try:
#                 chain = prompt | llm
#                 llm_response = chain.invoke({"query": f"{query}"})
#                 validated_response = correct_json_format(response=llm_response, llm= llm)
#                 response = parser.parse(json.dumps(validated_response))
#                 response.model_dump_json()
#                 return response
#             except Exception as e:
#                 logger.error(f"Error: {e}")
#                 if on_error_retry>0 and response is None and llm_response is not None:  # Error parsing response, retry
#                     kwargs['on_error_retry'] = on_error_retry-1
#                     kwargs['error'] = f"Error parsing response: {e}"
#                     return wrapper(*args, **kwargs)
#                 else:
#                     logger.warning("Response is None.")
#                     if llm_response is not None:
#                         logger.warning(f"LLM Response:{llm_response}")
#                         logger.warning(f"LLM Response Type:{type(llm_response)}")
#                     return None  
#         return wrapper
#     else:
#         parser = BasicTypeOutputParser(return_type=return_type)
#         prompt = PromptTemplate(
#             template="{doc_string}\n{query}\n{format_instructions}\n",
#             input_variables=["query"],
#             partial_variables={"format_instructions": parser.get_format_instructions(), "doc_string": doc_string}
#         )
        
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             llm = kwargs.get('llm')
#             if llm is None:
#                 raise Exception("LLM model not initialized.")
#             return_type = get_return_annotation(func)
#             parser = BasicTypeOutputParser(return_type=return_type)
#             query = kwargs.get('query')
#             # print(f"Prompt: {prompt.format_prompt(query=query)}")
#             chain = prompt | llm # | parser
#             response = chain.invoke({"query": f"{query}"})
#             try:
#                 if response is None:
#                     logger.warning("Response is None.")
#                     return None
#                 logger.debug(f"Response:{response}")
#                 return parser.parse(response)
#             except Exception as e:
#                 logger.error(f"Error: {e}")
#                 logger.error(e.__traceback__)
#                 return None    
#         return wrapper
        
        
    
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
                response = response.replace("\n", "").replace("\t", "").replace("\r", "")
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


    
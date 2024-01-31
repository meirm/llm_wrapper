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


llm_openai=initialize_ollama(model='openai')
llm_openhermes=initialize_ollama(model='openhermes')
llm_openhermes = llm_openai


@llm_func
def random_set() -> set:
    """Return a random set of numbers.
    """
    pass

# Example usage
query="Give me a random set of numbers."
print(f"Query: {query}")
myset = random_set(llm=llm_openai,query=query)
print(myset)

@llm_func
def random_names() -> list:
    """Return a list of random names.
    """
    pass

# Example usage
query="Give me 3 random female latino names."
print(f"Query: {query}")
names = random_names(llm=llm_openai,query=query, on_error_retry=3)
print(names)

@llm_func
def random_identification() -> dict:
    """Return a json object: {"username": __random_name__, "password": __random_password__}.
    """
    pass

# Example usage
query="Give me a random username and password."
print(f"Query: {query}")
random_id = random_identification(llm=llm_openai,query=query)
print(random_id)

# Example usage
query="Give me a random password for username meirm."
print(f"Query: {query}")
random_id = random_identification(llm=llm_openai,query=query)
print(random_id)


@llm_func
def check_grammar() -> float:
    """Check the grammar of the sentence and return a float number between 0 and 1 reflecting its correctness."""
    pass

query = "I are a student."
correctness = check_grammar(llm=llm_openai, query=query)
print(correctness) 



@llm_func
def probability()-> float:
    """Return a probability between 0 and 1 of the text being offensive.
    """
    pass

query="You are an idiot."
print(f"Query: {query}")
offensive = probability(llm=llm_openai,query=query)
print(offensive)


@llm_func
def is_offensive() -> bool:
    """returns 'True' if the text is offensive else 'False'.
    """
    pass

# Example usage
query="You are an idiot."
print(f"Query: {query}")
offensive = is_offensive(llm=llm_openai,query=query)
print(offensive)

query="I am glad you are here."
print(f"Query: {query}")
offensive = is_offensive(llm=llm_openai,query=query)
print(offensive)


@llm_func
def quote() -> str:
    """Create a quote related to subject.
    """
    pass

# Example usage
query="subject: love"
print(f"Query: {query}")
myquote = quote(llm=llm_openai,query=query)

print(myquote)

@llm_func
def dice_roller() -> int:
    """Roll {nr_dices} and return the total value.
    """
    pass

# Example usage
query="Roll 2 dices."
print(f"Query: {query}")
myroll = dice_roller(llm=llm_openai,query=query)
print(myroll)




class BlogEntry(BaseModel):
    """A blog entry with a title and body.
    Attributes:
        title: The title of the blog entry.
        body: The body of the blog entry.
    """
    title: Annotated[str, Field(description="The title of the blog entry.", min_length=1)]
    body: Annotated[str, Field(description="The body of the blog entry.", min_length=1)]


# Usage example
@llm_func
def blog_entry() -> BlogEntry:
    """Create a short blog entry with a title and body.
    """
    pass

query="Why you should eat once a day."
print(f"Query: {query}")
answer = blog_entry(query=query, llm=llm_openai)
print(answer.model_dump_json())

# Usage example
@llm_func
def funny_blog_entry() -> BlogEntry:
    """Create a  short funny blog entry with a title and body.
    """
    pass

query="Why you should eat once a day."
print(f"Query: {query}")
answer = funny_blog_entry(query=query, llm=llm_openai)
print(answer.model_dump_json())



class AccessGranted(BaseModel):
    """Access granted to the user.
    Attributes:
        access: The access granted to the user.
    """
    grant: Annotated[bool, Field(description="grant access to user.", default=False)]
    reason: Annotated[str, Field(description="The reason why the person is granted access or not.", min_length=1)]
    
# Example usage
@llm_func
def access() -> AccessGranted:
    """Grant access to users aged 18 and above.
    If the username is 'John', access is denied and the reason is 'access denied'.
    If the user is underage, the reason is 'bring your parents' else the reason is 'access granted'.
    
    """
    pass

query="My username is John and I am 50 years old."
print(f"Query: {query}")
access_granted = access(query=query, llm=llm_openai)

# Print the model schema with docstring
print(access_granted.model_dump_json())


query="I am in kindergarten and my name is Mary."
print(f"Query: {query}")
access_granted = access(query=query, llm=llm_openai)

# Print the model schema with docstring
print(access_granted.model_dump_json())



query="I am a teacher at Oxford."
print(f"Query: {query}")
access_granted = access(query=query, llm=llm_openai)

# Print the model schema with docstring
print(access_granted.model_dump_json())


class Person(BaseModel):
    """A person with a name and age."""
    name: Annotated[str, Field(..., description="The name of the person.", min_length=1)]
    age: Annotated[int, Field(..., description="The age of the person.", gt=0)]
    
# Example usage
@llm_func
def example_function() -> Person:
    """
    extract the name and age of a person.
    
    """
    pass



class SentimentAnalysis(BaseModel):
    """The sentiment analysis of the text
    Attributes:
        value: The sentiment value of the text.
        tag: The sentiment tag of the text. Lowercase.
    
    """
    value: float = Field(..., description="The sentiment value of the text.", gt=-1.0, lt=1.0)
    tag: Annotated[str, Field(..., description="The sentiment tag of the text.", pattern="^(positive|negative|neutral)$")]
    
# Example usage
@llm_func
def sentiment() -> SentimentAnalysis:
    """
    Run sentiment analysis on the given text.
    """
    pass


my_wife = example_function(query="His name was Cristobal Nopil and he was 30 years old when he died.", llm=llm_openhermes)

# Print the model schema with docstring
print(my_wife.model_dump_json())
query="Your coffee is bad."
print(f"Query: {query}")
ifeel = sentiment(query=query, llm=llm_openai)

# Print the model schema with docstring
print(ifeel.model_dump_json())


query="Your coffee is good."
print(f"Query: {query}")
ifeel = sentiment(query=query, llm=llm_openai)

# Print the model schema with docstring
print(ifeel.model_dump_json())



query="Your coffee is just ok."
print(f"Query: {query}")
ifeel = sentiment(query=query, llm=llm_openai)

# Print the model schema with docstring
print(ifeel.model_dump_json())


class Contact(BaseModel):
    """A contact with a name and phone number.
    Attributes:
        name: The name of the contact.
        phone_number: The phone number of the contact.
    """
    name: Annotated[str, Field(description="The name of the contact.", min_length=1)]
    phone_number: Annotated[str, Field( description="The phone number of the contact.", min_length=1)]
    
# Example usage
@llm_func
def contact() -> Contact:
    """Extract the contact's name and phone number.
    """
    pass

query="My name is John and my phone number is 555-555-5555."
print(f"Query: {query}")
my_contact = contact(query=query, llm=llm_openhermes)

# Print the model schema with docstring

print(my_contact.model_dump_json())


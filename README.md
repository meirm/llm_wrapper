# LLM Functions Library Usage Guide

This guide provides a comprehensive overview of how to use the `llm_wrapper` library, which offers a versatile wrapper, `llm_func`, designed for seamless interactions with various language learning models (LLMs). The wrapper simplifies model initialization, query execution, and structured output parsing, supporting a wide range of return types including basic data types (`int`, `float`, `str`, `bool`, `set`, `list`, `dict`) and complex Pydantic `BaseModel` structures.

## Getting Started
```python
from llm_wrapper import llm_func
from langchain_openai import OpenAI

@llm_func
def famous_quote() -> str:
    """Returns a famous quote according to the subject provided."""
    pass

llm = OpenAI()

query = "Peace and War"
quote = famous_quote(llm=llm, query=query)
print(quote)  # Output: "Peace is not a relationship of nations. It is a condition of mind brought about by a serenity of soul. Peace is not merely the absence of war. It is also a state of mind. Lasting peace can come only to peaceful people. - Jawaharlal Nehru

@llm_func
def check_grammar() -> float:
    """Check the grammar of the sentence and return a float number between 0 and 1 reflecting its correctness."""
    pass

query = "I are a student."
correctness = check_grammar(llm=llm, query=query)
print(correctness)  # Output: 0.5
query = "I am a student."
correctness = check_grammar(llm=llm, query=query)
print(correctness)  # Output: 1.0
```
### Installation

Ensure the `llm_wrapper` library is installed in your environment. You can install it using pip:

```bash
pip install llm_wrapper
```

### Importing the Library

Start by importing the necessary components:

```python
from llm_wrapper import llm_func
from pydantic import BaseModel
```

### Initializing Your LLM Object

You'll need to instantiate your preferred LLM object. This library is designed to work flexibly with various LLMs:

```python
llm = YourPreferredLLM()
```

## Using the `llm_func` Wrapper

The `llm_func` wrapper is designed to streamline your interaction with LLMs. It automatically handles functions returning basic types (`int`, `float`, `str`, `bool`) or Pydantic `BaseModel` instances.

### Defining Functions with `@llm_func`

Annotate your functions with `@llm_func` and define a clear return type. Here's how to define functions returning basic types and Pydantic models:

#### Basic Types

```python
@llm_func
def calculate_score() -> int:
    """Returns an integer score based on the input text."""
    pass

@llm_func
def is_valid() -> bool:
    """Determines if the text meets certain criteria, returning True or False."""
    pass
```

#### Pydantic BaseModel

```python
class User(BaseModel):
    name: str
    age: int

@llm_func
def get_user_details() -> User:
    """Extracts user details from the text and returns them as a User model."""
    pass
```

### Executing Queries

Pass your query to the function, along with the instantiated LLM object. The wrapper will process the input and return a structured output based on the defined return type.

```python
query = "Calculate the score for the following text..."
score = calculate_score(llm=llm, query=query)
print(score)  # Output will be of type int

query = "Check if the following text is valid..."
validity = is_valid(llm=llm, query=query)
print(validity)  # Output will be of type bool

query = "Extract user details from the following text..."
user_details = get_user_details(llm=llm, query=query)
print(user_details)  # Output will be a User instance
```

### Support and Development

Currently, `llm_func` supports functions returning basic data types (`int`, `float`, `str`, `bool`, `set`, `list`, `dict`) and Pydantic `BaseModel` instances. Support for additional types is under active development, and updates will be released periodically to enhance the library's functionality.

By following these guidelines, you can efficiently use the `llm_wrapper` library to interact with language models, perform queries, and handle structured outputs, all while writing clear and maintainable code.

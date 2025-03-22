from langchain_core.tools import tool
from typing import Annotated
import requests
from .python_repl import PythonREPL
import os
local_working_directory = os.getenv("LOCAL_WORKING_DIRECTORY")

@tool
def multiply(a:Annotated[int,"first number"], b:Annotated[int,"Second Number"]):
    """Multiply two numbers"""
    return a * b



@tool(parse_docstring=True,error_on_invalid_docstring = False)
def read_webpage(url: str) -> str:
    """Read the webpage at the given URL, use whenever you need to access the content of a webpage. 

    Args:
        url: The URL of the webpage to read

    Returns:
        str: The content of the webpage
    """
    response = requests.get("https://r.jina.ai/" + url)
    return response.text

python_repl = PythonREPL(working_dir=local_working_directory)

@tool
def repl_tool(code:str):
    """
    Run Python code in a REPL environment. To see the result, you must include the print statement in the code. If no print statement, you will not see the execution result and an empty string will be returned. Python repl cannot be used to call other tools or actions you have.

    Args:
        code: The Python code to run.

    Returns:
        str: The output of the code execution.
    """
    if "default_api" in code:
        return "ERROR, YOU ARE HALLUCINATING. You cannot call other tools or actions in the Python REPL."
    return python_repl.run(code)


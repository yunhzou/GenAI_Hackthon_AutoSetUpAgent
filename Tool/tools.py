from langchain_core.tools import tool
from typing import Annotated
import requests
from .python_repl import PythonREPL
import os
import subprocess
from .pexpect_interactive_shell import InteractiveShell

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
def repl_tool(code:Annotated[str, "The Python Script to run"]):
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



shell = InteractiveShell()

@tool
def execute_shell_command(shell_command:Annotated[str,"The shell command to execute"]):
    """Execute a shell command and return the output, the shell is preserved by session, thus it is fine to cd first and then ls."""
    output = shell.execute(shell_command)
    return output

@tool 
def ask_question_to_user(question:Annotated[str,"The question to ask the user"]):
    """Ask a question to the user and return the answer"""
    return input(f"Agent is asking: {question}, please type your feedback" )

@tool
def create_file():
    filename = input("Enter the filename: ")
    try:
        with open(filename, 'w') as file:
            file.write("")  # Optionally write initial content
        print(f"File '{filename}' created successfully!")
    except Exception as e:
        print(f"Error creating file '{filename}': {e}")
    


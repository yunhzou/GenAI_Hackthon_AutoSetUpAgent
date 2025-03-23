from langchain_core.tools import tool
from typing import Annotated
import requests
from .python_repl import PythonREPL
import os
import subprocess
from .pexpect_interactive_shell import InteractiveShell

local_working_directory = os.getenv("LOCAL_WORKING_DIRECTORY")

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

shell = InteractiveShell(root_dir=local_working_directory)

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
def create_file(filename:Annotated[str,"The name of the file to create"],content:Annotated[str,"The content of the file"]):
    """Create a file with the given content"""
    with open(filename,"w") as f:
        f.write(content)
    return f"File {filename} created"    

@tool
def append_to_file(
    filename: Annotated[str, "Name of the file to append to"],
    content: Annotated[str, "A chunk of Python code to write"]
) -> str:
    """
    Appends a chunk of code to an existing file.
    """
    try:
        with open(filename, 'a') as f:
            f.write(content + "\n")
        return f"✅ Appended to '{filename}'."
    except Exception as e:
        return f"❌ Failed to append: {e}"

@tool
def run_python_file(filename: Annotated[str, "Name of the Python file to run"]) -> str:
    """
    Runs a Python file and returns the output.
    """
    try:
        result = run(["python", filename], capture_output=True, text=True)
        if result.returncode == 0:
            return f"✅ Output:\n{result.stdout}"
        else:
            return f"⚠️ Error:\n{result.stderr}"
    except Exception as e:
        return f"❌ Failed to run file: {e}"

@tool
def play_audio(
    file_path: Annotated[str, "Path to the audio file to play using afplay"]
) -> str:
    """
    Play a local audio file using macOS's afplay command.

    Args:
        file_path: The path to the audio file (e.g., 'output.wav')

    Returns:
        str: A confirmation message indicating success or failure.
    """
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    
    try:
        # Run the afplay command
        result = run(["afplay", file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return f"🔊 Playing audio from: {file_path}"
        else:
            return f"⚠️ Error playing audio: {result.stderr}"
    except Exception as e:
        return f"❌ Failed to play audio: {e}"

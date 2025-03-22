from langchain_core.tools import tool
from typing import Annotated


@tool
def multiply(a:Annotated[int,"first number"], b:Annotated[int,"Second Number"]):
    """Multiply two numbers"""
    return a * b

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import ast
import re
from langchain_experimental.tools import PythonREPLTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Any, Dict, Optional, Type, Annotated
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from AgentNet.Agent.langchain_chatbot import LangChainChatBot 

class PrintStatementVerifier(LangChainChatBot):
    def __init__(self,
                model:str,
                session_id: Optional[str] = "PrintStatementVerifier",
                **kwargs
                ):
        super().__init__(model,session_id,**kwargs)

    def _get_prompt_template(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                 You are a helpful assistant to verify code.
                There is a special requirement for the code you verify: 
                You must decide how to include a print statement in the code to provide useful feedback.
                For example, when the code generate a plot, you should suggest add a print message like 'plot generated successfully, please view the pop up window'. 
                 """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        return prompt
    

def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


class CodeInput(BaseModel):
    query: str = Field(description = "The python code to execute.")

class PythonREPLToolModified(PythonREPLTool):
    """A modified tool for running python code in a REPL with a print() requirement."""

    args_schema: Type[BaseModel] = CodeInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool, ensuring a print() statement is included in the query."""
        if self.sanitize_input:
            query = sanitize_input(query)
        if "print(" not in query:
            suggestion = PrintStatementVerifier(model = "gpt-4o-mini",session_id="ML_test").invoke(query)
            return ("CODE EXECUTION FAILED." + suggestion.content + "Retry.") 
            #return "CODE EXECUTION FAILED. Your query must include a print() statement inside code to provide useful feedback. print either results or feedback indicating code execution is success."
        return super()._run(query, run_manager)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously, ensuring a print() statement is included."""
        if self.sanitize_input:
            query = sanitize_input(query)
        if "print(" not in query:
            suggestion = PrintStatementVerifier(model = "gpt-3.5-turbo-0125",session_id="ML_test").invoke(query)
            
            #return "CODE EXECUTION FAILED. Your query must include a print() statement inside code to provide useful feedback. print either results or feedback indicating code execution is success."
            return ("CODE EXECUTION FAILED." + suggestion.content + "Retry.") 
        return await super()._arun(query, run_manager)
    



    
from AgentNet.config import nosql_service
from langchain_experimental.utilities.python import PythonREPL
class PyMongoREPL(PythonREPL):
    """Simulates a Python REPL with pymongo client."""

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize MongoDB client and make it globally available
        self.globals['client'] = nosql_service

def _get_default_python_repl() -> PyMongoREPL:
    return PyMongoREPL(_globals=globals(), _locals=None)

# create a wrapper of MongoDBTool as a tool for agent to use
class PyMongoREPLToolModified(PythonREPLToolModified):  
    """A modified tool for running PyMongo code in a REPL with a print() requirement."""
    name: str = "PyMongo_REPL"
    description: str = (
        "A Python shell with global variable 'client'. Use this to execute pymongo commands. "
        "The MongoDB instance will be a variable called client. "
        "You should directly use the client variable to interact with the database. For example, you can directly run print(client.list_database_names())  "
        "Input should be a valid python command. "
        "You must print the output with `print(...)` to see the output."
        "If you get an error message such as Code Execution Failed, read the suggestions, debug your code and try again."
    )
    python_repl: PyMongoREPL = Field(default_factory=_get_default_python_repl)
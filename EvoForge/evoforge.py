from EvoForge.Agent import LangGraphAgent
from EvoForge.Tool.tools import *
from typing import List
from langchain_core.tools import BaseTool
from .prompt import SETUP_AGENT_SYSTEM_MESSAGE,TEACHING_AGENT_SYSTEM_MESSAGE,EXPERT_SYSTEM_MESSAGE


# Create a agent that setup the thing, after the setup, the setup should be documented in case of future reference.
# Teaching Material generation, the agent should be able to generate teaching material for the user.
# When the project is activated again after the setup, it should be the agent that learned how to use the project in charge of the project, such that it can use the project as user wish. 


class EvoForge:
    def __init__(self):
        pass
    
    def spawn_setup_agent(self,session:str,model:str="claude-3-7-sonnet-20250219"):
        """Spawn an agent in the given session"""
        setup_tools = [read_webpage,execute_shell_command,ask_question_to_user,create_file]
        setup_agent = LangGraphAgent(model=model,session_id=session)
        setup_agent.rewrite_system_message(SETUP_AGENT_SYSTEM_MESSAGE)
        setup_agent.add_tool(setup_tools)
        return setup_agent
    

    def spawn_teaching_material_agent(self,session:str,model:str="claude-3-7-sonnet-20250219"):
        """Spawn an agent in the given session"""
        teaching_material_tools = [read_webpage,execute_shell_command,ask_question_to_user,create_file]
        teaching_material_agent = LangGraphAgent(model=model,session_id=session)
        teaching_material_agent.rewrite_system_message(TEACHING_AGENT_SYSTEM_MESSAGE)
        teaching_material_agent.add_tool(teaching_material_tools)
        return teaching_material_agent
    
    def spawn_expert_agent(self,session:str,model:str="claude-3-7-sonnet-20250219"):
        """Spawn an agent in the given session"""
        expert_tools = [read_webpage,execute_shell_command,ask_question_to_user,create_file]
        expert_agent = LangGraphAgent(model=model,session_id=session)
        expert_agent.rewrite_system_message(EXPERT_SYSTEM_MESSAGE)
        expert_agent.add_tool(expert_tools)
        return expert_agent 




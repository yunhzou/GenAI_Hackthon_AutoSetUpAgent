from EvoForge.Agent import LangGraphAgent
from EvoForge.Tool.tools import *
from typing import List
from langchain_core.tools import BaseTool
from .prompt import SETUP_AGENT_SYSTEM_MESSAGE,TEACHING_AGENT_SYSTEM_MESSAGE,EXPERT_SYSTEM_MESSAGE
from .grounding import generate_tree_structure,load_ignore_patterns
from pathlib import Path

# Create a agent that setup the thing, after the setup, the setup should be documented in case of future reference.
# Teaching Material generation, the agent should be able to generate teaching material for the user.
# When the project is activated again after the setup, it should be the agent that learned how to use the project in charge of the project, such that it can use the project as user wish. 
local_working_directory = os.getenv("LOCAL_WORKING_DIRECTORY")

def update_grounding(**kwarg):
    # Load the ignore patterns
    ignore_patterns = load_ignore_patterns(".ignore")
    # Generate the tree structure for the current directory
    local_grounding = generate_tree_structure(local_working_directory, ignore_patterns)
    # Get files metadata
    grounding = "This is the current working directory file structure\n\n###### Local Working Directory #####\n"+ local_grounding
    return grounding


class EvoForge:
    def __init__(self):
        pass
    
    def spawn_setup_agent(self,session:str,model:str="claude-3-7-sonnet-20250219"):
        """Spawn an agent in the given session"""
        execute_shell_command = create_execute_shell_command_tool(Path(local_working_directory)/session)
        setup_tools = [read_webpage,execute_shell_command,ask_question_to_user,create_file]
        setup_agent = LangGraphAgent(model=model,session_id=session)
        setup_agent.rewrite_system_message(SETUP_AGENT_SYSTEM_MESSAGE)
        setup_agent.add_tool(setup_tools)
        setup_agent.inject_text_block_human_message(block_name="file_directory_view",block_description="",block_update_function=update_grounding)
        return setup_agent
    

#TODO: teaching and expert currently are not in use
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

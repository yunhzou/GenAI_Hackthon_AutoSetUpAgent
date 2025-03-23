from pydantic import BaseModel, Field, model_validator,ValidationError
from typing_extensions  import Optional, Annotated, TypedDict, Union,List, get_args 
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from .langgraph_supporter import LangGraphSupporter
from .agent_utils import get_date, extract_all_text_exclude_edges
from langchain.tools import BaseTool


class TaskDecompositionAgent(LangGraphSupporter):
    """
    task decomposition agent/planning agent.
    Decompose a task into smaller subgoals.
    And assign actions needed to complete those smaller subgoals.

    """ 

    class sub_task(BaseModel):
        task_name: str = Field(description="name of the subtask")
        task_context: str = Field(description="context of the subtask such as objective and technical details")

    class task_decomposition(BaseModel):
        steps: List["TaskDecompositionAgent.sub_task"] = Field(description="list of subtasks")
        
        @model_validator(mode='before')
        def validate_and_fix_structure(cls, values):
            """
            Validate and fix structure by normalizing field names and handling incorrect nesting.
            """
            def find_value(data, key_to_find):
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key.lower() == key_to_find.lower():
                            return value
                        else:
                            found = find_value(value, key_to_find)
                            if found is not None:
                                return found
                elif isinstance(data, list):
                    for item in data:
                        found = find_value(item, key_to_find)
                        if found is not None:
                            return found
                return None

            def fix_data(data, model):
                if not isinstance(data, dict):
                    return data
                fixed_data = {}
                for field_name, field_info in model.model_fields.items():
                    field_value = None
                    # Try to get the value matching the field name, case-insensitive
                    for key, val in data.items():
                        if key.lower() == field_name.lower():
                            field_value = val
                            break
                    # If not found, try to find it in nested data
                    if field_value is None:
                        field_value = find_value(data, field_name)
                    # If field is a nested model or Union
                    if field_value is not None:
                        if hasattr(field_info.annotation, '__pydantic_model__'):
                            field_value = fix_data(field_value, field_info.annotation)
                        elif (
                            hasattr(field_info.annotation, '__origin__')
                            and field_info.annotation.__origin__ is Union
                        ):
                            # Try each type in the Union
                            for sub_type in get_args(field_info.annotation):
                                try:
                                    field_value_fixed = fix_data(field_value, sub_type)
                                    # Validate by trying to instantiate the sub_type
                                    sub_instance = sub_type(**field_value_fixed)
                                    field_value = field_value_fixed  # Valid sub_type found
                                    break
                                except ValidationError:
                                    continue
                    fixed_data[field_name] = field_value
                return fixed_data

            values = fix_data(values, cls)
            return values


    class AgentState(TypedDict):
        agent_name:str
        session_id: Optional[str] 
        messages: Annotated[list, add_messages]
        messages_clean:Annotated[list, add_messages]
        plan: List[dict]

    def __init__(self, model: str, 
                 session_id: Optional[str] = "AgentSession" + get_date(), 
                 agent_name:str="default_name", 
                 memory_db_name = "chat_history",
                 **kwargs):
        self.model = model
        self.agent_schema = self.task_decomposition
        self.graph_state = self.AgentState
        self.llm = ChatOpenAI(temperature=0, model=model, **kwargs)
        super().__init__(session_id=session_id,agent_name=agent_name,memory_db_name= memory_db_name, **kwargs)

    def _initialize_system_message(self):
        if self.system_message:
            return self.rewrite_system_message(self.system_message)
        else:
            system_message = """You are ask to form subtasks to complete user's requirement.
            """
            return self.rewrite_system_message(system_message)
    
    def _create_agent(self):
        self.structured_llm = self.llm.with_structured_output(self.agent_schema,method='json_schema')
        workflow = StateGraph(self.graph_state)
        workflow.add_node("TaskDecomposition", self._agent_thought_process)
        workflow.add_edge(START, "TaskDecomposition")
        workflow.add_edge("TaskDecomposition", END)
        self.agent = workflow.compile(checkpointer=self.memory_manager)

    def _agent_thought_process(self,state: AgentState):
        """Update memory and state

        Returns:
            _type_: _description_
        """
        # Retrieve the last message from the user
        parsed_response = self.structured_llm.invoke(state["messages"]).model_dump()
        # Append the AI's thoughts to the message history
        response = "Here is the plan you created:\n" + str(parsed_response)
        response = AIMessage(content=response)
        return {"messages":[response],"plan":parsed_response}

    def generate_pseudocode_with_calls(self,goal_data):
        """goal_data is the result['plan'] from the agent

        Args:
            goal_data (_type_): _description_

        Returns:
            context (str): the joint pseudocode of the plan
            steps_pseudocode (List[str]): the pseudocode of each step stored as list of string
        """
        pseudocode = f"# Target: {goal_data['target']}\n"
        pseudocode += f"# {'=' * (len(goal_data['target']) + 8)}\n\n"
        steps_pseudocode = []
        for step_idx, step in enumerate(goal_data['steps'], start=1):
            step_pseudo = f"# Step {step_idx}: {step['goal_name']}\n"
            step_pseudo += f"# Description: {step['goal_description']}\n"
            for action in step['goal_actions']:
                function_name = action['action_name'].replace(" ", "_")
                args = ", ".join(
                    f"{arg['arg_name'].replace(' ', '_')}={repr(arg['arg_input'])}" 
                    for arg in action['action_args']
                )
                step_pseudo += f"{function_name}({args})\n"

            step_pseudo += f"# End of Step {step_idx}\n\n"
            steps_pseudocode.append(step_pseudo)
            pseudocode += step_pseudo

        return pseudocode.strip(), steps_pseudocode
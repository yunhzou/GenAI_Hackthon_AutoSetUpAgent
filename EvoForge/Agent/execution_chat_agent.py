from typing import Optional
from langchain_openai import ChatOpenAI
from langgraph.graph.graph import CompiledGraph
from .langgraph_supporter import LangGraphSupporter
from langgraph.graph.message import add_messages
from operator import add
from pydantic import BaseModel, Field
from typing import Annotated, Union, List, TypedDict
from langgraph.graph import StateGraph, END, START
from .agent_utils import get_date
from .agenttoolnode import AgentNetToolNode
from .agent_utils import content_read_out, LLM_Failure
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import os 
from langchain_core.messages import AIMessage, HumanMessage


class LangGraphAgent(LangGraphSupporter):   
    class Progress(BaseModel):
        """Template to formulate an action proposal which only contains one action """
        progress_percentage: int = Field(description="An estimate of current progress in percentage.")

    class AgentState(TypedDict):
        target_task:str
        agent_name:str
        progress: int
        messages: Annotated[list, add_messages]
        messages_clean: Annotated[list, add_messages]
        tool_used: Annotated[List[dict], add]
        session_id: Optional[str] 
        memory_db_name: Optional[str]
        continue_from_error: Optional[bool] = False
        
        """
        AgentState is the global graph state manager. Think of each field as global variables that can be accessed by each node of graph

        goal: the goal of the agent. The agent work for this goal until its completed or failed
        agent_name: the name of the agent
        messages: the conversation messages history
        tool_used: the tools used by the agent to achieve the goal
        continue_from_error: if True, the agent will continue from the last error point. Default is False
        """
    
    graph_state = AgentState

    def __init__(self,
                 model:str,
                 session_id: Optional[str] = "AgentSession" + get_date(),
                 interrupt_before: List[str]=[],
                 agent_name:str="default_name",
                 memory_db_name = "chat_history",
                 handle_tool_errors: bool = True,
                 **kwargs
                 ):
        """
        Text completion in json mode

        Args:
            client (_type_): OpenAI client
            model (_type_): choose from openai exisiting models
            usersetup (_type_): user instructions that the assitant will always follow
        """
        self.model=model
        self.progress_schema = self.Progress
        if "gpt" in model:
            self.llm = ChatOpenAI(temperature=0,model=model,max_tokens=None, **kwargs)
        elif "gemini" in model:
            self.llm = ChatGoogleGenerativeAI(temperature=0,model=model, max_tokens=None,**kwargs)
        elif "claude" in model:
            if "3.7" in model:
                self.llm = ChatAnthropic(temperature=0,
                                         model=model,
                                         max_tokens=128000,
                                         thinking={"type": "enabled", "budget_tokens": 4000},
                                           **kwargs)
            else:
                self.llm = ChatAnthropic(temperature=0,model=model, **kwargs)
        else:
            raise ValueError(f"Model {self.model} not supported")
        self.interrupt_before = interrupt_before
        self.handle_tool_errors = handle_tool_errors
        self.memory_db_name = memory_db_name
        super().__init__(session_id=session_id,agent_name=agent_name,memory_db_name= memory_db_name, **kwargs)


    def _initialize_system_message(self):
        if self.system_message:
            return self.rewrite_system_message(self.system_message)
        else:
            system_message = """You are a helpful assistant"""
            return self.rewrite_system_message(system_message)

    def _create_agent(self)->CompiledGraph:
        workflow = StateGraph(self.graph_state)
        tools2add = self.tools+self.un_detachable_tools
        self.sturctured_llm = self.llm.with_structured_output(self.progress_schema)
        if tools2add != []:
            tool_node = AgentNetToolNode(tools2add,
                                         handle_tool_errors=self.handle_tool_errors)
            if "gpt" in self.model: 
                self.llm_with_tools = self.llm.bind_tools(tools2add,parallel_tool_calls=False,tool_choice="auto")
            elif "gemini" in self.model:
                self.llm_with_tools = self.llm.bind_tools(tools2add,tool_choice="auto")
            elif "claude" in self.model:
                self.llm_with_tools = self.llm.bind_tools(tools2add,tool_choice="auto")
            workflow.add_node("Record_Target", self._record_target_task)
            workflow.add_node("agent", self.call_model)
            workflow.add_node("tools", tool_node)
            workflow.add_node("sync_messages", self.sync_state)
            workflow.add_node("progress_estimate", self.progress_estimation)
            workflow.add_edge(START, "Record_Target")
            workflow.add_edge("Record_Target", "agent")
            workflow.add_conditional_edges("agent", self.should_continue, {"tools": "tools", END: END})
            workflow.add_edge("tools", "sync_messages")
            workflow.add_edge("sync_messages", "progress_estimate")
            workflow.add_edge("progress_estimate", "agent")
        else:
            # If no tools, it will be chatbot
            self.llm_with_tools = self.llm
            workflow.add_node("Record_Target", self._record_target_task)
            workflow.add_node("agent", self.call_model)
            workflow.add_edge(START, "Record_Target")
            workflow.add_edge("Record_Target", "agent")
            workflow.add_edge("agent",END)
        self.agent = workflow.compile(checkpointer=self.memory_manager,interrupt_before=self.interrupt_before)

    def should_continue(self,state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        self.formatted_history_record["progress"] = 100
        return END
    
    def progress_estimation(self,state:AgentState):
        message_hisotry = self.get_message_history(state)
        target = state["target_task"]
        message_hisotry.append(HumanMessage(content=f"The target is {target}, Based on the history. What is the progress? Note the past progress is {state['progress']}"))
        parsed_response = self.sturctured_llm.invoke(message_hisotry).model_dump()  
        progress = parsed_response["progress_percentage"]
        self.formatted_history_record["progress"] = progress
        print("The current progress is ",progress)
        self.memory_manager.agent_chat.update_one(
            {
                "thread_id": self.session_id,
                "timestamp": self.current_time
            },
            {
                "$set": {
                    "formatted_history": self.formatted_history_record,
                    "agent": self.agent_name
                }
            },
            upsert=True
        )
        return {"progress":progress}    
    
    def sync_state(self, state:AgentState):
        tool_message = state["messages"][-1]
        structured_tool_message, _ = self.structure_tool_message(tool_message)
        state["messages"][-1] = structured_tool_message
        
        
        return {"messages":state["messages"],"messages_clean": [tool_message]}

    def _record_target_task(self,state:AgentState):
        target_task = content_read_out(state["messages_clean"][-1])
        return {"target_task":target_task,"progress":0}

    def call_model(self,state: AgentState):
        self.current_time = datetime.utcnow()
        self.formatted_history_record = {}
        message_history = self.get_message_history(state)
        try:
            response = self.llm_with_tools.invoke(message_history)
        except Exception as e:
            raise LLM_Failure(f"LLM Error: {e}") from e
        #handling this claude bug
        if response.content == []:
            Warning("Claude bug")
            response.content = "At current point, I might need user input to continue. let me know what is next. You can also just ask me to continue"
        tool_informations = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name:str = tool_call['name']
                if "args" not in list(tool_call.keys()):
                    args:dict = {}
                else:
                    args:dict = tool_call['args']
                    if args == {}:
                        #handling this claude bug 
                        response = AIMessage(
                                        content=f"I am encountering max output token issue, that means I am putting too much input context into the {tool_name}, please instruct me what to do next, thank you!", id=response.id
                                    )
                        break
                        #raise LLM_Failure(f"Tool call {tool_name} has no arguments")
                tool_information = {"tool_name": tool_name, "args": args,"toolcall_timestamp" : datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")}
                tool_informations.append(tool_information)
                print(f"Tool call: {tool_name} with args: {args}")
                print('response')
                print(response)
                self.formatted_history_record["action"] =content_read_out(response) 
                self.formatted_history_record["tool_used"]= tool_informations

                print(self.session_id)
                print(self.agent_name)
                self.memory_manager.agent_chat.insert_one({
                            "thread_id": self.session_id,
                            "formatted_history": self.formatted_history_record,
                            "agent": self.agent_name,
                            "timestamp": datetime.utcnow()  # Add the current UTC timestamp
                        })
            return {"messages": [response],"messages_clean": [response],"tool_used": tool_informations,"session_id":self.session_id,"memory_db_name":self.memory_db_name}
        
        self.formatted_history_record["action"] = content_read_out(response)
        print('response')
        print(content_read_out(response))

        print(self.session_id)
        print(self.agent_name)
        self.memory_manager.agent_chat.update_one(
            {
                "thread_id": self.session_id,
                "timestamp": self.current_time
            },
            {
                "$set": {
                    "formatted_history": self.formatted_history_record,
                    "agent": self.agent_name
                }
            },
            upsert=True
        )
        return {"messages": [response],"messages_clean": [response],"session_id":self.session_id,"memory_db_name":self.memory_db_name}
    
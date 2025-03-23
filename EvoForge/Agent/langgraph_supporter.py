# abc class
from abc import ABC, abstractmethod
from typing import Optional, Union,List,Callable
from .mongodb_checkpointer import MongoDBSaver
from pydantic import BaseModel, Field
from typing import Optional, Annotated, TypedDict, Union,List
from langgraph.graph.message import add_messages
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from .agent_utils import image_to_base64, get_date, content_read_out
import copy
from operator import add
from EvoForge.config import nosql_service
from datetime import datetime



class LangGraphSupporter(ABC):
    class GeneralState(TypedDict):
        agent_name:str
        messages: Annotated[list, add_messages]
        messages_clean: Annotated[list, add_messages]
        continue_from_error: Optional[bool] = False
    history_key = "messages"
    graph_state = GeneralState
    system_message = None
    recursion_limit = 1000
    def __init__(self,
                 session_id: Optional[str] = "AgentSession" + get_date(),
                 agent_name:str = "default_name",
                 memory_db_name:str = "chat_history",
                 **kwargs):
        self.agent_name = agent_name
        self.memory_db_name = memory_db_name
        self.memory_manager = MongoDBSaver(client = nosql_service, db_name = memory_db_name, session = session_id)
        self.session_id = session_id 
        self.config = {"configurable": {"thread_id": session_id}, "recursion_limit": self.recursion_limit}
        self.tools = []
        self.un_detachable_tools = []
        self.switchable_system_messages=[]
        self._create_agent()
        self._initialize_system_message()
        self.message_blocks = []
        #self.stream_logger = AgentStreamLogger(database=nosql_service, agent_name=agent_name,memory_db_name=memory_db_name)

    @abstractmethod
    def _create_agent(self)->CompiledGraph:
        """Call this function to create the graph as agent,
            self.agent: CompiledGraph
        """
        workflow = StateGraph(self.graph_state)
        self.agent = workflow.compile(checkpointer=self.memory_manager)
        raise NotImplementedError
        
    def _initialize_system_message(self):
        """
        Insert agent system message
        """
        system_message = "You are a helpful assistant!"
        self.rewrite_system_message(system_message)

    def rewrite_system_message(self,system_message:str):
            """
            Overwrite the systen message in the agent memory

            Args:
                system_message (str): The system message
            """
            # check if system message contains placeholder such as {}
            if "{}" in system_message:
                Exception("System message injection currently do not support placeholder, build your custom agent instead for this purpose")
            self.system_message = system_message
            past_state = self.agent.get_state(self.config).values
            if past_state == {}:
                self.agent.update_state(self.config, {"messages": [SystemMessage(content=system_message)],"messages_clean": [SystemMessage(content=system_message)]})
                #self.agent.update_state(self.config, {"messages_clean": [SystemMessage(content=system_message)]})
            else:
                # find the first system message in past_state['messages']
                for i, msg in enumerate(past_state['messages']):
                    if type(msg) == SystemMessage:
                        past_system_message = past_state['messages'][i]
                        past_system_message_clean = past_state['messages_clean'][i]
                        break
                    else:
                        raise Exception("No system message found in the past state")
                past_system_message.content = system_message
                past_system_message_clean.content = system_message
                self.agent.update_state(self.config, {"messages": past_state['messages'],"messages_clean": past_state['messages_clean']})
                #self.agent.update_state(self.config, {"messages_clean": past_state['messages']})

    def rewrite_system_message_without_track(self,system_message:str):
            """
            Overwrite the systen message in the agent memory

            Args:
                system_message (str): The system message
            """
            # check if system message contains placeholder such as {}
            if "{}" in system_message:
                Exception("System message injection currently do not support placeholder, build your custom agent instead for this purpose")
            past_state = self.agent.get_state(self.config).values
            if past_state == {}:
                self.agent.update_state(self.config, {"messages": [SystemMessage(content=system_message)],"messages_clean": [SystemMessage(content=system_message)]})
                #self.agent.update_state(self.config, {"messages_clean": [SystemMessage(content=system_message)]})
            else:
                # find the first system message in past_state['messages']
                for i, msg in enumerate(past_state['messages']):
                    if type(msg) == SystemMessage:
                        past_system_message = past_state['messages'][i]
                        past_system_message_clean = past_state['messages_clean'][i]
                        break
                    else:
                        raise Exception("No system message found in the past state")
                past_system_message.content = system_message
                past_system_message_clean.content = system_message
                self.agent.update_state(self.config, {"messages": past_state['messages'],"messages_clean": past_state['messages_clean']})
                #self.agent.update_state(self.config, {"messages_clean": past_state['messages']})

    def rewrite_system_message_linker(self,system_message:str):
            """
            Overwrite the systen message in the agent memory

            Args:
                system_message (str): The system message
            """
            # check if system message contains placeholder such as {}
            if "{}" in system_message:
                Exception("System message injection currently do not support placeholder, build your custom agent instead for this purpose")
            self.system_message = system_message
            past_state = self.agent.get_state(self.config).values
            if past_state == {}:
                self.agent.update_state(self.config, {"messages": [SystemMessage(content=system_message)],"messages_clean": [SystemMessage(content=system_message)]})
                #self.agent.update_state(self.config, {"messages_clean": [SystemMessage(content=system_message)]})
            else:
                # find the first system message in past_state['messages']
                for i, msg in enumerate(past_state['messages']):
                    if type(msg) == SystemMessage:
                        past_system_message = past_state['messages'][i]
                        past_system_message_clean = past_state['messages_clean'][i]
                        break
                    else:
                        raise Exception("No system message found in the past state")
                past_system_message.content = system_message
                past_system_message_clean.content = system_message
                self.agent.update_state(self.config, {"messages": past_state['messages'],"messages_clean": past_state['messages_clean']})
                #self.agent.update_state(self.config, {"messages_clean": past_state['messages']})

    def add_switchable_text_block_system_message(self,
                                                 block_name:str,
                                                 block_description:str, 
                                                 block_content:str):
        """
        Not In Used at the moment, DO NOT USE THIS EVER. 
        LLMs are not sensitive to system message at all. 
        """
        # append a grounding message section to the system message. When this function is called, always remove the previous grounding message
        block_prefix = block_name + ":\n"
        block_message = f"\n{block_prefix}\n{block_description}\n{block_content}"
        # check if system message contains grounding message
        # TODO: this assumes grounding always comes after the system message
        if block_prefix in self.system_message:
            self.system_message = self.system_message.split(block_prefix)[0]
        self.system_message += block_message
        self.rewrite_system_message(self.system_message)

    def inject_text_block_human_message(self,
                                        block_name:str,
                                        block_description:str, 
                                        block_update_function:Callable):
            """ Add a switchable text block to the agent memory"""
            block = (block_name, block_description, block_update_function)
            self.message_blocks.append(block)

    def append_system_message(self,system_message:str):
        # check if system message contains placeholder such as {}
        if system_message in self.system_message:
            Warning("System message already exists in the agent")
        else:
            self.system_message = self.system_message+system_message
        self.rewrite_system_message(self.system_message)

    def append_switchable_system_messsage(self,system_message:str,pos=0):
        if len(self.switchable_system_messages) == 0:
            self.switchable_system_messages.append(system_message)
        else:
            self.switchable_system_messages[pos] = system_message
        # join the switchable_system_messages
        self.rewrite_system_message_without_track(self.system_message+"\n".join(self.switchable_system_messages))

    def add_tool(self,tool: Union[BaseTool, list]):
        if isinstance(tool, list):
            for t in tool:
                if t in self.tools:
                    continue
                self.tools.append(t)
        
        else:
            if tool in self.tools:
                print(f"tool already exists in the agent")
            else:   
                self.tools.append(tool)
        self._create_agent()

    def add_un_detachable_tools(self,tool: Union[BaseTool, list]):
        if isinstance(tool, list):
            for t in tool:
                if t in self.un_detachable_tools:
                    continue
                self.un_detachable_tools.append(t)
        
        else:
            if tool in self.un_detachable_tools:
                print(f"tool already exists in the agent")
            else:   
                self.un_detachable_tools.append(tool)
        self._create_agent()

    def detach_tool(self):
        self.tools = []
        self._create_agent()

    def invoke(self,user_input,image_urls:Optional[List[str]]=None)->List[BaseMessage]:
        """image_urls: list of image urls, can be local dir"""
        collection = self.memory_manager.user_chat
        current_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")
        collection.insert_one({"thread_id":f"{self.session_id}","query":user_input,"timestamp":current_timestamp,"agent":self.agent_name} )
        user_input,user_input_clean = self._structure_user_input(user_input,image_urls)
        if user_input is None:
            self.change_continue_from_error(True)
            response = self.agent.invoke(input=user_input,config=self.config)
        else:
            response = self.agent.invoke(input={"messages":[user_input],
                                                "messages_clean":[user_input_clean],
                                                "continue_from_error":False,"agent_name":self.agent_name},config=self.config)
        return response["messages"]

    def invoke_return_graph_state(self,
                                  user_input,
                                  image_urls:Optional[List[str]]=None)->dict:
        """invoke method but return the state"""
        user_input,user_input_clean = self._structure_user_input(user_input,image_urls)
        if user_input is None:
            self.change_continue_from_error(True)
            response = self.agent.invoke(input=user_input,config=self.config)
        else:
            response = self.agent.invoke({"messages":[user_input],
                                          "messages_clean":[user_input_clean],
                                          "continue_from_error":False,"agent_name":self.agent_name},self.config)
        return response

    def stream(self,
               user_input,
               image_urls:Optional[List[str]]=None,
               print_=True)->List[BaseMessage]:
        """image_urls: list of image urls, can be local dir"""
        user_input,user_input_clean = self._structure_user_input(user_input,image_urls)
        messages = []
        if user_input is None:
            self.change_continue_from_error(True)
            for event in self.agent.stream(None,config=self.config,stream_mode='values'):
                #self.stream_logger.store(event)
                if "messages" in event.keys():
                    messages.append(event["messages"][-1])
                    if print_:
                        content = content_read_out(event["messages"][-1])
                        print(content)
            return messages
        else:
            for event in self.agent.stream({"messages": [user_input],
                                            "messages_clean": [user_input_clean],
                                            "continue_from_error":False,"agent_name":self.agent_name},config=self.config,stream_mode='values'):
                #self.stream_logger.store(event)
                if "messages" in event.keys():
                    messages.append(event["messages"][-1])
                    if print_:
                        content = content_read_out(event["messages"][-1])
                        print(content)
            return messages

    
    def stream_return_graph_state(self,
                                  user_input,
                                  image_urls:Optional[List[str]]=None,
                                  print_=False)->dict:
        """image_urls: list of image urls, can be local dir"""
        collection = self.memory_manager.user_chat
        current_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")
        collection.insert_one({"thread_id":f"{self.session_id}","query":user_input,"timestamp":current_timestamp,"agent":self.agent_name} )
        user_input,user_input_clean = self._structure_user_input(user_input,image_urls)
        messages = []
        if user_input is None:
            self.change_continue_from_error(True)
            for event in self.agent.stream(None,config=self.config,stream_mode='values'):
                #self.stream_logger.store(event)
                if "messages" in event.keys():
                    messages.append(event["messages"][-1])
                    if print_:
                        content = content_read_out(event["messages"][-1])
                        print(content)
            return event
        else:
            for event in self.agent.stream({"messages": [user_input],
                                            "messages_clean": [user_input_clean],
                                            "continue_from_error":False,
                                            "agent_name":self.agent_name},
                                            config=self.config,
                                            stream_mode='values'):
                #self.stream_logger.store(event)
                if "messages" in event.keys():
                    messages.append(event["messages"][-1])
                    if print_:
                        content = content_read_out(event["messages"][-1])
                        print(content)
            return event

    def clear_memory(self,print_log=True):
        thread_id = self.config["configurable"]["thread_id"]
        self.memory_manager.delete(thread_id,print_log=print_log)
        self._initialize_system_message()

    def _structure_user_input(self,user_input:Union[str,None],image_urls:Optional[List[str]]=None,print_=True):
        user_input_clean = copy.deepcopy(user_input)
        if user_input is None:
            return None, "filler"
        if len(self.message_blocks)>0:
            additional_message = ""
            for block in self.message_blocks:
                block_name, block_description, block_update_function = block
                # by default must have **kwargs in the block_update_function
                block_information = block_update_function(user_input=user_input)
                additional_message += f"\n{block_name}:\n{block_description}\n{block_information}\n"
            user_input = user_input + additional_message
        if image_urls is None:
            image_urls = []
        image_data = [image_to_base64(image_url) for image_url in image_urls]
        user_input = HumanMessage(
            content=[{"type": "text", "text": user_input}] + [{"type": "image_url", "image_url": {"url": img}} for img in image_data]
        )
        user_input_clean = HumanMessage(
            content=[{"type": "text", "text": user_input_clean}] + [{"type": "image_url", "image_url": {"url": img}} for img in image_data]
        )
        return user_input, user_input_clean
    
    def structure_tool_message(self,tool_message:ToolMessage):
        tool_message_clean = copy.deepcopy(tool_message)
        if len(self.message_blocks)>0:
            additional_message = ""
            for block in self.message_blocks:
                block_name, block_description, block_update_function = block
                # by default must have **kwargs in the block_update_function
                block_information = block_update_function()
                additional_message += f"\n{block_name}:\n{block_description}\n{block_information}\n"
            tool_message.content = tool_message.content + additional_message

        return tool_message, tool_message_clean
        
    
    def change_continue_from_error(self,continue_from_error:bool):
        self.agent.update_state(self.config,{"continue_from_error":continue_from_error})

    def render_tool_messages(self,tools:List[BaseTool]):
        """
        Render the tool messages to the agent. 
        Usually added to system message (recommended)
        or 
        Switchable textblock during input. 
        """
        tool_messages = "You are given the following actions with the specification of their input args \n"
        for i,tool in enumerate(tools):
            name: str = tool.name
            description: str = tool.description
            args: dict = tool.get_input_schema().model_json_schema()["properties"]
            tool_messages += f"The {i+1} action '{name}' is described as '{description}' with the following arguments: {args};\n"
        return tool_messages

    # def freeze_state(self):
    #     """
    #     Freeze the current state of the agent
    #     """
    #     self.freezed_state=self.agent.get_state(self.config).values

    def get_message_history(self,state):
        """
        Get the message history of the agent, combine clean message and message with injected context. Always leave the last message with injected context
        """
        # find the index of the last human message
        messages = state["messages"]
        last_human_message_index = messages.index([msg for msg in messages if type(msg) == HumanMessage][-1])
        message_history = state["messages_clean"][:-1]
        last_message = state["messages"][-1]
        message_history.append(last_message)
        message_history[last_human_message_index] = messages[last_human_message_index]
        return message_history

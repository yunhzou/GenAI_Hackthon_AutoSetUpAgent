from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolCall
from typing import List, Dict
from langchain_core.tools import BaseTool
import anthropic

def anthropic_message_history_converter(message_history: List[BaseMessage]) -> List[Dict]:
    """
    Converts a message history into a list of strings
    """
    # TODO: not finished yet 
    messages = []
    for m in message_history:
        if isinstance(m,HumanMessage):
            message = {"role":"user","content":m.text}
            messages.append(message)
        elif isinstance(m,AIMessage):
            yield "AI: "+m.text
        elif isinstance(m,ToolCall):
            yield "Tool: "+m.text
        else:
            yield "Unknown: "+m.text

    return system_message, messages


def anthropic_response2AIMessage(response:Dict) -> AIMessage:
    return AIMessage(response["content"])


def convert_to_function_schema(func):
    return {
        "name": func.name,
        "description": func.description,
        "input_schema": {
            "type": "object",
            "properties": {
                key: {
                    "type": value.get("type", "string"),
                    "description": value.get("description", "")
                }
                for key, value in func.args.items()
            },
            "required": list(func.args.keys())
        }
    }


class ChatAnthropic():
    def __init__(self,model:str="claude-3.7-latest",system:str="you are a helpful assistant"):
        self.model = model
        self.system = system
        self.message_history = []
        self.message_history_converter = anthropic_message_history_converter
        self.client = anthropic.Anthropic()
        self.tools_schema = None
        
    def bind_tools(self,tools:List[BaseTool]):
        self.tools_schema = [convert_to_function_schema(tool) for tool in tools]




    def invoke(self,messages:List[BaseMessage],thinking_budget_tokens:int=32000,max_tokens:int=128000):
        if thinking_budget_tokens < 2000:
            raise ValueError("Thinking budget should be at least 2000 tokens")
        
        system_message, message_history = anthropic_message_history_converter(messages) 
        with self.client.beta.messages.stream(
            model=self.model,
            system = system_message,
            max_tokens=max_tokens,
            thinking={
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens
            },
            tools=self.tools_schema,
            messages=message_history,
            betas=["output-128k-2025-02-19"],
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    print(f"\nStarting {event.content_block.type} block...")
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        print(f"{event.delta.thinking}", end="", flush=True)
                    elif event.delta.type == "text_delta":
                        print(f"{event.delta.text}", end="", flush=True)
                elif event.type == "content_block_stop":
                    print("\nBlock complete.")
        anthropic_response2AIMessage(event)
        

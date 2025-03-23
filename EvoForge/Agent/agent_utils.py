import json
from langchain_openai import OpenAIEmbeddings
import httpx
import base64
from datetime import datetime
from EvoForge.config import openai_api_key
from EvoForge.config import nosql_service
from langchain_core.messages import BaseMessage, AIMessage
import mimetypes



def text_embedding(text:str):
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    text_embedding = embeddings.embed_query(text)
    return text_embedding

def json_embedding(json_doc:dict):
    """convert json to json_string and embed the string

    Args:
        json_doc (dict): json document to be embedded

    Returns:
        List(string): the embedding
    """
    json_doc_string = json.dumps(json_doc)
    json_embedding = text_embedding(json_doc_string)
    return json_embedding


def check_url(image_url:str):
    # check if image_url is a valid url
    try:
        response = httpx.get(image_url)
        if response.status_code == 200:
            return True
        else:
            return False
    except:
        return False
    
def image_to_base64(image_path: str) -> str:
    if check_url(image_path):
        image_data = httpx.get(image_path).content
    else:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

    # Encode the image data to base64
    encoded_image = base64.b64encode(image_data).decode("utf-8")

    # Guess MIME type (defaults to 'image/jpeg' if not found)
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    return f"data:{mime_type};base64,{encoded_image}"

def get_date():
    """
    Returns the current date.

    Returns:
    datetime.date: The current date.
    """
    return str(datetime.now().date())   


def extract_all_text_exclude_edges(messages,last_num_messages=2):
    """
    Extracts and concatenates all text from a list of message objects, excluding the first and last messages.
    Handles different content types for HumanMessage and others.

    Parameters:
        messages (list): List of message objects, each with a .content attribute.

    Returns:
        str: Concatenated text content from all messages except the first and last.
    """
    if len(messages) <= 2:
        return ""  # If there are 2 or fewer messages, nothing remains after removing first and last

    extracted_text = []
    # cut the head and the end
    filtered_messages = messages[1:-1]
    # check if index is out of bond, if so, return all messages
    if last_num_messages > len(filtered_messages):
        last_num_messages = len(filtered_messages)
    if last_num_messages == 0:
        return "No Context Needed"
    # get the last n messages
    filtered_messages = filtered_messages[-last_num_messages:]
    # Exclude the first and last message
    for message in filtered_messages:
        if isinstance(message.content, list):
            # For HumanMessage, extract the 'text' field from each dictionary
            extracted_text.extend(item['text'] for item in message.content if 'text' in item)
        else:
            # For other message types, append the content directly
            extracted_text.append(message.content)
    
    return "\n".join(extracted_text)


def content_read_out(message: BaseMessage):
    """Read out the message from the BaseMessage"""
    content = message.content
    if type(content) == list:
        if 'text' in content[0]:
            content = content[0]['text']
        else:
            content = ""
    return content


def tool_read_out(message: BaseMessage):
    """Read out the tool from the BaseMessage"""
    if message.type == "ai":
        tool_calls = message.tool_calls
        return tool_calls
    else:
        return None

def extract_base64_from_image(message:BaseMessage):
    """Extract base64 from the image message"""
    # TODO: implement this function 2025/3/13
    pass

  
class LLM_Failure(Exception):
    """Raised when API fails."""
    def __init__(self, message="LLM API Error"):
        super().__init__(f"LLM API Error: {message}, once fixed, press continue from break point button")
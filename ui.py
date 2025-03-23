import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import datetime
import motor.motor_asyncio
import asyncio
import random
import time
import requests
import os 
import threading
from dotenv import load_dotenv
load_dotenv()

# Retrieve your API key from the environment
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Define the API endpoint (update this to your actual endpoint)
url = "https://api.anthropic.com/your_endpoint"

# Set up the headers with your API key
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Example payload for Anthropic API (only executed once here)
payload = {
    "prompt": "Hello, how can I help you?",
    # other required parameters...
}

response = requests.post(url, json=payload, headers=headers)
print("Status code:", response.status_code)
print("Response:", response.json())

# MongoDB Connection Setup (Motor Async)
MONGO_URI = os.getenv("MONGODB_URL")
DB_NAME = "chat_history"
AGENT_COLLECTION_NAME = "agent_history"
USER_COLLECTION_NAME = "user_history"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection_agent = db[AGENT_COLLECTION_NAME]
collection_user = db[USER_COLLECTION_NAME]

from EvoForge import EvoForge  
from EvoForge.Tool.tools import read_webpage, execute_shell_command, ask_question_to_user, create_file
evoforge_client = EvoForge()
agent = evoforge_client.spawn_setup_agent(session="EvoForge1")

# Async functions to retrieve chat history
async def get_agent_chat_history(thread_id):
    cursor = collection_agent.find({"thread_id": thread_id}, {"_id": 0}).sort("timestamp", 1)
    return await cursor.to_list(length=100000)

async def get_user_chat_history(thread_id): 
    cursor = collection_user.find({"thread_id": thread_id}, {"_id": 0}).sort("timestamp", 1)
    return await cursor.to_list(length=100000)

# Run an async coroutine in a new thread with its own event loop.
def run_async_in_new_thread(coro):
    result = None
    def runner():
        nonlocal result
        result = asyncio.run(coro)
    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    return result

def send_message(user_input):
    payload = {"thread_id": thread_id, "user_input": user_input}
    # Ensure API_URL is defined or replace with your actual URL string
    response = requests.post(f"{API_URL}/chat", json=payload)
    st.write("Send Message API Response:", response.status_code)  # Debug
    if response.status_code == 200:
        return True
    st.write("Send Message Error:", response.text)  # Debug error
    return False

st.markdown(
    """
    <style>
    /* Dark gradient background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #0D1B2A, #1B263B);
        background-attachment: fixed;
        color: white; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to run async tasks safely in Streamlit
def run_async_task(task):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # If no running loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(task)

# Retrieve thread ID
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "GENAI_0"
thread_id = st.session_state.thread_id

st.title("🤖 EvoForge")
st.markdown("EvoForge will assist your repo research process according to your needs!")

# Fetch chat history (use correct event loop handling)
if "chat_history" not in st.session_state:
    st.session_state.chat_agent_history = run_async_task(get_agent_chat_history(thread_id))
    st.session_state.chat_user_history = run_async_task(get_user_chat_history(thread_id))
    
    

st.markdown("### Chat History")
for entry in st.session_state.chat_agent_history:
    timestamp = entry["timestamp"]
    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    message = entry["formatted_history"].get("action", "")
    
    if entry["agent"] == "human_user":
        # Human message: Align right in a 2-column layout.
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.empty()
        with col2:
            st.markdown(
                f"""
                <div style="text-align:right; margin-bottom:20px;">
                    <div style="
                        display:inline-block; 
                        background-color:#0077B6; 
                        color: white;
                        padding:10px; 
                        border-radius:10px; 
                        max-width:80%;">
                        <strong>[{formatted_time}] You:</strong> {message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:  # assistant_bot
        # AI message: Align left in a 2-column layout.
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.markdown(
                f"""
                <div style="text-align:left; margin-bottom:20px;">
                    <div style="
                        display:inline-block; 
                        background-color:#00B4D8; 
                        color: black;
                        padding:10px; 
                        border-radius:10px; 
                        max-width:80%;">
                        <strong>[{formatted_time}] EvoForge:</strong> {message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            st.empty()

# Text input with on_change callback for when Enter is pressed
def on_enter():
    user_input = st.session_state.get("user_input", "").strip()
    if user_input:
        agent.stream_return_graph_state(user_input)
        with st.spinner("Processing... Please wait."):
            time.sleep(2)  # Simulate delay
            if send_message(user_input):
                st.rerun()

st.text_input("Enter your message:", key="user_input", on_change=on_enter)

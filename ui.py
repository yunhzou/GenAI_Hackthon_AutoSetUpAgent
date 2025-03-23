import streamlit as st
import datetime
import motor.motor_asyncio
import asyncio
import random
import time
import requests
import os 

# MongoDB Connection Setup (Motor Async)
MONGO_URI = os.getenv("MONGODB_URL")
DB_NAME = "chat_history"
AGENT_COLLECTION_NAME = "agent_history"
USER_COLLECTION_NAME = "user_history"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection_agent = db[AGENT_COLLECTION_NAME]
collection_user = db[USER_COLLECTION_NAME]
# Function to retrieve chat history (async)
async def get_agent_chat_history(thread_id):
    cursor = collection_agent.find({"thread_id": thread_id}, {"_id": 0}).sort("timestamp", 1)
    return await cursor.to_list()

async def get_user_chat_history(thread_id): 
    cursor = collection_user.find({"thread_id": thread_id}, {"_id": 0}).sort("timestamp", 1)
    return await cursor.to_list()

def send_message(user_input):
    payload = {"thread_id": thread_id, "user_input": user_input}
    response = requests.post(f"{API_URL}/chat", json=payload)
    st.write("Send Message API Response:", response.status_code)  # Debug
    if response.status_code == 200:
        return True
    st.write("Send Message Error:", response.text)  # Debug error
    return False

# Function to generate an AI response (replace with actual AI model)
def generate_ai_response(user_input):
    responses = [
        f"That's an interesting question about '{user_input}'. Let me think...",
        f"I'm not sure, but I can look up more about '{user_input}'.",
        f"Regarding '{user_input}', there are a few perspectives to consider.",
        f"Let me provide some insights on '{user_input}'...",
        f"Thanks for asking about '{user_input}'. Here’s what I know..."
    ]
    return random.choice(responses)

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
for entry in st.session_state.chat_history:
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

user_input = st.text_input("Enter your message:")

if st.button("Send") and user_input.strip():
    with st.spinner("Processing... Please wait."):
        time.sleep(2)  # Placeholder task to simulate delay
        if send_message(user_input):
            st.rerun()  # Refresh chat after sending




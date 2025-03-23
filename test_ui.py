import streamlit as st
import threading
import time
import pymongo

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["chat_history"]
collection = db["agent_history"]

# Get distinct thread_ids
thread_ids = collection.distinct("thread_id")

# Set page title
st.set_page_config(page_title="Multi-Task Progress Tracker", page_icon="📊")
st.title("📊 Multi-Task Progress Tracker")

# Initialize task storage
tasks = {}

# Create progress bars dynamically
for thread_id in thread_ids:
    st.write(f"### Thread {thread_id}")
    tasks[thread_id] = {
        "bar": st.progress(0, text=""),
        "status": st.empty()
    }

# Function to simulate task progress
def update_progress(thread_id):
    progress = 0
    while progress < 100:
        progress += 2
        time.sleep(0.05)

        # Only show label if progress > 1%
        label = f"Thread {thread_id} progress: {progress}%" if progress > 1 else ""
        tasks[thread_id]["bar"].progress(progress, text=label)

    tasks[thread_id]["status"].success(f"✅ Thread {thread_id} - Task Complete!")

# Start each task in a thread
for thread_id in thread_ids:
    threading.Thread(target=update_progress, args=(thread_id,), daemon=True).start()

st.success("✅ All MongoDB tasks are running in parallel!")

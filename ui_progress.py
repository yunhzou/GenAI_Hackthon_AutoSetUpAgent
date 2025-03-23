import streamlit as st
import motor.motor_asyncio
import asyncio
import time

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "chat_history"
COLLECTION_NAME = "agent_history"

# Initialize MongoDB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

async def fetch_progress(thread_id):
    """Fetch progress from MongoDB based on the progress dictionary."""
    document = await collection.find_one({"thread_id": thread_id}, {"progress": 1, "_id": 0})
    if not document or "progress" not in document:
        return 0  # Default to 0% progress if missing

    progress_data = document["progress"]
    current = progress_data.get("current", 0)
    total = progress_data.get("total", 100)  # Default max to 100

    # Calculate percentage
    progress_percent = min(int((current / total) * 100), 100)
    return progress_percent

async def track_progress(thread_id):
    """Continuously update UI with real-time progress."""
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for progress in range(101):  # 0 to 100
        progress_bar.progress(progress)
        progress_text.text(f"Progress: {progress}%")
        time.sleep(0.1)  # Simulate real-time updates
    st.success("✅ Task Completed!")
    for progress in range(101):  # 0 to 100
        progress_bar.progress(progress)
        progress_text.text(f"Progress: {progress}%")
        time.sleep(0.1)  # Simulate real-time updates
    st.success("✅ Task Completed!")

   
    # while True:
    #     progress = await fetch_progress(thread_id)
    #     progress_bar.progress(progress)
    #     progress_text.text(f"Progress: {progress}%")

    #     if progress >= 100:
    #         st.success("✅ Task Completed!")
    #         break

    #     await asyncio.sleep(.1)  # Poll MongoDB every 2 seconds

# Streamlit UI
st.set_page_config(page_title="📊 Progress Tracker", page_icon="📈")

st.title("📊 Real-Time Progress Tracker")
st.markdown("Tracking model progress in action!")

# Select a thread_id (hardcoded, or get dynamically)
thread_id = "GENAI_0"

# Start tracking progress
asyncio.run(track_progress(thread_id))

# Navigation button
if st.button("🔙 Go Back"):
    st.session_state["page"] = "main"
    st.rerun()

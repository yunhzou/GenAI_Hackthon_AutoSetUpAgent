import streamlit as st
import time
import threading

# Set page title
st.set_page_config(page_title="Multi-Task Progress Tracker", page_icon="📊")
st.title("📊 Multi-Task Progress Tracker")

# Define tasks
tasks = {
    "Task 1": {"bar": None, "status": None},
    "Task 2": {"bar": None, "status": None},
    "Task 3": {"bar": None, "status": None},
}

# Initialize progress bars
for task in tasks:
    st.write(f"### {task}")
    tasks[task]["bar"] = st.progress(0, text=f"Loading {task}...")
    tasks[task]["status"] = st.empty()

# Function to update progress bars asynchronously
def update_progress(task_name):
    """Simulates progress updates for each task independently."""
    progress = 0
    while progress < 100:
        progress += 2  # Increment progress
        progress = min(progress, 100)  # Ensure it does not exceed 100%
        
        # Update progress bar
        tasks[task_name]["bar"].progress(progress, text=f"{task_name} progress: {progress}%")
        time.sleep(0.05)  # Simulate processing time

    # Display completion message
    tasks[task_name]["status"].success(f"✅ {task_name} - Task Complete!")

# Launch progress updates in separate threads
for task_name in tasks:
    threading.Thread(target=update_progress, args=(task_name,), daemon=True).start()

st.success("✅ All tasks are running in parallel!")

if st.button("🔄 Restart Simulation"):
    st.rerun()  # Refresh UI to restart progress

from Agent import LangGraphAgent
agent = LangGraphAgent(model="claude-3-7-sonnet-20250219",session_id="GENAI_5")
from Tool.tools import multiply, read_webpage
agent.add_tool(read_webpage)
def create_file():
    filename = input("Enter the filename: ")
    try:
        with open(filename, 'w') as file:
            file.write("")  # Optionally write initial content
        print(f"File '{filename}' created successfully!")
    except Exception as e:
        print(f"Error creating file '{filename}': {e}")
result = agent.stream_return_graph_state("Can you try this and make an https://huggingface.co/Helsinki-NLP/opus-mt-en-fr")
print(result["messages_clean"][-1].content)
agent.clear_memory()
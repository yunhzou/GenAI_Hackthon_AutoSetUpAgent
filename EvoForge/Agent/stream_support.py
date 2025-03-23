from .agent_utils import content_read_out, tool_read_out
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# store states properly, 

class AgentStreamLogger():
    def __init__(self,
                 database,
                 agent_name: str,
                 memory_db_name: str,
                 ):
    # each agent is only associated with one document, no time sequence, only used for streaming.
        self.database = database[memory_db_name]
        self.agent_name = agent_name
        self.collection = self.database["streaming_state"]
        self.initialize_document()
        
    def initialize_document(self):
        """
        Inserts a new document for the agent if it doesn't already exist.
        This creates an ObjectId for persistence and ensures a unique document for the agent.
        """
        # Check if a document for the agent already exists
        existing_document = self.collection.find_one({"agent_name": self.agent_name})
        
        if existing_document:
            print(f"Document for agent '{self.agent_name}' already exists.")
            return existing_document["_id"]  # Return the existing ObjectId
        
        # Create the initial document
        initial_document = {
            "agent_name": self.agent_name,
            "messages": [],
            "status": "initialized",  # Example field to indicate the document is initialized
        }
        
        # Insert the document and retrieve the ObjectId
        result = self.collection.insert_one(initial_document)
        print(f"Initialized document for agent '{self.agent_name}' with ObjectId {result.inserted_id}.")
        
        return result.inserted_id

    def handle_messages(self, messages):
        message4storage = []
        for message in messages:
            if isinstance(message, AIMessage):
                content = content_read_out(message)
                tool_call = tool_read_out(message)
                ai_message = {"type":"ai","text": content, "tool_call": tool_call}
                message4storage.append(ai_message)
            elif isinstance(message, HumanMessage):
                content = content_read_out(message)
                human_message = {"type":"human","text": content}
                message4storage.append(human_message)
            elif isinstance(message, ToolMessage):
                content = content_read_out(message)
                tool_call = tool_read_out(message)
                tool_message = {"type":"tool","text": content, "tool_call": tool_call}
                message4storage.append(tool_message)
        return message4storage
    
    def store(self, content):
        content = content.copy()
        if "messages" in content:
            message = content['messages']
            message4storage = self.handle_messages(message)
            content['messages'] = message4storage

        # Use agent_name as a unique identifier
        agent_name = content.get("agent_name")
        if not agent_name:
            raise ValueError("Content must include 'agent_name' as a unique identifier.")
        
        # Check if the document exists
        existing_document = self.collection.find_one({"agent_name": agent_name})
        if existing_document:
            # Update if the document exists
            self.collection.update_one(
                {"agent_name": agent_name},
                {"$set": content}
            )
        else:
            # Insert if the document does not exist
            self.collection.insert_one(content)




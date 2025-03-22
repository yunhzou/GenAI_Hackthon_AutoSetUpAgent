from dotenv import load_dotenv
import os
from pymongo import MongoClient
# dot env
def connect_to_mongodb(uri):
    client = MongoClient(uri)
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)
        raise Exception("Unable to connect to the MongoDB deployment. Check your URI")        


if os.getenv("override") == "false":
    load_dotenv(".env")
else:
    load_dotenv(".env",override=True)
    os.environ["override"] = "false"
openai_api_key = os.getenv("OPENAI_API_KEY")
mongodb_connection_string = os.getenv("MONGODB_URL")
nosql_service = connect_to_mongodb(uri = mongodb_connection_string)


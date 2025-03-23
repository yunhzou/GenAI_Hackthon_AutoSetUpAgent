from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, this is your backend!"}

@app.post("/chat")
async def chat_endpoint(payload: dict):
    # Process the incoming payload (e.g., a prompt) and return a response
    return {"response": "This is a response from the backend."}
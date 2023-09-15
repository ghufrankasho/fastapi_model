from Api import query
from fastapi import FastAPI
 
 

app = FastAPI()


@app.post("/ask")
def ask(text):
    output = query(text)
    return output

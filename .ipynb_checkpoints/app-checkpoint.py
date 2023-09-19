from fastapi import FastAPI
from transcribe import transcribe_audio

app = FastAPI()


@app.get("/get_summary")
def summarize(link: str, meeting_id: str):
    json_response = transcribe_audio(link, meeting_id)
    return json_response


@app.post("/classroom_summary")
def summary(link: str, meeting_id: str):
    json_response = transcribe_audio(link, meeting_id)

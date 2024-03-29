from fastapi import FastAPI
from breakout_room import evaluate_discussion
from transcribe import transcribe_audio
from critical_analysis import analyse_critical_thinking


app = FastAPI()


@app.get("/get_summary")
def summarize(link: str, meeting_id: str):
    json_response = transcribe_audio(link, meeting_id)
    return json_response


@app.post("/classroom_summary")
def summary(link: str, meeting_id: str):
    json_response = transcribe_audio(link, meeting_id)


@app.get("/analyse_discussion")
def discussion_evaluation(link: str, meeting_id: str, title: str):
    json_response = evaluate_discussion(link, meeting_id, title)
    if json_response:
        return "Summary created successfully!"

@app.get('/analyse_critical_thinking')
def analyse_thinking(text: str):
    output = analyse_critical_thinking(text)
    return output
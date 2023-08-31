from fastapi import FastAPI
from transcribe import transcribe_audio
from fastapi.responses import JSONResponse


app = FastAPI()


@app.get("/get_summary")
def summarize(link: str):
    json_response = transcribe_audio(link)

    # return JSONResponse(content=json_response, media_type='application/json')
    return json_response

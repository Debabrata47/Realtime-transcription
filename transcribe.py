import whisper
import firebase_admin
from firebase_admin import credentials, db
from threading import *
import os
import time
import librosa
import json
from transformers import pipeline

MODEL = whisper.load_model('small.en')
SUMMARIZER = pipeline('summarization', model='facebook/bart-large-cnn')

firebaseConfig = {
    'apiKey': "AIzaSyCGmeVM6OPnRbVGaW6O7DVqafArIGEm5Ys",
    'authDomain': "silwalk-inc.firebaseapp.com",
    'projectId': "silwalk-inc",
    'databaseURL': "https://silwalk-inc-default-rtdb.firebaseio.com",
    'storageBucket': "silwalk-inc.appspot.com",
    'messagingSenderId': "665210785578",
    'serviceAccount': "ServiceKey.json",
    'appId': "1:665210785578:web:6279247f0704ec73be5853",
    'measurementId': "G-EW5W77X7X7"
}

cred = credentials.Certificate("ServiceKey.json")
firebase_admin.initialize_app(cred, firebaseConfig)

rtdb = db.reference()


class video(Thread):
    def __init__(self, link, meeting_id):
        self.link = link
        Thread.__init__(self)
        if not os.path.exists(f'meetings/{meeting_id}'):
            os.makedirs(f'meetings/{meeting_id}')

    def run(self, meeting_id):
        cmd = f'ffmpeg -y -i {self.link} -vn -c:a copy meetings/{meeting_id}/audiotst.wav'
        os.system(cmd)


def transcribe_audio(hrl_link, meeting_id):
    t1 = video(link=hrl_link)
    t1.start()
    while True:
        if 'audiotst.wav' in os.listdir(f'meetings/{meeting_id}'):
            break
    count = 0
    trans = {}
    summary = {}
    time.sleep(40)
    while True:
        start = time.time()
        cmd = f'ffmpeg -y -i meetings/hlsnew/audiotst.wav -ss {count} -t 60 -vn -c:a copy meetings/{meeting_id}/audioclip.wav'
        os.system(cmd)

        d, sr = librosa.load(f'meetings/{meeting_id}/audioclip.wav', sr=None)

        if len(d) == 0:
            break
        out = MODEL.transcribe(f'meetings/{meeting_id}/audioclip.wav')['text']
        print(out)

        ind = (count // 60) + 1
        trans[ind] = out

        file = open(f'meetings/{meeting_id}/transcript.json', 'w')
        json.dump(trans, file, ensure_ascii=False)
        file.close()

        rtdb.child(meeting_id).child('transcription').set(trans)

        if ind % 3 == 0:
            t3m = ''

            file = open(f'meetings/{meeting_id}/transcript.json', 'r')
            nas = json.load(file)
            file.close()

            for i in range(ind - 2, ind + 1):
                t3m += (nas[str(i)] + ' ')

            s3m = SUMMARIZER(t3m, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

            summary[ind // 3] = s3m

            file = open(f'meetings/{meeting_id}/summary.json', 'w')
            json.dump(summary, file, ensure_ascii=False)
            file.close()

        rtdb.child(meeting_id).child('summary').set(summary)

        count += 60

        end = time.time()
        t = end - start
        print(cmd, t)

        if t > 0:
            time.sleep(60 - t)

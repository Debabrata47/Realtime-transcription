import whisper
from threading import *
import os
import time
import librosa
import json
from transformers import pipeline

MODEL = whisper.load_model('small.en')
SUMMARIZER = pipeline('summarization', model='facebook/bart-large-cnn')


class video(Thread):
    def __init__(self, link):
        self.link = link
        Thread.__init__(self)
        if not os.path.exists('hls/hlsnew'):
            os.makedirs('hls/hlsnew')

    def run(self):
        cmd = f'ffmpeg -y -i {self.link} -vn -c:a copy hls/hlsnew/audiotst.wav'
        os.system(cmd)


def transcribe_audio(hrl_link):
    t1 = video(link=hrl_link)
    t1.start()
    while True:
        if 'audiotst.wav' in os.listdir('hls/hlsnew'):
            break
    count = 0
    trans = {}
    summary = {}
    time.sleep(20)
    while True:
        start = time.time()
        cmd = f'ffmpeg -y -i hls/hlsnew/audiotst.wav -ss {count} -t 60 -vn -c:a copy hls/hlsnew/audioclip.wav'
        os.system(cmd)

        d, sr = librosa.load('hls/hlsnew/audioclip.wav', sr=None)

        if len(d) == 0:
            break
        out = MODEL.transcribe('hls/hlsnew/audioclip.wav')['text']
        print(out)

        ind = (count // 60) + 1
        trans[ind] = out

        file = open('hls/hlsnew/transcript1.json', 'w')
        json.dump(trans, file, ensure_ascii=False)
        file.close()

        if ind % 3 == 0:
            t3m = ''
            file = open('hls/hlsnew/transcript1.json', 'r')
            nas = json.load(file)
            file.close()

            for i in range(ind - 2, ind + 1):
                t3m += (nas[str(i)] + ' ')

            s3m = SUMMARIZER(t3m, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

            summary[ind // 3] = s3m

            file = open('hls/hlsnew/summary.json', 'w')
            json.dump(summary, file, ensure_ascii=False)
            file.close()
            break

        count += 60

        end = time.time()
        t = end - start
        print(cmd, t)

        if t < 60:
            time.sleep(60 - t)
        else:
            time.sleep(t % 60)

    transcript_file = open('hls/hlsnew/transcript1.json', 'r')
    transcript_data = json.load(transcript_file)
    transcript_file.close()

    summary_file = open('hls/hlsnew/summary.json', 'w')
    summary = json.load(file)
    summary_file.close()

    print("Final response:")
    print(transcript_data)

    return summary

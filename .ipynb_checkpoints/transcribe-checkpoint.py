import whisper
import firebase_admin
from firebase_admin import credentials, db
from threading import *
import os
import time
import librosa
import json
import spacy
from transformers import pipeline
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.callbacks import get_openai_callback
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

os.environ['OPENAI_API_KEY'] = ''
model = whisper.load_model('small.en')
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
        cmd = f'ffmpeg -y -i {self.link} -vn -c:a copy meetings/{meeting_id}/audiomain.wav'
        os.system(cmd)


def transcribe_audio(hrl_link, meeting_id):
    t1 = video(link=hrl_link, meeting_id=meeting_id)
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
        cmd = f'ffmpeg -y -i meetings/{meeting_id}/audiomain.wav -ss {count} -t 60 -vn -c:a copy meetings/{meeting_id}/audioclip.wav'
        os.system(cmd)

        d, sr = librosa.load(f'meetings/{meeting_id}/audioclip.wav', sr=None)

        if len(d) == 0:
            break
        out = model.transcribe(f'meetings/{meeting_id}/audioclip.wav')['text']
        print(out)

        ind = (count // 60) + 1
        trans[ind] = out

        # file = open(f'meetings/{meeting_id}/transcript.json', 'w')
        # json.dump(trans, file, ensure_ascii=False)
        # file.close()

        rtdb.child(meeting_id).child('transcription').set(trans)

        if ind % 3 == 0:
            t3m = ''

            # file = open(f'meetings/{meeting_id}/transcript.json', 'r')
            # nas = json.load(file)
            # file.close()

            for i in range(ind - 2, ind + 1):
                t3m += (trans[str(i)] + ' ')

            s3m = SUMMARIZER(t3m, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

            summary[ind // 3] = s3m

            # file = open(f'meetings/{meeting_id}/summary.json', 'w')
            # json.dump(summary, file, ensure_ascii=False)
            # file.close()

        rtdb.child(meeting_id).child('summary').set(summary)

        count += 60

        end = time.time()
        t = end - start
        print(cmd, t)

        if t > 0:
            time.sleep(60 - t)

    transcript = model.transcribe(f'meetings/{meeting_id}/audiomain.wav')

    segment = transcript['segments']
    transcript = transcript['text']
    
    seg = []

    count = 1
    for i in segment:
        if i['end'] > count*600:
            count += 1
        if len(seg) < count:
            a = { 'trans':'',
            'start':-1,
            'end' : -1
                     }
            seg.append(a)
        seg[count-1]['trans'] += i['text']
        if seg[count-1]['start'] == -1:
            seg[count-1]['start'] = i['start']
        seg[count-1]['end'] = i['end']

    text = [i['trans'] for i in seg]
    
    os.system('python -m spacy download en_core_web_sm')
    
    nlp = spacy.load("en_core_web_sm")

    s=set()
    s.update(['okay','right','know','like','thank','you'])

    textpp = []
    for i in text:
        sentence = [token.lemma_.lower()
                    for token in nlp(i)
                    if token.is_alpha and not token.is_stop and token.lemma_.lower() not in s]
        textpp.append(' '.join(sentence))

    chat = ChatOpenAI(temperature=0)

    template = """ You are a helpful assistant that generates topics from list of keywords"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = ("""
    Identify a topic from a list of keyword from topic modelling by stitching it in 3-5 words {topic}.
    """)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    nltk.download('stopwords')
    stop = nltk.corpus.stopwords.words('english')

    vect =TfidfVectorizer(stop_words=stop,max_features=1000)
    vect_text=vect.fit_transform(textpp)

    rep_lda = [[] for i in range(lda_model.components_.shape[0])]
    vocab = vect.get_feature_names_out()
    for i, comp in enumerate(lda_model.components_):
         vocab_comp = zip(vocab, comp)
         sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
         print("Topic "+str(i)+": ")
         for t in sorted_words:
              rep_lda[i].append(t[0])

    top = []
    for i in rep_lda:
        output = chat(
          chat_prompt.format_prompt(
              topic=i
          ).to_messages()
          )
        out = output.content
        if out[0] == '"':
            out = out[1:-1]
        top.append(out)

    d_lda = {}
    for i,mat in enumerate(lda_top):
        ind = np.argmax(mat)
        tp = top[ind]
        if tp not in d_lda:
            d_lda[tp] = {
                          'trans' : seg[i]['trans'],
                          'start' : seg[i]['start'],
                          'end' : seg[i]['end'],
                          'score' : mat[ind]
            }
        else:
            if d_lda[tp]['score'] < mat[ind]:
                d_lda[tp]['score'] = mat[ind]
                d_lda[tp]['trans'] = seg[i]['trans']
                d_lda[tp]['end'] = seg[i]['end']
                d_lda[tp]['start'] = seg[i]['start']

    rtdb.child(meet_id).child('timestamp').set(d_lda)
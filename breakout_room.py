import os
import json
from utils import structure_summary
from dotenv import load_dotenv
from firebase_db import rtdb
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain, StuffDocumentsChain

BRANCH = 'r1.20.0'
os.system('apt-get install sox libsndfile1 ffmpeg')
os.system(f'python -m pip install git+https://github.com/NVIDIA/NeMo.git@${BRANCH}')

from moviepy.editor import *
import wget
import urllib
import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import librosa

from omegaconf import OmegaConf
from utils_nemo import *
import shutil
from scipy.io import wavfile
from utils_nemo import load_align_model, align
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
import ffmpeg

import whisper
import pandas as pd
import torch
import torchaudio
import nemo
import glob

# Loads environment
load_dotenv()


def evaluate_discussion(link: str, meeting_id: str, title: str):
    if not os.path.exists(f'meetings/{meeting_id}'):
        os.makedirs(f'meetings/{meeting_id}')

    if not os.path.exists(f'meetings/{meeting_id}/{title}'):
        os.makedirs(f'meetings/{meeting_id}/{title}')

    AUDIO_FILENAME = f'meetings/{meeting_id}/{title}/zoom_audio_16000.wav'
    os.makedirs(f'meetings/{meeting_id}/{title}/nemo')
    data_dir = f'meetings/{meeting_id}/{title}/nemo/'

    urllib.request.urlretrieve(link, f'meetings/{meeting_id}/{title}/zoom_meeting.mp4')

    audioclip = AudioFileClip(f'meetings/{meeting_id}/{title}/zoom_meeting.mp4')
    audioclip.write_audiofile(f'meetings/{meeting_id}/{title}/zoom_audio.wav')

    signal, sample_rate = librosa.load(f'meetings/{meeting_id}/{title}/zoom_audio.wav', sr=16000)
    wavfile.write(AUDIO_FILENAME, sample_rate, signal)

    DOMAIN_TYPE = "meeting"  # Can be meeting or telephonic based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"

    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"

    if not os.path.exists(os.path.join(data_dir, CONFIG_FILE_NAME)):
        CONFIG = wget.download(CONFIG_URL, data_dir)
    else:
        CONFIG = os.path.join(data_dir, CONFIG_FILE_NAME)

    cfg = OmegaConf.load(CONFIG)
    print(OmegaConf.to_yaml(cfg))

    meta = {
        'audio_filepath': AUDIO_FILENAME,
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': None,
        'rttm_filepath': None,
        'uem_filepath': None
    }
    with open(os.path.join(data_dir, 'input_manifest.json'), 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')

    cfg.diarizer.manifest_filepath = os.path.join(data_dir, 'input_manifest.json')
    os.system(f'cat {cfg.diarizer.manifest_filepath}')

    pretrained_speaker_model = 'titanet_large'
    cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
    cfg.diarizer.out_dir = data_dir  # Directory to store intermediate files and prediction outputs
    cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    cfg.diarizer.clustering.parameters.oracle_num_speakers = False

    # Using Neural VAD and Conformer ASR
    cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
    # cfg.diarizer.asr.model_path = 'stt_en_conformer_ctc_large'
    cfg.diarizer.oracle_vad = False  # ----> Not using oracle VAD
    cfg.diarizer.asr.parameters.asr_based_vad = False

    asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)

    asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)

    model = whisper.load_model('medium.en')
    out = model.transcribe(f'meetings/{meeting_id}/{title}/zoom_audio_16000.wav')

    device = 'cuda'
    SAMPLE_RATE = 16000
    audio = load_audio(AUDIO_FILENAME, SAMPLE_RATE)
    model_a, metadata = load_align_model(language_code=out["language"], device=device)
    result = align(out["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    word_hyp = {'zoom_audio_16000': []}
    word_ts_hyp = {'zoom_audio_16000': []}
    for i in result['word_segments']:
        if 'start' in i.keys() and 'end' in i.keys():
            word_hyp['zoom_audio_16000'].append(i['word'])
            word_ts_hyp['zoom_audio_16000'].append([i['start'], i['end']])

    diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)

    def read_file(path_to_file):
        with open(path_to_file) as f:
            contents = f.read().splitlines()
        return contents

    predicted_speaker_label_rttm_path = f"{data_dir}/pred_rttms/zoom_audio_16000.rttm"
    pred_rttm = read_file(predicted_speaker_label_rttm_path)

    pred_labels = rttm_to_labels(predicted_speaker_label_rttm_path)

    trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)

    transcription_path_to_file = f"{data_dir}/pred_rttms/zoom_audio_16000.txt"
    transcript = read_file(transcription_path_to_file)

    transcriptsp = [i.split(' ', 3)[-1] for i in transcript]

    discussion = '\n'.join(transcriptsp)

    # Write a function to extract the diarization for the discussion.

    map_template = """
    The following is a set of documents
    {docs} 
    Based on the list of docs, please identify the main themes for each speaker. 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_template = """The following is set of summaries 
    {doc_summaries} 
    Generate a summary that analyses each speaker on the basis of how strong point he made. Did everyone agree. How much he spoke. Quality of views the speaker put forward. Did the speaker change his viewpoint on the basis of other speaker. Did he agree to other fair points.
    Along with a overall key takeways summary separately. Along with that generate another summary that describes the quality of the discussion. Another to suggest what improvements could be made. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")
    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False
    )
    # Receive the discussion after the diarization
    docs = Document(page_content=discussion)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents([docs])
    output = map_reduce_chain.run(all_splits)
    json_summaries = structure_summary(output)
    rtdb.child(meeting_id).child('breakout_room').child(title).child('summary').set(json_summaries)
    return json_summaries

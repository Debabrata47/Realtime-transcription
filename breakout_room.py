import os
import json
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
from utils import structure_summary
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain, StuffDocumentsChain

# Loads environment
load_dotenv()

llm = ChatOpenAI(temperature=0, mdoel='gpt-3.5-turbo-16k')


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


def evaluate_discussion(link: str, meeting_id: str, title: str):
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

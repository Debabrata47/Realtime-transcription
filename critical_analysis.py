from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k')


def analyse_critical_thinking(text: str):
    docs = Document(page_content=text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents([docs])
    prompt_template = """Generate an analysis of each speaker on the basis of their critical thinking ability demonstrated in the classroom. Analyse their critical thinking on the basis of below parameters
    PARAMTERS - Argumentation Skills, Counterargument Handling, Clarity of Thought, Questioning and Curiosity, Problem-Solving Abilities, Critical Reflection.
    {text}

    Take a deep breath and think step by step.

    REPORT:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce an analysis for each speaker on the basis of their critical thinking ability in the classroom discussion.\n"
        "We have provided an critical analysis of the  up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing analysis on the basis of points made by the speakers below."
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original analysis"
        "If the context isn't useful, return the original analysis."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    result = chain({"input_documents": all_splits}, return_only_outputs=True)
    return result['output_text']

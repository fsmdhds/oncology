import json
import time
import openai
from openai import OpenAI
import os
import streamlit as st
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import requests
import time
# from retrying import retry

import os 
from io import StringIO, BytesIO
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
# from langchain.llms import OpenAI
import streamlit as st
from prompts import *
import fitz
from io import StringIO      


disclaimer = """**Disclaimer:** This is a tool to provide information about colon cancer screening for educational use only. Your use of this tool accepts the following:   
1. This tool does not generate validated medical content. \n 
2. This tool is not a real doctor. \n    
3. You will not take any medical action based solely on the output of this tool. \n   

That said - the tool makes use of GPT models constrained to emphasize authoritative sources of information linked on the sidebar. The technique applied is a variation of RAG (retrieval 
augmented generation) https://arxiv.org/abs/2005.11401.
"""

st.set_page_config(page_title='Learn about Colon Cancer Screening', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
st.title("Learn about Colon Cancer Screening")
st.write("ALPHA version 0.5")

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=st.secrets["OPENAI_API_KEY"]
)

with st.expander('Important Disclaimer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.info(disclaimer)
    st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.3, 0.01)
    st.write("Last updated 10/9/23")
    

    
@st.cache_data
def create_retriever(texts, name, save_vectorstore=False):
    
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",
                                  openai_api_base = "https://api.openai.com/v1/",
                                  openai_api_key = st.secrets['OPENAI_API_KEY']
                                  )
    try:
        vectorstore = FAISS.from_texts(texts, embeddings)
        if save_vectorstore:
            vectorstore.save_local(f"{name}.faiss")
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(k=5)

    return retriever


def answer_using_prefix_old(prefix, sample_question, sample_answer, my_ask, temperature, history_context, model, print = True):

    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4":
        model = "gpt-4"
    if model == "openai/gpt-4-1106-preview":
        model = "gpt-4-1106-preview"
    if history_context == None:
        history_context = ""
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # st.write(f'question: {history_context + my_ask}')
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k" or model == "gpt-4-1106-preview":
        openai.api_base = "https://api.openai.com/v1/"
        openai.api_key = st.secrets['OPENAI_API_KEY']
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = 500,
        stream = True,   
        )
    else:      
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = st.secrets["OPENROUTER_API_KEY"]
        # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        route = "fallback",
        messages = messages,
        # headers={ "HTTP-Referer": "http://52.70.26.4:8503/", # To identify your app
        #     "X-Title": "Learn about Colon Cancer Screening"},
        headers={ "HTTP-Referer": "http://localhost:8504/", # To identify your app
            "X-Title": "Learn about Colon Cancer Screening"},
        temperature = temperature,
        max_tokens = 500,
        stream = True,   
        )
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in completion:   
        if print:     
            c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content

def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context, model):
    # st.write('yes the function is being used!')
    messages = [{'role': 'system', 'content': prefix},
        {'role': 'user', 'content': sample_question},
        {'role': 'assistant', 'content': sample_answer},
        {'role': 'user', 'content': history_context + my_ask},]
    # st.write(messages)
    model = "gpt-4-turbo-preview"
    # st.write('here is the model: ' + model)
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(
        base_url="https://api.openai.com/v1",
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    params = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "stream": True,
    }
    # st.write(f'here are the params: {params}')
    try:    
        completion = client.chat.completions.create(**params)
    except Exception as e:
        st.write(e)
        st.write(f'Here were the params: {params}')
        return None

    placeholder = st.empty()
    full_response = ''
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            # full_response.append(chunk.choices[0].delta.content)
            placeholder.markdown(full_response)
    placeholder.markdown(full_response)
    return full_response

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True  
    
def set_llm_chat(model, temperature):
    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4":
        model = "gpt-4"
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":
        return ChatOpenAI(model=model, openai_api_base = "https://api.openai.com/v1/", openai_api_key = st.secrets["OPENAI_API_KEY"], temperature=temperature)
    else:
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
          "X-Title": "GPT and Med Ed"}
        return ChatOpenAI(model = model, openai_api_base = "https://openrouter.ai/api/v1", openai_api_key = st.secrets["OPENROUTER_API_KEY"], temperature=temperature, max_tokens = 500, headers=headers)

@st.cache_data  # Updated decorator name from cache_data to cache
def load_docs(files):
    all_text = ""
    for file in files:
        file_extension = os.path.splitext(file.name)[1]
        if file_extension == ".pdf":
            pdf_data = file.read()  # Read the file into bytes
            pdf_reader = fitz.open("pdf", pdf_data)  # Open the PDF from bytes
            text = ""
            for page in pdf_reader:
                text += page.get_text()
            all_text += text

        elif file_extension == ".txt":
            stringio = StringIO(file.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text 

@st.cache_data
def split_texts(text, chunk_size, overlap, split_method):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        # st.error("Failed to split document")
        st.stop()

    return splits


if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = ""


if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = []


if "temp" not in st.session_state:
    st.session_state["temp"] = 0.3
    
if "your_question" not in st.session_state:
    st.session_state["your_question"] = ""
    
if "texts" not in st.session_state:
    st.session_state["texts"] = ""
    
if "retriever" not in st.session_state:
    st.session_state["retriever"] = ""
    
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-3.5-turbo"
    
if "colon_screen_user_question" not in st.session_state:
    st.session_state["colon_screen_user_question"] = []

if "colon_screen_user_answer" not in st.session_state:
    st.session_state["colon_screen_user_answer"] = []
    
if "history" not in st.session_state:
    st.session_state["history"] = ""


if check_password():
    
    
    # st.header("Chat about Colon Cancer Screening!")
    # st.info("""Embeddings, i.e., reading your file(s) and converting words to numbers, are created using an OpenAI [embedding model](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and indexed for searching. Then,
    #         your selected model (e.g., gpt-3.5-turbo-16k) is used to answer your questions.""")
    with st.sidebar.expander("LLM Selection - default GPT-3.5-turbo-16k"):
        st.session_state.model = st.selectbox("Model Options", ("openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-16k", "openai/gpt-4", "openai/gpt-4-1106-preview"), index=3)
    
    st.sidebar.markdown('[Article 1](https://www.ncbi.nlm.nih.gov/books/NBK570913/)')
    st.sidebar.markdown('[Article 2](https://jamanetwork.com/journals/jama/fullarticle/2779985)')
    st.sidebar.write("And, patient education materals from Northwestern Medicine.")
    # Reenable in order to create another vectorstore!    
        
    # uploaded_files = []
    # uploaded_files = st.file_uploader("Choose your file(s)", accept_multiple_files=True)
    # vectorstore_name = st.text_input("Please enter a name for your vectorstore (e.g., colon_ca):")

    # if uploaded_files is not None:
    #     documents = load_docs(uploaded_files)
    #     texts = split_texts(documents, chunk_size=1250,
    #                                 overlap=200, split_method="splitter_type")

    #     retriever = create_retriever(texts, vectorstore_name, save_vectorstore=True)

    #     # openai.api_base = "https://openrouter.ai/api/v1"
    #     # openai.api_key = st.secrets["OPENROUTER_API_KEY"]

    #     llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)
    #     # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_base = "https://api.openai.com/v1/")

    #     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # else:
    #     st.warning("No files uploaded.")       
    #     st.write("Ready to answer your questions!")
    
    # Disable section below if you'd like to make new vectorstores!
    
    embeddings = OpenAIEmbeddings(base_url="https://api.openai.com/v1",
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    if "vectorstore" not in st.session_state:
        openai.api_base = "https://api.openai.com/v1/"
        openai.api_key = st.secrets['OPENAI_API_KEY']
        st.session_state["vectorstore"] = FAISS.load_local("./parkinson_disease.faiss", embeddings)
        


    # llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)

    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.retriever)



    # colon_screen_chat_option = st.radio("Select an Option", ("For Patients", "For Clinicians"))
    # if colon_screen_chat_option == "For Patients":
    #     reading_level = "5th grade"
    #     colon_ca_context = colon_ca_context_patient
    #     user = "patient"
    # if colon_screen_chat_option == "For Clinicians":
    #     reading_level = "medical professional"
    #     colon_ca_context = colon_ca_context_clinician
    #     user = "clinician"
             
    parkinson_dis_chat_option = st.radio("Select an Option", ("For Patients", "For Clinicians"))
    if parkinson_dis_chat_option == "For Patients":
        reading_level = "5th grade"
        parkinson_dis_context = parkinson_dis_context_patient  # Replace with appropriate variable
        user = "patient"
    if parkinson_dis_chat_option == "For Clinicians":
        reading_level = "medical professional"
        parkinson_dis_context = parkinson_dis_context_clinician  # Replace with appropriate variable
        user = "clinician"

    name = st.text_input("Please enter your first name only:")
    if st.button("Begin here!!! What are the main symptoms of PD?"):
        st.session_state.history = ""   
        initial_question = "Why is screening for colon cancer important?"
        openai.api_base = "https://api.openai.com/v1/"
        openai.api_key = st.secrets['OPENAI_API_KEY']
        docs = st.session_state.vectorstore.similarity_search("What are main symptoms of PD?")
        full_question = f'{name} a {user}: {initial_question} /n/n **Respond `Hi {name},` and answer only using reference {docs} for facts. **Accuracy is essential** for health and safety.'
        with st.spinner("Thinking..."):
            colon_screen_answer = answer_using_prefix(prefix = parkinson_dis_context, sample_question = "What are the main symptos of PD?", sample_answer = "These include rigidity, tremor, and dyskinesia", my_ask = full_question, temperature = 1.0, history_context = st.session_state.history, model = "gpt-3.5-turbo", print = True)
        st.session_state.history += (f'Question: {initial_question} AI answer: {colon_screen_answer}')   
        st.session_state.colon_screen_user_question.append(initial_question)
        st.session_state.colon_screen_user_answer.append(colon_screen_answer)
        
            
    user_question = st.text_input("Please ask another question about PD:")
    
        

    if st.button("Ask more questions about screening for colon cancer"):
        # index_context = f'Use only the reference document for knowledge. Question: {user_question}'
        docs = st.session_state.vectorstore.similarity_search(user_question)
        user_question_context = f"{name}, a {user}, with a follow-up question: {user_question} /n/n Answer only using reference {docs} for facts. Address the user using the right name, {name}, in your response, e.g., {name}, .... **Accuracy is essential** for health and safety of patients."
        # chain = load_qa_chain(llm=llm, chain_type="stuff")
        # with get_openai_callback() as cb:
        #     colon_screen_answer = chain.run(input_documents = docs, question = chain)
        # name
        with st.spinner("Thinking..."): 
            colon_screen_answer = answer_using_prefix(prefix = colon_ca_context, sample_question = colon_ca_sample_question, sample_answer = colon_ca_sample_answer, my_ask = user_question_context, temperature = st.session_state.temp, history_context = st.session_state.history, model = st.session_state.model, print = True)
        # Append the user question and colon_screen answer to the session state lists
        st.session_state.colon_screen_user_question.append(f'{name}: {user_question}')
        st.session_state.colon_screen_user_answer.append(colon_screen_answer)
        st.session_state.history += (f'{name}, a {user}: {user_question} AI answer: {colon_screen_answer}')

        # Display the colon_screen answer
        # st.write(colon_screen_answer)

        # Prepare the download string for the colon_screen questions
        colon_screen_download_str = f"{disclaimer}\n\ncolon_screen Questions and Answers:\n\n"
        for i in range(len(st.session_state.colon_screen_user_question)):
            colon_screen_download_str += f"{st.session_state.colon_screen_user_question[i]}\n"
            colon_screen_download_str += f"Answer: {st.session_state.colon_screen_user_answer[i]}\n\n"
            st.session_state.current_thread = colon_screen_download_str

        # Display the expander section with the full thread of questions and answers
        
    if st.session_state.history != "":    
        with st.sidebar.expander("Your Conversation", expanded=False):
            for i in range(len(st.session_state.colon_screen_user_question)):
                st.info(f"{st.session_state.colon_screen_user_question[i]}", icon="🧐")
                st.success(f"Answer: {st.session_state.colon_screen_user_answer[i]}", icon="🤖")

            if st.session_state.current_thread != '':
                st.download_button('Download', st.session_state.current_thread, key='colon_screen_questions')
    
    if st.sidebar.button("Start a new conversation"):
        st.session_state.history = ""

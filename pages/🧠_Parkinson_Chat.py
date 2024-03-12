import streamlit as st
import faiss
import os
import numpy as np
import openai
from openai import OpenAI
import pickle
from prompts import rag_prompt, references_used

st.set_page_config(page_title='Neurology Chats', layout = 'centered', page_icon = "💬", initial_sidebar_state = 'auto')    

def check_password2():
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
        if st.secrets["use_docker"] == "False":
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
            return False
        else:
            st.session_state["password_correct"] = True
            return True
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
@st.cache_data
def analyze_rag_docs(prompt, model="gpt-3.5-turbo"):
    client = OpenAI(base_url="https://api.openai.com/v1", api_key=st.secrets['OPENAI_API_KEY'])
    # Generate the answer using OpenAI
    # stream = client.chat.completions.create(
    #     model=model,
    #     messages =[{"role": "system", "content": rag_prompt}, 
    #                {"role": "user", "content": prompt}],
    #     # max_tokens=1024,
    #     # n=1,
    #     # stop=None,
    #     temperature=0.3,
    #     stream = True,
    # )
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            messages =[{"role": "system", "content": rag_prompt}, 
                    {"role": "user", "content": prompt}],
            temperature=0.3,
            stream=True,
        )
        st.write_stream(stream)


client = OpenAI(base_url="https://api.openai.com/v1", api_key=st.secrets['OPENAI_API_KEY'])

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the FAISS database
index_dir = "parkinson_disease.faiss"
index_file = os.path.join(index_dir, "index.faiss")

# Load the embeddings
embeddings_file = os.path.join(script_dir, "../parkinson_disease.faiss/index.pkl")
with open(embeddings_file, "rb") as f:
    embeddings = pickle.load(f)

vectorstore = faiss.read_index(index_file)  
# embeddings = np.load("parkinson_disease_embeddings.npy")

# OpenAI API key
openai.api_key = st.secrets['OPENAI_API_KEY']

# Streamlit app
st.title("Parkinson's Disease Question/Answering")

with st.expander("ℹ️ About this App and Settings"):
    st.warning("Validate all responses - this is for exploration of AI at the AAN meeting.")
    st.write("Author: David Liebovitz, MD")
    model = st.selectbox("Select a model:", ["gpt-3.5-turbo", "gpt-4-turbo-preview"])
    st.markdown(references_used)
    

if st.secrets["use_docker"] == "True" or check_password2():

    # Get user input
    user_role = st.radio("What is your role?", ["Patient", "Neurologist", "Other"], horizontal=True)
    if user_role == "Other":
        user_role = st.text_input("Enter your role:")

    query = st.text_input("Ask a question about Parkinson's Disease:")

    final_query = f'As a {user_role}, so please use appropriate terms, {query}'

    # If the user enters a query, get the answer
    if query:
        with st.spinner("Fomulating Answer..."):
            # Encode the query
            query_embedding = client.embeddings.create(input=query, model="text-embedding-3-large").data[0].embedding

            # Perform similarity search
            scores, indices = vectorstore.search(np.array([query_embedding]), k=10)
            
            # st.write(f'Scores: {scores}')
            # st.write(f'Indices: {indices}')

            # st.write(f'Embeddings Length: {len(embeddings)}')
            
            # st.write(f'One embedding: {embeddings[1][209]}')
            
            # Retrieve the most relevant documents
            relevant_documents = [embeddings[1][idx] for idx in indices[0]]
            
        

            # Construct the prompt for OpenAI
            prompt = f"What do the documents say regarding: {final_query}\n\nRelevant Documents to use for answering the question:\n"
            for doc in relevant_documents:
                prompt += f"- {doc}\n"
            prompt += "Answer:"
            
            analyze_rag_docs(final_query, model=model)

           
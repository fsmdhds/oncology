import streamlit as st
import faiss
import os
import numpy as np
import openai
from openai import OpenAI
import pickle
from prompts import rag_prompt

@st.cache_data
def analyze_rag_docs(prompt, model="gpt-3.5-turbo"):
    client = OpenAI(base_url="https://api.openai.com/v1", api_key=st.secrets['OPENAI_API_KEY'])
    # Generate the answer using OpenAI
    response = client.chat.completions.create(
        model=model,
        messages =[{"role": "system", "content": rag_prompt}, 
                   {"role": "user", "content": prompt}],
        max_tokens=1024,
        # n=1,
        stop=None,
        temperature=0,)
    return response.choices[0].message.content

client = OpenAI(base_url="https://api.openai.com/v1", api_key=st.secrets['OPENAI_API_KEY'])

st.set_page_config(page_title='Neurology Chats', layout='centered', page_icon="💬", initial_sidebar_state='auto')

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
        
        answer = analyze_rag_docs(final_query)

        st.write(answer)
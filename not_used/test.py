import streamlit as st
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.chains import VectorDBQA
from langchain.chains import RetrievalQA
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

llm = ChatOpenAI(openai_api_key=st.secrets['OPENAI_API_KEY'], model_name ="gpt-3.5-turbo", temperature=0)

# Load the FAISS database
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_API_KEY'],model="text-embedding-3-large")
vectorstore = FAISS.load_local("parkinson_disease.faiss", embeddings)

# # Set up the OpenAI LLM
# llm = OpenAI(temperature=0)

# Create the question-answering chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

# Streamlit app
st.title("Parkinson's Disease Question Answering")

# Get user input

user_role = st.selectbox("What is your role?", ["Patient", "Neurologist", "Other"])
if user_role == "Other":
    user_role = st.text_input("Enter your role:")

query = st.text_input("Ask a question about Parkinson's Disease:")

final_query = f'As a {user_role}, so please use appropriate terms, {query}'

# If the user enters a query, get the answer
if query:
    answer = qa_chain(query)
    st.write(answer["result"])
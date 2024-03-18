import streamlit as st
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.chains import VectorDBQA
from langchain.chains import RetrievalQA
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from prompts import rag_prompt, references_used
from langchain.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title='Cancer Chats', layout = 'centered', page_icon = "💬", initial_sidebar_state = 'auto')    

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


# Streamlit app
st.title("Cancer Care Question Answering")
with st.expander("ℹ️ About this App and Settings"):
    st.warning("Validate all responses - this is for exploration of AI at the AAN meeting.")
    st.write("Author: David Liebovitz, MD")
    
with st.sidebar:
    model = st.selectbox("Select a model:", ["gpt-3.5-turbo", "gpt-4-turbo-preview"])
    with st.expander("Cancer Care Sources"):
        st.markdown("List of cancer care references and resources.")
    

# Get user input

st.warning("""This app uses pre-processed content from reputable cancer care resources. The purpose here is to illustrate grounding answers
           in reliable sources through Retrieval Augmented Generation (RAG). Processed content is stored in a vector database and used when crafting a response. Sources are listed on the left. 
           The response will indicate if the reference material available fails to answer the question.""")

if st.secrets["use_docker"] == "True" or check_password2():
    st_callback = StreamlitCallbackHandler(st.container())
    with st.spinner("Preparing Databases..."):
        llm = ChatOpenAI(openai_api_key=st.secrets['OPENAI_API_KEY'], 
                         model_name =model, 
                         temperature=0.3,
                         streaming=True,
                )

        # Load the FAISS database
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_API_KEY'],model="text-embedding-3-large")
        vectorstore = FAISS.load_local("parkinson_disease.faiss", embeddings)

    # # Set up the OpenAI LLM
    # llm = OpenAI(temperature=0)

    # Create the question-answering chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff", callbacks=[st_callback],)

    user_role = st.radio("What is your role?", ["Patient", "Oncologist", "Other"], horizontal=True)
    if user_role == "Other":
        user_role = st.text_input("Enter your role:")

    query = st.text_input("Ask a question about cancer care:")

    final_query = f'As a {user_role}, please provide information using appropriate terms for cancer care, {query}.'

    # If the user enters a query, get the answer
    if query:
        with st.spinner("Fomulating Answer..."):

            st.write(qa_chain(final_query)["result"])
            # st.write(answer["result"])import streamlit as st
from langchain_community.chat_models import ChatOpenAI

# Set page config
st.set_page_config(page_title="Cancer Chat", page_icon="🧬")

# Title and introduction
st.title("Cancer Chat")
st.write("Welcome to the Cancer Chat! This tool is designed to provide information and support for oncology-related topics.")

# Initialize the chat model
chat_model = ChatOpenAI(model="gpt-4-turbo-preview")

# Chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input
user_input = st.text_input("Ask a question or start a conversation about cancer:")

# Handle the conversation
if user_input:
    response = chat_model.chat(user_input, history=st.session_state['history'])
    st.session_state['history'].append({"user": user_input, "assistant": response})

    # Display the conversation
    for exchange in st.session_state['history']:
        st.write(f"Q: {exchange['user']}")
        st.write(f"A: {exchange['assistant']}")

    # Clear the input box after the response
    st.session_state['user_input'] = ""

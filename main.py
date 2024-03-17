import base64
import json
import os
import random
import requests
import time
import getpass
import itertools
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import fitz  # PyMuPDF
import streamlit as st
import openai
from bs4 import BeautifulSoup
from fpdf import FPDF
from PIL import Image
from urllib.parse import urlparse, urlunparse
from openai import OpenAI

# Specific imports from modules where only specific functions/classes are used
from langchain.chains import QAGenerationChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS


from prompts import *
from using_docker import using_docker
from functions import *

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
        if not using_docker:
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

   
st.set_page_config(page_title='AAN feedback for AI Tools', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')    
title1, title2 = st.columns([1, 3])

with title1:

    medimate_robot = Image.open('static/medimate_robot.png')
    st.image(medimate_robot, use_column_width=True)
    
with title2:
        
    st.title("AAN Feedback on Tools for Neurology")

    with st.expander('About MediMate - Important Disclaimer'):
        st.write("Author: David Liebovitz, MD, Northwestern University")
        st.info(disclaimer)
        st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.3, 0.01)
        st.write("Last updated 3/17/24")
        st.write("ALPHA version 1.0")
    # st.info("With OpenAI announcement 11-6-2023, new model added: GPT-4-1106-preview. It's in beta and allows longer text inputs than GPT-4.")

if st.secrets["use_docker"] == "True" or check_password():
    st.warning("""Thank you for trying out our various use cases! Large language models (LLMs) hallucinate. This is particularly a concern in any healthcare context. Here, early methods
           to mitigate this are used including [CoVE](https://arxiv.org/abs/2309.11495) and grounding the final output with web content from reliable sites.
           Explore the links listed in the sidebar and copied below for easier phone use.""")
    col1, col2, col3 = st.columns(3)

    with col2:
        st.page_link("pages/🧠_Parkinson_Chat.py", label= "Parkinson's Chat", icon = "🧠")
        st.page_link("pages/🧐_Interview_Practice.py", label = "Interview Practice", icon = "🧐")
        st.page_link("pages/🗨️_Communication_and_DDx.py", label = "Communication and DDx", icon = "🗨️")
        st.page_link("pages/😞_Bias_Detection.py", label = "Bias Detection", icon = "😞")
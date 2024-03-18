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



def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


def process_model_name(model):
    prefix = "openai/"
    if model.startswith(prefix):
        model = model[len(prefix):]
    return model


def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    # st.write('yes the function is being used!')
    messages_blank = []
    st.session_state.expanded = False
    messages = update_messages(
        messages = messages_blank, 
        system_content=f'{prefix}; Sample question: {sample_question} Sample response: {sample_answer} Preceding conversation: {history_context}', 
        assistant_content='',
        user_content=my_ask,
        )
    # st.write(messages)
    model = st.session_state.model
    api_key = st.secrets["OPENROUTER_API_KEY"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    params = {
        "extra_headers": {
            "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app/",
            "X-Title": 'MediMate GPT and Med Ed',
        }
    }
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
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

def answer_using_prefix_openai(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    # st.write('yes the function is being used!')
    st.session_state.expanded = False
    messages_blank = []
    messages = update_messages(
        messages = messages_blank, 
        system_content=f'{prefix}; Sample question: {sample_question} Sample response: {sample_answer} Preceding conversation: {history_context}', 
        assistant_content='',
        user_content=my_ask,
        )
    # st.write(messages)
    model = process_model_name(st.session_state.model)
    # st.write('here is the model: ' + model)
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
    )
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
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


# def generate_chat_completion(
#     messages: List[Dict[str, str]],
#     model: str,
#     stream: bool,
#     api: str = "openai",
#     frequency_penalty: Optional[float] = 0,
#     logit_bias: Optional[Dict[int, int]] = None,
#     max_tokens: Optional[int] = None,
#     n: Optional[int] = 1,
#     presence_penalty: Optional[float] = 0,
#     response_format: Optional[Dict[str, str]] = None,
#     seed: Optional[int] = None,
#     stop: Optional[Union[str, List[str]]] = None,
#     temperature: Optional[float] = 1,
#     top_p: Optional[float] = 1,
#     tools: Optional[List[str]] = None,
#     tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
#     user: Optional[str] = None
# ) -> Any:
#     """
#     Generates a chat completion using the specified model and conversation context.

#     :param messages: A list of message dictionaries comprising the conversation so far.
#     :param model: ID of the model to use.
#     :param stream: If set to True, partial message deltas will be sent.
#     :param api: Determines which API to use, 'openai' or 'openrouter'.
#     :param frequency_penalty: Adjusts likelihood to repeat the same line verbatim.
#     :param logit_bias: Modifies likelihood of specified tokens appearing in the completion.
#     :param max_tokens: The maximum number of tokens to generate in the chat completion.
#     :param n: How many chat completion choices to generate for each input message.
#     :param presence_penalty: Increases likelihood to talk about new topics.
#     :param response_format: Specifies the format that the model must output.
#     :param seed: If specified, attempts to sample deterministically.
#     :param stop: Sequences where the API will stop generating further tokens.
#     :param temperature: Controls randomness of output.
#     :param top_p: Controls nucleus sampling.
#     :param tools: A list of tools the model may call.
#     :param tool_choice: Controls which function is called by the model.
#     :param user: A unique identifier representing the end-user.
#     :return: The generated chat completion.
#     """
#     # Initialize the params dictionary with extra_headers if using 'openrouter'
#     if api == "openrouter":

#         api_key = config["OPENROUTER_API_KEY"]
#         client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=api_key,
#         )
#         params = {
#             "extra_headers": {
#                 "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app/",
#                 "X-Title": 'MediMate GPT and Med Ed',
#             }
#         }
#     else:  # Default to 'openai'
#         model = process_model_name(model)
#         api_key = config["OPENAI_API_KEY"]
#         client = OpenAI(
#             base_url="https://api.openai.com/v1",
#             api_key=config["OPENAI_API_KEY"],
#         )
#         params = {}
        
#     # Construct the parameters as a dictionary
#     params = {
#         "model": model,
#         "messages": messages,
#         "stream": stream
#     }

#     # Add optional parameters if they are provided (i.e., not None)
#     if frequency_penalty is not None:
#         params["frequency_penalty"] = frequency_penalty
#     if logit_bias is not None:
#         params["logit_bias"] = logit_bias
#     if max_tokens is not None:
#         params["max_tokens"] = max_tokens
#     if n is not None:
#         params["n"] = n
#     if presence_penalty is not None:
#         params["presence_penalty"] = presence_penalty
#     if response_format is not None:
#         params["response_format"] = response_format
#     if seed is not None:
#         params["seed"] = seed
#     if stop is not None:
#         params["stop"] = stop
#     if temperature is not None:
#         params["temperature"] = temperature
#     if top_p is not None:
#         params["top_p"] = top_p
#     if tools is not None:
#         params["tools"] = tools
#     if tool_choice is not None:
#         params["tool_choice"] = tool_choice
#     if user is not None:
#         params["user"] = user
    


#     try:    
#         completion = client.chat.completions.create(**params)
#     except Exception as e:
#         st.write(e)
#         return None
#     if stream:
#         placeholder = st.empty()
#         full_response = ''
#         for chunk in completion:
#             if chunk.choices[0].delta.content is not None:
#                 full_response += chunk.choices[0].delta.content
#                 # full_response.append(chunk.choices[0].delta.content)
#                 placeholder.markdown(full_response)
#         placeholder.markdown(full_response)
#         return full_response
#     else:
#         return completion.choices[0].message.content

def update_messages(messages, system_content=None, assistant_content=None, user_content=None):
    """
    Updates a list of message dictionaries with new system, user, and assistant content.

    :param messages: List of message dictionaries with keys 'role' and 'content'.
    :param system_content: Optional new content for the system message.
    :param user_content: Optional new content for the user message.
    :param assistant_content: Optional new content for the assistant message.
    :return: Updated list of message dictionaries.
    """
    st.session_state.expanded = False
    # Update system message or add it if it does not exist
    system_message = next((message for message in messages if message['role'] == 'system'), None)
    if system_message is not None:
        if system_content is not None:
            system_message['content'] = system_content
    else:
        if system_content is not None:
            messages.append({"role": "system", "content": system_content})

    # Add assistant message if provided
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    # Add user message if provided
    if user_content is not None:
        messages.append({"role": "user", "content": user_content})

    return messages

@st.cache_data
def websearch_learn(web_query: str, deep, scrape_method, max) -> float:
    """
    Obtains real-time search results from across the internet. 
    Supports all Google Advanced Search operators such (e.g. inurl:, site:, intitle:, etc).
    
    :param web_query: A search query, including any Google Advanced Search operators
    :type web_query: string
    :return: A list of search results
    :rtype: json
    
    """
    st.session_state.expanded = False
    # st.info(f'Here is the websearch input: **{web_query}**')
    url = "https://real-time-web-search.p.rapidapi.com/search"
    querystring = {"q":web_query,"limit":max}
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response_data = response.json()
    # def display_search_results(json_data):
    #     data = json_data['data']
    #     for item in data:
    #         st.sidebar.markdown(f"### [{item['title']}]({item['url']})")
    #         st.sidebar.write(item['snippet'])
    #         st.sidebar.write("---")
    # st.info('Searching the web using: **{web_query}**')
    # display_search_results(response_data)
    # st.session_state.done = True
    # st.write(response_data)
    urls = []
    for item in response_data['data']:
        urls.append(item['url'])    
    if deep:
            # st.write(item['url'])
        if scrape_method != "Browserless":
            response_data = scrapeninja(urls, max)
        else:
            response_data = browserless(urls, max)
        # st.info("Web results reviewed.")
        return response_data, urls

    else:
        # st.info("Web snippets reviewed.")
        return response_data, urls

# def generate_medical_search(topic):
#     openai.api_base = "https://api.openai.com/v1/"
#     openai.api_key = st.secrets['OPENAI_API_KEY']
#     with st.spinner("Compressing messsages for summary..."):
#         completion = openai.ChatCompletion.create(
#             model = "gpt-3.5-turbo-1106",
#             temperature = 0.3,
#             messages = [
#                 {
#                     "role": "system",
#                     "content": nlm_query_template
#                 },
#                 {
#                     "role": "user",
#                     "content": topic
#                 }
#             ],
#             max_tokens = 300, 
#         )
#     return completion['choices'][0]['message']['content']

def reconcile_answers(context, question, old, new):
    st.session_state.expanded = False
    openai.api_base = "https://api.openai.com/v1/"
    openai.api_key = st.secrets['OPENAI_API_KEY']
    with st.spinner("Reconciling with new evidence..."):
        api_key = st.secrets["OPENAI_API_KEY"]
        from openai import OpenAI
        client = OpenAI(    
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )
        completion = client.chat.completions.create(
            model = "gpt-4-turbo-preview",
            temperature = 0.3,
            messages = [
                {
                    "role": "system",
                    "content": context
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": old
                },
                {
                    "role": "user",
                    "content": f'Revise your last response using this content retrieved from expert sources: {new} \n\n' + reconcile_prompt
                },
            ],
            max_tokens = 4096, 
        )
    return completion.choices[0].message.content

@st.cache_data
def browserless(url_list, max):
    # st.write(url_list)
    # if max > 5:
    #     max = 5
    st.session_state.expanded = False
    response_complete = []
    i = 0
    key = st.secrets["BROWSERLESS_API_KEY"]
    api_url = f'https://chrome.browserless.io/content?token={key}'
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }
    while i < max and i < len(url_list):
        url = url_list[i]
        url_parts = urlparse(url)
        # st.write("Scraping...")
        if 'uptodate.com' in url_parts.netloc:
            method = "POST"
            url_parts = url_parts._replace(path=url_parts.path + '/print')
            url = urlunparse(url_parts)
            # st.write(f' here is a {url}')
        payload =  {
            "url": url,
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        # response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            st.write(f'The site failed to release all content: {response.status_code}')
            # st.write(f'Response text: {response.text}')
            # st.write(f'Response headers: {response.headers}')
        try:
            # st.write(f'Response text: {response.text}')  # Print out the raw response text
            soup = BeautifulSoup(response.text, 'html.parser')
            clean_text = soup.get_text(separator=' ')
            # st.write(clean_text)
            # st.write("Scraped!")
            response_complete.append(clean_text)
        except json.JSONDecodeError:
            st.write("Error decoding JSON")
        i += 1
    full_response = ' '.join(response_complete)
    # limited_text = limit_tokens(full_response, 12000)
    # st.write(f'Here is the lmited text: {limited_text}')
    return full_response
    # st.write(full_response)    
    # Join all the scraped text into a single string
    # return full_response

@st.cache_data
def display_articles_with_streamlit(articles):
    st.session_state.expanded = False
    i = 1
    for article in articles:
        st.write(f"{i}. {article['title']}[{article['year']}]({article['link']})")
        i+=1
        # st.write("---")  # Adds a horizontal line for separation

def set_llm_chat(model, temperature):
    st.session_state.expanded = False
    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-4-turbo-preview":
        model = "gpt-4-turbo-preview"
    if model == "gpt-4-turbo-preview" or model == "gpt-3.5-turbo":
        return ChatOpenAI(model=model, openai_api_base = "https://api.openai.com/v1/", openai_api_key = st.secrets["OPENAI_API_KEY"], temperature=temperature)
    else:
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
          "X-Title": "GPT and Med Ed"}
        return ChatOpenAI(model = model, openai_api_base = "https://openrouter.ai/api/v1", openai_api_key = st.secrets["OPENROUTER_API_KEY"], temperature=temperature, max_tokens = 500, headers=headers)

def truncate_text(text, max_characters):
    if len(text) <= max_characters:
        return text
    else:
        truncated_text = text[:max_characters]
        return truncated_text

def semantic_search(search_term, max_results, year, publication_type):
    st.session_state.expanded = False
    # rsp = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/search?query={search_term}=url,title,abstract",
    rsp = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/search",
                       headers={'X-API-KEY': st.secrets["S2_API_KEY"]}, 
                       params={'query': search_term, 'fields': 'title,year,abstract,url', 'year': year, 'publicationTypes': publication_type, 'limit': max_results})
    rsp.raise_for_status()
    results = rsp.json()
    # abstracts = [f"{item['abstract']}" for item in results['data']]
    abstracts = '\n'.join([f"{item['abstract']} {item['year']}" for item in results['data']])
    # citations = [f"{item['title']}, {item['url']}" for item in results['data']]
    citations = '\n\n'.join([f"{item['title']} {item['year']} [link]({item['url']})" for item in results['data']])

    # st.write(f'here are your s2 results {abstracts}')
    return citations, abstracts

def clear_session_state_except_password_correct():
    # Make a copy of the session_state keys
    keys = list(st.session_state.keys())
    
    # Iterate over the keys
    for key in keys:
        # If the key is not 'password_correct', delete it from the session_state
        if key != 'password_correct':
            del st.session_state[key]




def fetch_api_key():
    api_key = None
    
    try:
        # Attempt to retrieve the API key as a secret
        api_key = st.secrets["OPENROUTER_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
    except KeyError:
        
        try:
            api_key = os.environ["OPENAI_API_KEY"]
            # If the API key is already set, don't prompt for it again
            return api_key
        except KeyError:        
            # If the secret is not found, prompt the user for their API key
            st.warning("Oh, dear friend of mine! It seems your API key has gone astray, hiding in the shadows. Pray, reveal it to me!")
            api_key = getpass.getpass("Please, whisper your API key into my ears: ")
            os.environ["OPENAI_API_KEY"] = api_key
            # Save the API key as a secret
            # st.secrets["my_api_key"] = api_key
            return api_key
    
    return api_key


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

def scrapeninja_old(url_list, max):
    # st.write(url_list)
    response_complete = []
    i = 0
    while max > i:
        i += 1
        url = url_list[i]
        st.write(f' here is a {url}')
        # st.write("Scraping...")
        payload = { "url": url }
        key = st.secrets["X-RapidAPI-Key"]
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": key,
            "X-RapidAPI-Host": "scrapeninja.p.rapidapi.com",
        }
        response = requests.post(url, json=payload, headers=headers)
        st.write(f'Status code: {response.status_code}')
        # st.write(f'Response text: {response.text}')
        # st.write(f'Response headers: {response.headers}')
        try:
            st.write(f'Response: {response}')
            response_data = response.json()
            st.write("Scraped!")
            return response_data
        except:
            json.JSONDecodeError
            st.write("Error decoding JSON")
        # response_data = response.json()
        # response_string = response_data['body']
        # return response_data

def limit_tokens(text, max_tokens=10000):
    tokens = text.split()  # split the text into tokens (words)
    limited_tokens = tokens[:max_tokens]  # keep the first max_tokens tokens
    limited_text = ' '.join(limited_tokens)  # join the tokens back into a string
    return limited_text

@st.cache_data
def scrapeninja(url_list, max):
    # st.write(url_list)
    st.session_state.expanded = False
    if max > 5:
        max = 5
    response_complete = []
    i = 0
    method = "POST"
    key = st.secrets["X-RapidAPI-Key"]
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": "scrapeninja.p.rapidapi.com",
    }
    while i < max and i < len(url_list):
        url = url_list[i]
        url_parts = urlparse(url)
        # st.write("Scraping...")
        if 'uptodate.com' in url_parts.netloc:
            method = "POST"
            url_parts = url_parts._replace(path=url_parts.path + '/print')
            url = urlunparse(url_parts)
            st.write(f' here is a {url}')
        payload =  {
            "url": url,
            "method": "POST",
            "retryNum": 1,
            "geo": "us",
            "js": True,
            "blockImages": False,
            "blockMedia": False,
            "steps": [],
            "extractor": "// define function which accepts body and cheerio as args\nfunction extract(input, cheerio) {\n    // return object with extracted values              \n    let $ = cheerio.load(input);\n  \n    let items = [];\n    $('.titleline').map(function() {\n          \tlet infoTr = $(this).closest('tr').next();\n      \t\tlet commentsLink = infoTr.find('a:contains(comments)');\n            items.push([\n                $(this).text(),\n              \t$('a', this).attr('href'),\n              \tinfoTr.find('.hnuser').text(),\n              \tparseInt(infoTr.find('.score').text()),\n              \tinfoTr.find('.age').attr('title'),\n              \tparseInt(commentsLink.text()),\n              \t'https://news.ycombinator.com/' + commentsLink.attr('href'),\n              \tnew Date()\n            ]);\n        });\n  \n  return { items };\n}"
        }
        
        response = requests.request(method, url, json=payload, headers=headers)
        # response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            st.write(f'The site failed to release all content: {response.status_code}')
            # st.write(f'Response text: {response.text}')
            # st.write(f'Response headers: {response.headers}')
        try:
            # st.write(f'Response text: {response.text}')  # Print out the raw response text
            soup = BeautifulSoup(response.text, 'html.parser')
            clean_text = soup.get_text(separator=' ')
            # st.write(clean_text)
            # st.write("Scraped!")
            response_complete.append(clean_text)
        except json.JSONDecodeError:
            st.write("Error decoding JSON")
        i += 1
    full_response = ' '.join(response_complete)
    limited_text = limit_tokens(full_response, 12000)
    # st.write(f'Here is the lmited text: {limited_text}')
    return limited_text
    # st.write(full_response)    
    # Join all the scraped text into a single string
    # return full_response

@st.cache_data
def websearch(web_query: str, deep, scrape_method, max) -> float:
    """
    Obtains real-time search results from across the internet. 
    Supports all Google Advanced Search operators such (e.g. inurl:, site:, intitle:, etc).
    
    :param web_query: A search query, including any Google Advanced Search operators
    :type web_query: string
    :return: A list of search results
    :rtype: json
    
    """
    st.session_state.expanded = False
    # st.info(f'Here is the websearch input: **{web_query}**')
    url = "https://real-time-web-search.p.rapidapi.com/search"
    querystring = {"q":web_query,"limit":"10"}
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response_data = response.json()
    # def display_search_results(json_data):
    #     data = json_data['data']
    #     for item in data:
    #         st.sidebar.markdown(f"### [{item['title']}]({item['url']})")
    #         st.sidebar.write(item['snippet'])
    #         st.sidebar.write("---")
    # st.info('Searching the web using: **{web_query}**')
    # display_search_results(response_data)
    # st.session_state.done = True
    # st.write(response_data)
    urls = []
    for item in response_data['data']:
        urls.append(item['url'])    
    if deep:
            # st.write(item['url'])
        if scrape_method != "Browserless":
            response_data = scrapeninja(urls, max)
        else:
            response_data = browserless(urls, max)
        # st.info("Web results reviewed.")
        return response_data, urls

    else:
        # st.info("Web snippets reviewed.")
        return response_data, urls

@st.cache_data
def pubmed_abstracts(search_terms, search_type="all"):
    st.session_state.expanded = False
    # URL encoding
    search_terms_encoded = requests.utils.quote(search_terms)

    # Define the publication type filter based on the search_type parameter
    if search_type == "all":
        publication_type_filter = ""
    elif search_type == "clinical trials":
        publication_type_filter = "+AND+Clinical+Trial[Publication+Type]"
    elif search_type == "reviews":
        publication_type_filter = "+AND+Review[Publication+Type]"
    else:
        raise ValueError("Invalid search_type parameter. Use 'all', 'clinical trials', or 'reviews'.")

    # Construct the search query with the publication type filter
    search_query = f"{search_terms_encoded}{publication_type_filter}"
    
    # Query to get the top 20 results
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax=20&api_key={st.secrets['pubmed_api_key']}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # Check if no results were returned, and if so, use a longer approach
        if 'count' in data['esearchresult'] and int(data['esearchresult']['count']) == 0:
            return st.write("No results found. Try a different search or try again after re-loading the page.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching search results: {e}")
        return []

    ids = data['esearchresult']['idlist']
    articles = []

    for id in ids:
        details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id}&retmode=json&api_key={st.secrets['pubmed_api_key']}"
        try:
            details_response = requests.get(details_url)
            details_response.raise_for_status()  # Raise an exception for HTTP errors
            details = details_response.json()
            if 'result' in details and str(id) in details['result']:
                article = details['result'][str(id)]
                year = article['pubdate'].split(" ")[0]
                if year.isdigit():
                    articles.append({
                        'title': article['title'],
                        'year': year,
                        'link': f"https://pubmed.ncbi.nlm.nih.gov/{id}"
                    })
            else:
                st.warning(f"Details not available for ID {id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching details for ID {id}: {e}")
        time.sleep(1)  # Introduce a delay to avoid hitting rate limits only if there's an error

    # Second query: Get the abstract texts for the top 10 results
    abstracts = []
    for id in ids:
        abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id}&retmode=text&rettype=abstract&api_key={st.secrets['pubmed_api_key']}"
        try:
            abstract_response = requests.get(abstract_url)
            abstract_response.raise_for_status()  # Raise an exception for HTTP errors
            abstract_text = abstract_response.text
            if "API rate limit exceeded" not in abstract_text:
                abstracts.append(abstract_text)
            else:
                st.warning(f"Rate limit exceeded when fetching abstract for ID {id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching abstract for ID {id}: {e}")
        time.sleep(1)  # Introduce a delay to avoid hitting rate limits only if there's an error

    return articles, "\n".join(abstracts)

def answer_using_prefix_openai_old(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    openai.api_base = "https://api.openai.com/v1/"
    openai.api_key = st.secrets['OPENAI_API_KEY']
    if st.session_state.model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if st.session_state.model == "openai/gpt-4-turbo-preview":
        model = "gpt-4-turbo-preview"
    if history_context == None:
        history_context = ""
    stream = True
    if st.session_state.model == "anthropic/claude-instant-v1":
        stream = False
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    with st.spinner("Generating response..."):
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = 750,
        stream = stream,   
        )
        
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in completion:        
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content


def answer_using_prefix_old(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    openai.api_key = os.environ['OPENAI_API_KEY']
    if history_context == None:
        history_context = ""
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    with st.spinner("Generating response..."):
    
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = st.session_state.model,
        route = "fallback",
        messages = messages,
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
            "X-Title": "GPT and Med Ed"},
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
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content

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



def create_retriever(texts):  
    st.session_state.expanded = False
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",
                                  openai_api_base = "https://api.openai.com/v1/",
                                  openai_api_key = st.secrets['OPENAI_API_KEY']
                                  )
    try:
        vectorstore = FAISS.from_texts(texts, embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(k=5)

    return retriever

@st.cache_data
def split_texts(text, chunk_size, overlap, split_method):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        # st.error("Failed to split document")
        st.stop()

    return splits

@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list
    openai.api_key = os.environ['OPENAI_API_KEY']

    # st.info("`Generating sample questions and answers...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:",i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full




def prepare_rag(text):
    st.session_state.expanded = False
    splits = split_texts(text, chunk_size=1000, overlap=100, split_method="recursive")
    st.session_state.retriever = create_retriever(splits)
    llm = set_llm_chat(model="gpt-4-turbo-preview", temperature=st.session_state.temp)
    rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.retriever)
    return rag
    
    

def fn_qa_run(_qa, user_question):
    response = _qa.run(user_question)
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in response:        
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event[0]
        answer += event_text
        full_answer += event_text
        time.sleep(delay_time)
    
    return full_answer

if 'prelim_response' not in st.session_state:
    st.session_state.prelim_response = ""
    
if 'evidence_response' not in st.session_state:
    st.session_state.evidence_response = ""

if 'patient_message_history' not in st.session_state:
    st.session_state.patient_message_history = []

if 'sample_patient_message' not in st.session_state:
    st.session_state.sample_patient_message = ""

if 'teaching_thread' not in st.session_state:
    st.session_state.teaching_thread = []

if "pdf_retriever" not in st.session_state:
    st.session_state.pdf_retriever = []

if 'dc_history' not in st.session_state:
    st.session_state.dc_history = []

if 'annotate_history' not in st.session_state:
    st.session_state.annotate_history = []

if 'history' not in st.session_state:
    st.session_state.history = []

if 'output_history' not in st.session_state:
    st.session_state.output_history = []
            
if 'sample_report' not in st.session_state:
    st.session_state.sample_report = ""
            
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
    
if 'model' not in st.session_state:
    st.session_state.model = "openai/gpt-3.5-turbo"
    
if 'temp' not in st.session_state:
    st.session_state.temp = 0.3
    
if "pdf_user_question" not in st.session_state:
    st.session_state["pdf_user_question"] = []
if "pdf_user_answer" not in st.session_state:
    st.session_state["pdf_user_answer"] = []

if "last_uploaded_files" not in st.session_state:
    st.session_state["last_uploaded_files"] = []
    
if "abstract_questions" not in st.session_state:
    st.session_state["abstract_questions"] = []
    
if "abstract_answers" not in st.session_state:
    st.session_state["abstract_answers"] = []

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = []

if "abstracts" not in st.session_state:
    st.session_state["abstracts"] = ""

if "s2_abstracts" not in st.session_state:
    st.session_state["s2_abstracts"] = ""
    
if "your_question" not in st.session_state:
    st.session_state["your_question"] = ""
    
if "texts" not in st.session_state:
    st.session_state["texts"] = ""
    
if "citations" not in st.session_state:
    st.session_state["citations"] = ""
    
if "s2_citations" not in st.session_state:
    st.session_state["s2_citations"] = ""
    
if "search_terms" not in st.session_state:
    st.session_state["search_terms"] = ""   
    
if "pt_ed_output_text" not in st.session_state:
    st.session_state["pt_ed_output_text"] = ""
    
if "alt_dx_output_text" not in st.session_state:
    st.session_state["alt_dx_output_text"] = ""
    
if "ddx_output_text" not in st.session_state:
    st.session_state["ddx_output_text"] = ""
    
if "skim_output_text" not in st.session_state:
    st.session_state["skim_output_text"] = ""
    
if "expanded" not in st.session_state:
    st.session_state["expanded"] = True
   
st.set_page_config(page_title='Communication and DDX', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')    
title1, title2 = st.columns([1, 3])

    
with title2:
        
    st.title("Communication and DDx")


    # st.info("With OpenAI announcement 11-6-2023, new model added: GPT-4-1106-preview. It's in beta and allows longer text inputs than GPT-4.")

if st.secrets["use_docker"] == "True" or check_password():
    
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key = st.secrets["OPENROUTER_API_KEY"]


    os.environ['OPENAI_API_KEY'] = fetch_api_key()


    with st.sidebar.expander("Select a GPT Language Model", expanded=True):
        st.session_state.model = st.selectbox("Model Options", ("openai/gpt-3.5-turbo",  "openai/gpt-4-turbo-preview",  "anthropic/claude-3-sonnet:beta", "anthropic/claude-instant-v1", "google/palm-2-chat-bison", "meta-llama/codellama-34b-instruct", "meta-llama/llama-2-70b-chat", "gryphe/mythomax-L2-13b", "nousresearch/nous-hermes-llama2-13b"), index=0)
        if st.session_state.model == "google/palm-2-chat-bison":
            st.warning("The Google model doesn't stream the output, but it's fast. (Will add Med-Palm2 when it's available.)")
            st.markdown("[Information on Google's Palm 2 Model](https://ai.google/discover/palm2/)")
        # if st.session_state.model == "openai/gpt-4-turbo-preview":
        #     # st.warning("GPT-4 preview JUST RELEASED 11-6-2023 has a huge context window but is in beta.")
        #     st.markdown("[Information on OpenAI's GPT-4](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)")
        
        if st.session_state.model == "openai/gpt-4-turbo-preview":
            st.warning("GPT-4 is much more expensive and sometimes, not always, better than others.")
            st.markdown("[Information on OpenAI's GPT-4](https://platform.openai.com/docs/models/gpt-4)")
        if st.session_state.model == "anthropic/claude-instant-v1":
            st.markdown("[Information on Anthropic's Claude-Instant](https://www.anthropic.com/index/releasing-claude-instant-1-2)")
        if st.session_state.model == "meta-llama/llama-2-70b-chat":
            st.markdown("[Information on Meta's Llama2](https://ai.meta.com/llama/)")
        # if st.session_state.model == "openai/gpt-3.5-turbo":
        #     st.markdown("[Information on OpenAI's GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)")
        if st.session_state.model == "openai/gpt-3.5-turbo":
            st.markdown("[Information on OpenAI's GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)")
        if st.session_state.model == "gryphe/mythomax-L2-13b":
            st.markdown("[Information on Gryphe's Mythomax](https://huggingface.co/Gryphe/MythoMax-L2-13b)")
        if st.session_state.model == "meta-llama/codellama-34b-instruct":
            st.markdown("[Information on Meta's CodeLlama](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)")
    tab1, tab2, tab3= st.tabs(["Draft Communication", "Patient Education", "Differential Diagnosis", ])
   
    
                    
                
    with tab1:
        # st.subheader("Patient Communication")
        col1, col2 = st.columns(2)
        with col2:
            health_literacy_level = st.radio("Output optimized for:", ("General Public Medical Knowledge", "Advanced Medical Knowledge"))
   

        with col1:
            task = st.radio("What do you want to do?", ("Generate discharge instructions", "Annotate a patient result", "Respond to a patient message"))

        if task == "Respond to a patient message":
            patient_message_type = st.sidebar.radio("Select a message type:", ("Make your own and go to Step 2!", "Patient message about symptoms", "Patient message about medications", "Patient message about medical problems", "Patient message about lifestyle advice"))
            patient_message_prompt = f'Generate a message sent by a patient with {health_literacy_level} asking her physician for advice. The patient message should include the (random) patient name and is a {patient_message_type}. '
            if patient_message_type != "Make your own and go to Step 2!":
                with st.sidebar:
                    # submitted_result = ""
                    if st.sidebar.button("Step 1: Generate a Patient Message"):
                        with col1:
                            if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-4-turbo-preview":
                                st.session_state.sample_patient_message = answer_using_prefix_openai(
                                    sim_patient_context, 
                                    prompt_for_generating_patient_question, 
                                    sample_patient_question, 
                                    patient_message_prompt, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            else:

                                st.session_state.sample_patient_message = answer_using_prefix(
                                    sim_patient_context, 
                                    prompt_for_generating_patient_question, 
                                    sample_patient_question, 
                                    patient_message_prompt, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            if st.session_state.model == "google/palm-2-chat-bison":
                                st.write("Patient Message:", st.session_state.sample_patient_message)
            else:
                with st.sidebar:
                    with col1:
                        st.session_state.sample_patient_message = st.text_area("Enter an example of a message from a patient.", placeholder="e.g., I have a headache and I am worried about a brain tumor.", label_visibility='visible',)
                        
            if st.button("Step 2: Generate Response for Patient Message"):
                try:
                    with col2:
                        if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-4-turbo-preview":
                            pt_message_response = answer_using_prefix_openai(
                                physician_response_context, 
                                sample_patient_question, 
                                sample_response_for_patient,
                                st.session_state.sample_patient_message, 
                                st.session_state.temp, 
                                history_context="",
                                )   
                        else:
                            pt_message_response = answer_using_prefix(
                                physician_response_context, 
                                sample_patient_question, 
                                sample_response_for_patient,
                                st.session_state.sample_patient_message, 
                                st.session_state.temp, 
                                history_context="",
                                )                    
                        if st.session_state.model == "google/palm-2-chat-bison":
                            st.write("Draft Response:", pt_message_response)

                        st.session_state.patient_message_history.append((pt_message_response))
                    with col1:
                        if task == "Respond to a patient message":
                            st.write("Patient Message:", st.session_state.sample_patient_message)
                except:
                    with col2:
                        st.write("API busy. Try again - better error handling coming. :) ")
                        st.stop()
        
        if task == "Generate discharge instructions":
            answer = ''
            start_time = time.time()
            reason_for_hospital_stay = st.text_area("Please enter the reason for the hospital stay.", placeholder="e.g., SCT for lymphoma", label_visibility='visible',)
            surg_procedure = st.text_area("Please enter any procedure(s) performed and any special concerns.", placeholder="e.g., lumbar puncture", label_visibility='visible',)
            other_concerns = st.text_area("Please enter any other concerns.", placeholder="e.g., chronic hand tremor", label_visibility='visible',)
            dc_meds = st.text_area("Please enter the discharge medications.", placeholder="e.g., lisinopril 10 mg daily for HTN", label_visibility='visible',)
            dc_instructions_needs = f'Generate discharge instructions for a patient as if it is authored by a physician for her patient with {health_literacy_level} discharged following {reason_for_hospital_stay} with this {surg_procedure}, {other_concerns} on {dc_meds}'
            if st.button("Generate Discharge Instructions"):
                try:
                    if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-4-turbo-preview":
                        dc_text = answer_using_prefix_openai(
                            dc_instructions_prompt, 
                            procedure_example, 
                            dc_instructions_example, 
                            dc_instructions_needs, 
                            st.session_state.temp, 
                            history_context="",
                            )
                    
                    else:
                        dc_text = answer_using_prefix(
                            dc_instructions_prompt, 
                            procedure_example, 
                            dc_instructions_example, 
                            dc_instructions_needs, 
                            st.session_state.temp, 
                            history_context="",
                            )
                    if st.session_state.model == "google/palm-2-chat-bison":
                        st.write("DC Instructions:", dc_text)
                    st.session_state.dc_history.append((dc_text))  
                except:
                    st.write("Error - please try again")
                      

        
            dc_download_str = []
                
                # ENTITY_MEMORY_CONVERSATION_TEMPLATE
                # Display the conversation history using an expander, and allow the user to download it
            with st.expander("View or Download Instructions", expanded=st.session_state.expanded):
                for i in range(len(st.session_state['dc_history'])-1, -1, -1):
                    st.info(st.session_state["dc_history"][i],icon="🧐")
                    st.success(st.session_state["dc_history"][i], icon="🤖")
                    dc_download_str.append(st.session_state["dc_history"][i])
                    
                dc_download_str = [disclaimer] + dc_download_str 
                
                
                dc_download_str = '\n'.join(dc_download_str)
                if dc_download_str:
                    st.download_button('Download',dc_download_str, key = "DC_Thread")   
                    export_as_pdf = st.button("Create PDF version", key = "dc_download")
                    if export_as_pdf:
                        st.session_state.expanded = True
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 5, dc_download_str)
                        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "dc_instructions")
                        st.sidebar.info("Here is your PDF file to download!")
                        st.sidebar.markdown(html, unsafe_allow_html=True)
                        # pdf.output("final_responses.pdf")     
                    

        if task == "Annotate a patient result":
            sample_report1 = st.sidebar.radio("Try a sample report:", ("Text box for your own content", "Sample 1 (Brain MR)", "Sample 2 (PET CT)", "Generate a sample report"))
            if sample_report1 == "Sample 1 (Brain MR)":
                st.session_state.sample_report = report1
                with col1:
                    st.write(report1)
            elif sample_report1 == "Sample 2 (PET CT)":
                st.session_state.sample_report = report2
                with col1:
                    st.write(report2)
            elif sample_report1 == "Text box for your own content":           
                with col1:                
                    st.session_state.sample_report = st.text_area("Paste your result content here without PHI.", height=600)
            
            elif sample_report1 == "Generate a sample report":
                with st.sidebar:
                    type_of_report = st.text_area("Enter the patient report type to generate", placeholder= 'e.g., brain CT with microvascular changes', height=100)
                    submitted_result = ""
                    if st.sidebar.button("Generate Sample Report"):
                        with col1:
                            if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-4-turbo-preview":
                                st.session_state.sample_report = answer_using_prefix_openai(
                                    report_prompt, 
                                    user_report_request, 
                                    generated_report_example, 
                                    type_of_report, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            else:
                                st.session_state.sample_report = answer_using_prefix(
                                    report_prompt, 
                                    user_report_request, 
                                    generated_report_example, 
                                    type_of_report, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            if st.session_state.model == "google/palm-2-chat-bison":
                                st.write("Answer:", st.session_state.sample_report)
                        
            
            
            report_prompt = f'Generate a brief reassuring summary as if it is authored by a physician for her patient with {health_literacy_level} with this {st.session_state.sample_report}. When appropriate emphasize that the findings are not urgent and you are happy to answer any questions at the next visit. '

            
            if st.button("Generate Annotation"):
                try:
                    with col2:
                        if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-4-turbo-preview":
                            annotate_text = answer_using_prefix_openai(
                                annotate_prompt, 
                                report1, 
                                annotation_example,
                                report_prompt, 
                                st.session_state.temp, 
                                history_context="",
                                )   
                            
                        else:
                            annotate_text = answer_using_prefix(
                                annotate_prompt, 
                                report1, 
                                annotation_example,
                                report_prompt, 
                                st.session_state.temp, 
                                history_context="",
                                )   
                        
                        if st.session_state.model == "google/palm-2-chat-bison":
                            st.write("Answer:", annotate_text)                 

                        st.session_state.annotate_history.append((annotate_text))
                    with col1:
                        if sample_report1 == "Generate a sample report":
                            st.write("Your Report:", st.session_state.sample_report)
                except:
                    with col2:
                        st.write("API busy. Try again - better error handling coming. :) ")
                        st.stop()
            
                    
        
            annotate_download_str = []
                
                # ENTITY_MEMORY_CONVERSATION_TEMPLATE
                # Display the conversation history using an expander, and allow the user to download it
            with st.expander("View or Download Annotations", expanded=st.session_state.expanded):
                for i in range(len(st.session_state['annotate_history'])-1, -1, -1):
                    st.info(st.session_state["annotate_history"][i],icon="🧐")
                    st.success(st.session_state["annotate_history"][i], icon="🤖")
                    annotate_download_str.append(st.session_state["annotate_history"][i])
                    
                annotate_download_str = [disclaimer] + annotate_download_str 
                
                
                annotate_download_str = '\n'.join(annotate_download_str)
                if annotate_download_str:
                    st.download_button('Download',annotate_download_str, key = "Annotate_Thread")
                    export_as_pdf = st.button("Create PDF version", key = "annotate_pdf")
                    if export_as_pdf:
                        st.session_state.expanded = True
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 5, annotate_download_str)
                        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "annotations")
                        st.sidebar.info("Here is your PDF file to download!")
                        st.sidebar.markdown(html, unsafe_allow_html=True)        
            
            
    with tab3:
        
        # st.subheader("Differential Diagnosis Tools")
        
        # st.info("Avoid premature closure and consider alternative diagnoses")
        
        ddx_strategy = st.radio("Choose an approach for a differential diagnosis!", options=["Find Alternative Diagnoses to Consider","Provide Clinical Data"], index=0, key="ddx strategy")


        if ddx_strategy == "Provide Clinical Data":    
            # st.title("Differential Diagnosis Generator")
            st.write("Add as many details as possible to improve the response. The prompts do not request any unique details; however, *modify values and do not include dates to ensure privacy.")

            age = st.slider("Age", 0, 120, 50)
            sex_at_birth = st.radio("Sex at Birth", options=["Female", "Male", "Other"], horizontal=True)
            presenting_symptoms = st.text_input("Presenting Symptoms")
            duration_of_symptoms = st.text_input("Duration of Symptoms")
            past_medical_history = st.text_input("Past Medical History")
            current_medications = st.text_input("Current Medications")
            relevant_social_history = st.text_input("Relevant Social History")
            physical_examination_findings = st.text_input("Physical Examination Findings")
            lab_or_imaging_results = st.text_input("Any relevant Laboratory or Imaging results")
            ddx_prompt = f"""
            Patient Information:
            - Age: {age}
            - Sex: {sex_at_birth}
            - Presenting Symptoms: {presenting_symptoms}
            - Duration of Symptoms: {duration_of_symptoms}
            - Past Medical History: {past_medical_history}
            - Current Medications: {current_medications}
            - Relevant Social History: {relevant_social_history}
            - Physical Examination Findings: {physical_examination_findings}
            - Any relevant Laboratory or Imaging results: {lab_or_imaging_results}
            """
            
            
            if st.button("Generate Differential Diagnosis"):
                # Your differential diagnosis generation code goes here
                if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-4-turbo-preview":
                    ddx_output_text = answer_using_prefix_openai(ddx_prefix, ddx_sample_question, ddx_sample_answer, ddx_prompt, temperature=0.3, history_context='')
                    st.session_state.ddx_output_text = ddx_output_text
                else:
                    ddx_output_text = answer_using_prefix(ddx_prefix, ddx_sample_question, ddx_sample_answer, ddx_prompt, temperature=0.3, history_context='')
                    st.session_state.ddx_output_text = ddx_output_text
                # st.write("Differential Diagnosis will appear here...")
                
                # ddx_download_str = []
            if st.session_state.ddx_output_text != "":    
                with st.expander("Differential Diagnosis Draft", expanded=st.session_state.expanded):
                    st.info(f'Topic: {ddx_prompt}',icon="🧐")
                    st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {st.session_state.ddx_output_text}', icon="🤖")                         
                    ddx_download_str = f"{disclaimer}\n\nDifferential Diagnoses for {ddx_prompt}:\n\n{st.session_state.ddx_output_text}"
                    if ddx_download_str:
                        st.download_button('Download', ddx_download_str, key = 'ddx_questions_1')
                        export_as_pdf = st.button("Create PDF version", key = "ddx_pdf")
                        if export_as_pdf:
                            st.session_state.expanded = True
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.multi_cell(0, 5, ddx_download_str)
                            html = create_download_link(pdf.output(dest="S").encode("latin-1"), "ald_ddx")
                            st.sidebar.info("Here is your PDF file to download!")
                            st.sidebar.markdown(html, unsafe_allow_html=True)  
                        
                        
        # Alternative Diagnosis Generator
        if ddx_strategy == "Find Alternative Diagnoses to Consider":
            # st.subheader("Alternative Diagnosis Generator")
            
            alt_dx_prompt = st.text_input("Enter your presumed diagnosis.")

            if st.button("Generate Alternative Diagnoses"):
                if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-4-turbo-preview":
                    alt_dx_output_text = answer_using_prefix_openai(alt_dx_prefix, alt_dx_sample_question, alt_dx_sample_answer, alt_dx_prompt, temperature=0.0, history_context='')
                    st.session_state.alt_dx_output_text = alt_dx_output_text
                else:
                    alt_dx_output_text = answer_using_prefix(alt_dx_prefix, alt_dx_sample_question, alt_dx_sample_answer, alt_dx_prompt, temperature=0.0, history_context='')
                    st.session_state.alt_dx_output_text = alt_dx_output_text
                if st.session_state.model == "google/palm-2-chat-bison":
                    st.write("Alternative Diagnoses:", alt_dx_output_text)
                    st.session_state.alt_dx_output_text = alt_dx_output_text
                # alt_dx_download_str = []
            if st.session_state.alt_dx_output_text != "":
                with st.expander("Alternative Diagnoses Draft", expanded=st.session_state.expanded):
                    st.info(f'Topic: {alt_dx_prompt}',icon="🧐")
                    st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {st.session_state.alt_dx_output_text}', icon="🤖")
                    alt_dx_download_str = f"{disclaimer}\n\nAlternative Diagnoses for {alt_dx_prompt}:\n\n{st.session_state.alt_dx_output_text}"
                    if alt_dx_download_str:
                        st.download_button('Download', alt_dx_download_str, key = 'alt_dx_output')
                        export_as_pdf = st.button("Create PDF version", key = "alt_dx_output_pdf")
                        if export_as_pdf:
                            st.session_state.expanded = True
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.multi_cell(0, 5, alt_dx_download_str)
                            html = create_download_link(pdf.output(dest="S").encode("latin-1"), "ald_ddx")
                            st.sidebar.info("Here is your PDF file to download!")
                            st.sidebar.markdown(html, unsafe_allow_html=True)  

    with tab2:

        pt_ed_health_literacy = st.radio("Pick a desired health literacy level:", ("General Public Medical Knowledge", "Advanced Medical Knowledge"))
        
        
        
        if pt_ed_health_literacy == "General Public Medical Knowledge":
            pt_ed_content_sample = pt_ed_basic_example

        if pt_ed_health_literacy == "Intermediate":
            pt_ed_content_sample = pt_ed_intermediate_example
        if pt_ed_health_literacy == "Advanced Medical Knowledge":
            pt_ed_content_sample = pt_ed_advanced_example
        
        sample_topic = "dietary guidance for a patient with migraines, kidney disease, hypertension, obesity, and CAD"
        patient_ed_temp = st.session_state.temp
        my_ask_for_pt_ed = st.text_area("Generate patient education materials:", placeholder="e.g., lifestyle guidance for cancer prevention", label_visibility='visible', height=100)
        my_ask_for_pt_ed = "Generate patient education materials for: " + my_ask_for_pt_ed.replace("\n", " ")
        my_ask_for_pt_ed = my_ask_for_pt_ed + "with health literacy level: " + pt_ed_health_literacy
        if st.button("Click to Generate **Draft** Custom Patient Education Materials"):
            st.info("Review all content carefully before considering any use!")
            if st.session_state.model == "openai/gpt-3.5-turbo"  or st.session_state.model == "openai/gpt-4-turbo-preview":
                pt_ed_output_text = answer_using_prefix_openai(pt_ed_system_content, sample_topic, pt_ed_content_sample, my_ask_for_pt_ed, patient_ed_temp, history_context="")
                st.session_state.pt_ed_output_text = pt_ed_output_text
                
            else:
                pt_ed_output_text = answer_using_prefix(pt_ed_system_content, sample_topic, pt_ed_content_sample, my_ask_for_pt_ed, patient_ed_temp, history_context="")
                st.session_state.pt_ed_output_text = pt_ed_output_text
            if st.session_state.model == "google/palm-2-chat-bison":
                st.write("Patient Education:", pt_ed_output_text)
                st.session_state.pt_ed_output_text = pt_ed_output_text

            
            # pt_ed_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
        if st.session_state.pt_ed_output_text != "":
            with st.expander("Patient Education Draft", expanded=st.session_state.expanded):
                st.info(f'Topic: {my_ask_for_pt_ed}',icon="🧐")
                st.success(f'Draft Patient Education Materials: **REVIEW CAREFULLY FOR ERRORS** \n\n {st.session_state.pt_ed_output_text}', icon="🤖")      
                pt_ed_download_str = f"{disclaimer}\n\nDraft Patient Education Materials: {my_ask_for_pt_ed}:\n\n{st.session_state.pt_ed_output_text}"
                if pt_ed_download_str:
                        st.download_button('Download', pt_ed_download_str, key = 'pt_ed_questions')
                        export_as_pdf = st.button("Create PDF version", key = "pt_ed_pdf")
                        if export_as_pdf:
                            st.session_state.expanded = True
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.multi_cell(0, 5, pt_ed_download_str)
                            html = create_download_link(pdf.output(dest="S").encode("latin-1"), "ald_ddx")
                            st.sidebar.info("Here is your PDF file to download!")
                            st.sidebar.markdown(html, unsafe_allow_html=True)  
                        
          
    
    
                    

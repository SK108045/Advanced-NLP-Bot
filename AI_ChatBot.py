import ollama
import streamlit as st
from dotenv import load_dotenv
import requests
import os
import base64
from functools import lru_cache

st.set_page_config(page_title="AI ChatBot", page_icon="🤖")

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #4527A0;
        padding: 1rem;
        border-bottom: 2px solid #7B1FA2;
        margin-bottom: 2rem;
        font-size: 2.5rem;
    }
    </style>
""", unsafe_allow_html=True)

MAX_HISTORY = 5
load_dotenv()
api_key = os.getenv('DG_API_KEY')

@st.cache_data(show_spinner=False)
def cached_text_to_speech(text):
    try:
        url = "https://api.deepgram.com/v1/speak"
        params = {
            "model": "aura-stella-en",
            "encoding": "linear16",
            "container": "wav",
            "sample_rate": "48000"
        }
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
        data = {"text": text}
        
        response = requests.post(url, params=params, headers=headers, json=data)
        return response.content
    except (requests.exceptions.ConnectionError, requests.exceptions.RequestException):
        return None

def get_audio_player(audio_content, message_id, auto_play=False):
    if audio_content is None:
        return ""
    
    b64 = base64.b64encode(audio_content).decode()
    html = f"""
    <div style="display: inline-block;">
        <audio id="audio_{message_id}" style="display:none">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        <span onclick="document.getElementById('audio_{message_id}').play()"
              style="cursor:pointer; color: white;">
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <i class="material-icons">play_circle_filled</i>
        </span>
        {'''<script>document.getElementById("audio_{0}").play();</script>'''.format(message_id) if auto_play else ''}
    </div>
    """
    st.components.v1.html(html, height=30)
    return ""

st.markdown("<h1 class='main-header'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)

# Initialize states
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "model" not in st.session_state:
    st.session_state["model"] = "llama3.2:1b"

def model_res_generator():
    response_text = ""
    recent_messages = st.session_state["messages"][-MAX_HISTORY:]
    
    shakespeare_prompt = {"role": "system", "content": "You are a helpful assistant."}
    
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=[shakespeare_prompt] + recent_messages,
        stream=True,
    )
    
    for chunk in stream:
        chunk_text = chunk["message"]["content"]
        response_text += chunk_text
        yield chunk_text
    
    return response_text

for idx, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            audio_content = cached_text_to_speech(message["content"])
            st.markdown(get_audio_player(audio_content, f"hist_{idx}", auto_play=False), unsafe_allow_html=True)

if prompt := st.chat_input("Enter your question?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})
        
        audio_content = cached_text_to_speech(message)
        st.markdown(get_audio_player(audio_content, f"new_{len(st.session_state['messages'])}", auto_play=True), unsafe_allow_html=True)

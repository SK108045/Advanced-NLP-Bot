import streamlit as st
import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
import PyPDF2
import tempfile

st.set_page_config(page_title="Document Chat", page_icon="📄") 


def pdf_to_text(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as temp_txt:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
          
            text = text.replace('\n', ' ')  
            text = ' '.join(text.split())   
            text = text.replace('|', ' ')    
            text = text.replace('\t', ' ')  
            text = text.replace('  ', ' ')  
            temp_txt.write(text + '\n\n')   
        return temp_txt.name


def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def stream_response(messages, context):
    response_text = ""
    full_messages = [
        {
            "role": "system",
            "content": f"""You are a document analysis assistant. You have been provided with a specific document to analyze.
            IMPORTANT: Always refer to and use the following document context to answer questions:


            DOCUMENT CONTEXT:
            {context}


            """
        }
    ] + messages[-5:]

    #Debugging for printing context
    print("Context being passed:", context)

    with st.status('Thinking...', expanded=True) as status:
        stream = ollama.chat(
            model="llama3.2:1b",
            messages=full_messages,
            stream=True,
        )
        
        status.update(label="Generating response...", state="running")
        
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            response_text += chunk_text
            yield chunk_text
            status.empty()
    
    return response_text

st.title("Document Assistant")

# Initialize session states
if "doc_data" not in st.session_state:
    st.session_state.doc_data = None
if "doc_messages" not in st.session_state:
    st.session_state.doc_messages = []

uploaded_file = st.file_uploader("Upload your document", type=['txt', 'pdf'])

if uploaded_file:
    # Process document only once when first uploaded
    if st.session_state.doc_data is None:
        with st.spinner('Processing document...'):
            if uploaded_file.name.endswith('.pdf'):
                # Convert PDF to text file first
                temp_txt_path = pdf_to_text(uploaded_file)
                paragraphs = parse_file(temp_txt_path)
                # Clean up temporary file
                os.unlink(temp_txt_path)
            else:
                paragraphs = parse_file(uploaded_file.name)
                
            embeddings = get_embeddings(uploaded_file.name, "nomic-embed-text", paragraphs)
            st.session_state.doc_data = {
                "paragraphs": paragraphs,
                "embeddings": embeddings
            }

    # Display chat history
    for message in st.session_state.doc_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create Q&A interface
    if question := st.chat_input("Ask a question about your document:"):
        st.session_state.doc_messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.doc_data:
            prompt_embedding = ollama.embeddings(
                model="nomic-embed-text",
                prompt=question
            )["embedding"]
            
            most_similar_chunks = find_most_similar(
                prompt_embedding,
                st.session_state.doc_data["embeddings"]
            )[:5]
            
            context = "\n".join(st.session_state.doc_data["paragraphs"][item[1]] for item in most_similar_chunks)
            
            conversation_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.doc_messages
            ]

            with st.chat_message("assistant"):
                response = st.write_stream(stream_response(conversation_messages, context))
                st.session_state.doc_messages.append({"role": "assistant", "content": response})




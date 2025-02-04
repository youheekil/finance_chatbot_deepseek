# https://github.com/laxmimerit/ollama-chatbot

import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

st.title(":brain: Make My Own History Enabled Chat Application with Deepseek, Ollama and Langchian!")
st.write("SIDE PROJECT - LLM FOR FINANCE")

model_name = "deepseek-r1:7b"

model = ChatOllama(model=model_name, 
                   base_url="http://localhost:11434")

st.write("DeepSeek is a powerful model that can generate human-like text. DeepSeek’s first-generation reasoning models, achieving performance comparable to OpenAI-o1 across math, code, and reasoning tasks.")


system_message = SystemMessagePromptTemplate.from_template(
    "You are helpful AI assistant. You work as a software programmer who like to code in short and correct. You also like to use a lot of print for debugging."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("llm-form"):
    text = st.text_area("Enter your question here.") 
    submit = st.form_submit_button("Submit")

def generate_response(chat_history):
    """pass chat history to the model and then create template and return the output"""
    chat_template = ChatPromptTemplate.from_messages(chat_history)

    chain = chat_template|model|StrOutputParser()
    
    response = chain.invoke({})
    
    return response

# user message in 'user' key
# ai message in 'assistant' key

def get_history():
    chat_history = [system_message]
    for chat in st.session_state.chat_history:
        chat_history.append(HumanMessagePromptTemplate.from_template(chat["user"]))
        chat_history.append(AIMessagePromptTemplate.from_template(chat["assistant"]))
    return chat_history

if submit and text:
    with st.spinner("Processing..."):
        promt = HumanMessagePromptTemplate.from_template(text)
        chat_history = get_history()
        chat_history.append(promt)
        response = generate_response(chat_history)
        st.session_state.chat_history.append({"user": text, "assistant": response})

st.write("## Chat History")
for chat in reversed(st.session_state.chat_history):
    st.write(f"**:adult: User**: {chat['user']}")
    st.write(f"**:brain: Assistant**: {chat['assistant']}")
    st.write("---")
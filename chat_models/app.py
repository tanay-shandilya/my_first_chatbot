import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Suppress minor ecosystem warnings
warnings.filterwarnings("ignore")
load_dotenv()

# 1. Page Configuration
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Your AI Chatbot")
st.caption("Powered by Gemini")

# 2. Verify API Key
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing Gemini API Key. Please add `GEMINI_API_KEY` to your `.env` file.")
    st.stop()

# 3. Model Initialization (cached so it does not reload on every user interaction)
@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.3,
        streaming=True
    )

model = load_model()

# 4. Initialize Persistent Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful, direct, and concise AI assistant.")
    ]

# 5. Sidebar Controls (Clear Chat History)
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful, direct, and concise AI assistant.")
        ]
        st.rerun()

# 6. Render Previous Conversation Turns
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 7. Handle New User Input
if user_prompt := st.chat_input("Type your message here..."):
    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Append to state
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))

    # Stream the assistant's response live
    with st.chat_message("assistant"):
        def generate_response():
            for chunk in model.stream(st.session_state.chat_history):
                yield chunk.content

        # st.write_stream renders tokens chunk by chunk and captures the full string
        full_response = st.write_stream(generate_response())

    # Append the assistant's completed message to state
    st.session_state.chat_history.append(AIMessage(content=full_response))
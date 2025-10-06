import streamlit as st
import base64
import os

st.set_page_config(page_title="Cognitive Mirror", layout="wide")
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Predefined questions for psychiatrist-like interview
questions = [
    "Can you tell me about your family background?",
    "What was your childhood like?",
    "How do you handle stress in your daily life?",
    "What are your goals and aspirations?",
    "How do you feel about your relationships with others?",
    "Have you experienced any significant traumas in your life?",
    "What makes you happy?",
    "How do you cope with negative emotions?",
    "What are your strengths and weaknesses?",
    "How do you see your future?"
]

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=3000
)

# Prompt for analysis
analysis_prompt = PromptTemplate(
    input_variables=["history"],
    template="Analyze the following conversation history for tone, emotions, and themes: {history}\nProvide a psychological and emotional profile summary."
)

analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

def set_bg_with_overlay(img_path, overlay_rgba="rgba(0,0,0,0)"):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient({overlay_rgba}, {overlay_rgba}), url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: black;
        }}
        .stApp .css-1d391kg {{ /* container text background tweak (class may vary) */
            background: rgba(255,255,255,0.0);
        }}
        .stApp * {{
            color: black !important;
        }}
        .stApp button {{
            color: white !important;
        }}
        .stApp button:first-of-type {{
            background-color: #00FFFF !important;
        }}
        .stApp .stTextInput label {{
            color: white !important;
        }}
        .stApp .stTextInput input {{
            color: black !important;
            background-color: #00FFFF !important;
        }}
        .stChatMessage {{
            background-color: #00FFFF !important;
            border-radius: 15px !important;
            padding: 10px !important;
            margin: 5px 0 !important;
            color: black !important;
        }}
        .stChatMessage[data-testid="stChatMessage-user"] {{
            background-color: #87CEEB !important; /* Light aqua for user */
        }}
        .stChatMessage[data-testid="stChatMessage-assistant"] {{
            background-color: #00FFFF !important; /* Aqua for assistant */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_with_overlay("pic.jpg", overlay_rgba="rgba(255,255,255,0.35)")

# Streamlit app
st.title("Cognitive Mirror: AI-Powered Interviewer")

if "responses" not in st.session_state:
    st.session_state.responses = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "session_started" not in st.session_state:
    st.session_state.session_started = False

if st.button("Start Session"):
    st.session_state.session_started = True
    st.session_state.responses = []
    st.session_state.current_question_index = 0

if st.session_state.session_started:
    # Display conversation history
    st.write("### Conversation History")
    for i, response in enumerate(st.session_state.responses):
        with st.chat_message("assistant"):
            st.write(questions[i])
        with st.chat_message("user"):
            st.write(response)

    # Ask current question
    if st.session_state.current_question_index < len(questions):
        current_question = questions[st.session_state.current_question_index]
        with st.chat_message("assistant"):
            st.write(f"Question {st.session_state.current_question_index + 1}: {current_question}")
        user_input = st.text_input("Your response:", key="user_input")

        if st.button("Submit Response"):
            if user_input:
                st.session_state.responses.append(user_input)
                st.session_state.current_question_index += 1
                st.rerun()
    else:
        # All questions answered, generate profile automatically
        with st.chat_message("assistant"):
            st.write("All questions answered. Generating your psychological profile...")
        # Concatenate all Q&A pairs into a single string for analysis
        history = ""
        for i, response in enumerate(st.session_state.responses):
            history += f"Q: {questions[i]}\nA: {response}\n"
        summary = analysis_chain.run(history=history)
        with st.chat_message("assistant"):
            st.write("Psychological Profile Summary:")
            st.write(summary)

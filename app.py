import streamlit as st
from openai import OpenAI
from openai import AsyncOpenAI
from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable
from langchain.tools import StructuredTool
from dotenv import load_dotenv
from assistant_wrapper import OpenAIAssistantWrapper
import asyncio
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')


# Function to interact with secondary assistant for search queries
def search_web(query):
    """
    Sends a search query to the Perplexity API via the second assistant.
    """
    pplx_model = "llama-3.1-sonar-large-128k-online"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    client = OpenAI(
        api_key="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",
        base_url="https://api.perplexity.ai",
    )

    reply = client.chat.completions.create(
        model=pplx_model,
        messages=messages,
    )
    reply_text = reply.choices[0].message.content
    return reply_text


tools = [
    StructuredTool.from_function(search_web),
]

client = OpenAI(api_key=openai_api_key)
assistant = OpenAIAssistantRunnable.create_assistant(
    client=client,
    name="Main assistant",
    instructions="Generic AI Assistant.",
    tools=tools,
    model="gpt-4",
    as_agent=True,
)

# Create the wrapper
assistant_wrapper = OpenAIAssistantWrapper(assistant, tools)

def ask_assistant(input_text):
    return assistant_wrapper.invoke(input_text)

# Streamlit UI
st.title("Dual Assistant Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = ask_assistant(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Start New Conversation"):
    st.session_state.messages = []
    st.session_state.thread_id = None
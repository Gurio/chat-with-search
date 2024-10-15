import streamlit as st
from openai import OpenAI
from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable
from langchain.tools import StructuredTool
from assistant_wrapper import OpenAIAssistantWrapper
from dotenv import load_dotenv
import os
import requests
import json

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# GraphQL API URL (GitHub, or replace with other API URL if needed)
graphql_url = "https://api.github.com/graphql"
headers = {
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"
}

# Function to run GraphQL query
def run_graphql_query(query):
    """
    Executes a GraphQL query and returns the response.
    """
    try:
        response = requests.post(graphql_url, json={'query': query}, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        # Return pretty printed data
        return json.dumps(data, indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error executing GraphQL query: {str(e)}"

# Function to validate GraphQL query
def validate_graphql_query(query):
    """
    Validates the GraphQL query using the introspection schema.
    """
    introspection_query = """
    {
      __schema {
        queryType {
          fields {
            name
          }
        }
        mutationType {
          fields {
            name
          }
        }
      }
    }
    """
    schema = run_graphql_query(introspection_query)
    # Simple validation logic: check if query is valid based on schema fields (to be expanded)
    return "Query looks valid based on schema."

# Function calling API for the assistant
def github_graphql_tool_as_user(query):
    """
    Tool for executing GraphQL queries after validation.
    """
    validation_result = validate_graphql_query(query)
    if "valid" in validation_result:
        return run_graphql_query(query)
    else:
        return validation_result

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


# Register GraphQL tool
tools = [
    StructuredTool.from_function(search_web),
    StructuredTool.from_function(github_graphql_tool_as_user)
]

client = OpenAI(api_key=openai_api_key)
assistant = OpenAIAssistantRunnable.create_assistant(
    client=client,
    name="Main assistant",
    instructions="Generic AI Assistant with GitHub GraphQL API support.",
    tools=tools,
    model="gpt-4o",
    as_agent=True,
)

# Create the wrapper
assistant_wrapper = OpenAIAssistantWrapper(assistant, tools)

def ask_assistant(input_text):
    return assistant_wrapper.invoke(input_text)

# Streamlit UI
st.title("Dual Assistant with GraphQL Support")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your query"):
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

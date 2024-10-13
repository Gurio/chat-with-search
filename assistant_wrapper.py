from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable
from langchain.tools import BaseTool

import streamlit as st

class OpenAIAssistantWrapper:
    def __init__(self, assistant: OpenAIAssistantRunnable, tools: list[BaseTool]):
        self.assistant = assistant
        self.tools = {tool.name: tool for tool in tools}
        # Use session state to persist thread_id
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = None

    def _execute_tools(self, actions):
        tool_outputs = []
        for action in actions:
            if action.tool in self.tools:
                output = self.tools[action.tool].run(action.tool_input)
                tool_outputs.append({"output": output, "tool_call_id": action.tool_call_id})
        return tool_outputs

    def invoke(self, input_text: str):
        if st.session_state.thread_id is None:
            response = self.assistant.invoke({"content": input_text})
        else:
            response = self.assistant.invoke({
                "content": input_text,
                "thread_id": st.session_state.thread_id
            })

        while isinstance(response, list) and len(response) > 0 and hasattr(response[0], 'tool'):
            tool_outputs = self._execute_tools(response)
            response = self.assistant.invoke({
                "tool_outputs": tool_outputs,
                "run_id": response[0].run_id,
                "thread_id": response[0].thread_id,
            })

        if hasattr(response, 'thread_id'):
            st.session_state.thread_id = response.thread_id

        return response.return_values["output"] if hasattr(response, 'return_values') else response

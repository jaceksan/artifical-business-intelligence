import asyncio

import streamlit as st
from streamlit_chat import message

from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from gooddata_api_client.model.chat_history_result import ChatHistoryResult

CHAT_HISTORY = "chat_history"
LAST_INTERACTION_ID = "last_interaction_id"

class GoodDataAiChatApp:
    def __init__(self, gd_sdk: GoodDataSdkWrapper) -> None:
        self.gd_sdk = gd_sdk
        self.workspace_id = st.session_state.workspace_id

    @staticmethod
    def init_chat_history():
        if CHAT_HISTORY not in st.session_state:
            st.session_state[CHAT_HISTORY] = []
        if LAST_INTERACTION_ID not in st.session_state:
            st.session_state[LAST_INTERACTION_ID] = 0

    async def ask_question(self, question: str):
        return self.gd_sdk.sdk.compute.ai_chat(workspace_id=self.workspace_id, question=question)

    def get_chat_history(self, last_interaction_id: int = 0) -> ChatHistoryResult:
        return self.gd_sdk.sdk.compute.ai_chat_history(
            workspace_id=self.workspace_id, chat_history_interaction_id=last_interaction_id
        )

    def cache_chat_history(self):
        last_interaction_id = st.session_state[LAST_INTERACTION_ID]
        print(f"Last interaction id: {last_interaction_id}")
        chat_history = self.get_chat_history(last_interaction_id)
        print(f"Chat history: {chat_history}")
        for interaction in chat_history["interactions"]:
            st.session_state[CHAT_HISTORY].append(interaction)
            st.session_state[LAST_INTERACTION_ID] = int(interaction['chatHistoryInteractionId'])
            print(f"NEW: Last interaction id={st.session_state[LAST_INTERACTION_ID]} interaction={interaction}")

    def poll_chat_history(self):
        finished = False
        while not finished:
            self.cache_chat_history()
            finished = st.session_state[CHAT_HISTORY][-1]["interactionFinished"]

    @staticmethod
    def render_chat_history():
        if st.session_state[CHAT_HISTORY]:
            for interaction in reversed(st.session_state[CHAT_HISTORY]):
                # TODO - other types
                text_response = interaction.get("textResponse")
                if not text_response:
                    print(f"Skipping interaction: {interaction}")
                else:
                    message(interaction["question"], is_user=True, key=f"{interaction["chatHistoryInteractionId"]}_user")
                    message(text_response, key=f"{interaction["chatHistoryInteractionId"]}_ai")


    def render(self) -> None:
        self.init_chat_history()
        self.cache_chat_history()
        user_input = st.text_area("Ask GoodData a question:")
        columns = st.columns(2)
        with columns[0]:
            submit = st.button("Submit Query", type="primary")
        # with columns[1]:
        #     if st.button("Clear chat history", type="primary"):
        #         st.session_state[CHAT_HISTORY] = []

        if submit:
            asyncio.run(self.ask_question(user_input))
            self.poll_chat_history()

        self.render_chat_history()

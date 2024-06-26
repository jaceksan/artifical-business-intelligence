import openai
import streamlit as st
from streamlit_chat import message

from gooddata.agents.libs.gd_openai import GoodDataOpenAICommon


class GoodDataChatApp:
    def __init__(self) -> None:
        self.agent = GoodDataOpenAICommon(
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            workspace_id=st.session_state.workspace_id,
        )

    def render(self) -> None:
        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        try:
            chain = self.agent.get_conversation_chain()
            user_input = st.text_area("Ask GoodData a question:")

            columns = st.columns(2)
            with columns[0]:
                if st.button("Submit Query", type="primary"):
                    output = chain.run(input=user_input)
                    st.session_state.past.append(user_input)
                    st.session_state.generated.append(output)
            with columns[1]:
                if st.button("Clear chat history", type="primary"):
                    st.session_state["generated"] = []
                    st.session_state["past"] = []

            if st.session_state["generated"]:
                for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        except openai.AuthenticationError as e:
            st.error("OpenAI unknown authentication error")
            st.error(e)

import streamlit as st
from openapi_parser import parse
from openapi_parser.specification import Specification

from gooddata.agents.api_agent import ApiAgent


class GoodDataApiExecutorApp:
    def __init__(self) -> None:
        self.agent = ApiAgent(
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            workspace_id=st.session_state.workspace_id,
        )

    def render(self):
        query = st.text_area("Enter question:")
        if st.button("Submit Query", type="primary"):
            if query:
                answer = self.agent.process(query, get_api_spec())
                st.dataframe(answer)


@st.cache_data
def get_api_spec() -> Specification:
    return parse("gooddata/open-api-spec.json", strict_enum=False)

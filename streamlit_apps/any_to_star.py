import streamlit as st
from gooddata_sdk import CatalogDataSource, GoodDataSdk

from gooddata.agents.any_to_star_agent import AnyToStarResultType, GoodDataAnyToStarAgent
from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from gooddata.tools import get_name_for_id


class GoodDataAnyToStarApp:
    def __init__(self, gd_sdk: GoodDataSdkWrapper) -> None:
        self.gd_sdk = gd_sdk
        self.agent = GoodDataAnyToStarAgent(
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            workspace_id=st.session_state.workspace_id,
        )

    def render_data_source_picker(self):
        data_sources = list_data_sources(self.gd_sdk.sdk)
        if "data_source_id" not in st.session_state:
            st.session_state["data_source_id"] = data_sources[0].id
        st.selectbox(
            label="Data sources:",
            options=[x.id for x in data_sources],
            format_func=lambda x: get_name_for_id(data_sources, x),
            key="data_source_id",
        )

    @staticmethod
    def render_result_type_picker():
        if "result_type" not in st.session_state:
            st.session_state["result_type"] = AnyToStarResultType.SQL.name
        st.selectbox(
            label="Result type:",
            options=[x.name for x in list(AnyToStarResultType)],
            key="result_type",
        )

    def render(self):
        columns = st.columns(2)
        with columns[0]:
            self.render_data_source_picker()
        with columns[1]:
            self.render_result_type_picker()

        st.info("This agent can be quite slow, please be patient.")

        if st.button("Generate", type="primary"):
            try:
                data_source_id = st.session_state.data_source_id
                result_type = AnyToStarResultType[st.session_state.result_type]
                result = self.agent.process(data_source_id, result_type)
                st.write(result)
            except Exception as e:
                st.error(str(e))


@st.cache_data
def list_data_sources(_sdk: GoodDataSdk) -> list[CatalogDataSource]:
    return _sdk.catalog_data_source.list_data_sources()

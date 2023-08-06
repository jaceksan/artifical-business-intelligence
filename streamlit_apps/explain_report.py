import pandas as pd
import streamlit as st
from gooddata_pandas import GoodPandas
from gooddata_sdk import GoodDataSdk, Insight

from gooddata.agents.explain_report_agent import GoodDataExplainReportAgent
from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from gooddata.tools import get_title_for_id


class GoodDataExplainReportApp:
    def __init__(self, gd_sdk: GoodDataSdkWrapper) -> None:
        self.gd_sdk = gd_sdk
        self.agent = GoodDataExplainReportAgent(
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            workspace_id=st.session_state.workspace_id,
        )

    def render_insight_picker(self, workspace_id: str):
        insights = get_insights(self.gd_sdk.sdk, workspace_id)
        st.selectbox(
            label="Insights:",
            options=[w.id for w in insights],
            format_func=lambda x: get_title_for_id(insights, x),
            key="insight_id",
        )

    def render(self):
        workspace_id = self.agent.workspace_id
        self.render_insight_picker(workspace_id)
        insight_id = st.session_state.get("insight_id")
        if insight_id:
            query = st.text_area("Enter question about this insight:")
            if st.button("Submit Query", type="primary"):
                if query:
                    try:
                        result = self.agent.process(query, insight_id)
                        print(result)
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                        else:
                            st.info(result)
                    except Exception as e:
                        st.error(str(e))
            if st.checkbox("Data preview"):
                st.dataframe(execute_insight(self.gd_sdk.pandas, st.session_state.workspace_id, insight_id))


@st.cache_data
def get_insights(_sdk: GoodDataSdk, workspace_id: str) -> list[Insight]:
    return _sdk.insights.get_insights(workspace_id)


@st.cache_data
def execute_insight(_pandas: GoodPandas, workspace_id: str, insight_id: str) -> pd.DataFrame:
    return _pandas.data_frames(workspace_id).for_insight(insight_id)

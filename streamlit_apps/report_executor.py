import pandas as pd
import streamlit as st

from gooddata.agents.common import AIMethod
from gooddata.agents.report_agent import ReportAgent
from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from streamlit_apps.constants import ChartType


class GoodDataReportExecutorApp:
    def __init__(self, gd_sdk: GoodDataSdkWrapper) -> None:
        self.gd_sdk = gd_sdk
        self.workspace_id = st.session_state.workspace_id
        self.agent = ReportAgent(
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            workspace_id=self.workspace_id,
        )

    @staticmethod
    def render_openai_model_methods_picker():
        if "openai_method" not in st.session_state:
            st.session_state["openai_method"] = AIMethod.RAW.name
        st.selectbox(
            label="OpenAI model method:",
            options=[x.name for x in list(AIMethod)],
            format_func=lambda x: AIMethod[x].value,
            key="openai_method",
        )

    @staticmethod
    def render_chart_type_picker():
        if "chart_type" not in st.session_state:
            st.session_state["chart_type"] = ChartType.TABLE.name
        st.selectbox(
            label="Chart type:",
            options=[x.name for x in list(ChartType)],
            format_func=lambda x: ChartType[x].value,
            key="chart_type",
        )

    def show_model_entities(self) -> None:
        columns = st.columns(2)
        with columns[0]:
            st.write("Attributes:")
            attributes_string = ""
            i = 1
            for attribute in self.gd_sdk.attributes(self.workspace_id):
                attributes_string += f"{i}. `{attribute[1]}`\n"
                i += 1
            st.markdown(attributes_string)
        with columns[1]:
            st.write("Metrics:")
            metrics_string = ""
            i = 1
            for metric in self.gd_sdk.metrics(self.workspace_id):
                metrics_string += f"{i}. `{metric[1]}`\n"
            st.markdown(metrics_string)

    def render(self):
        columns = st.columns(2)
        with columns[0]:
            self.render_openai_model_methods_picker()
        with columns[1]:
            self.render_chart_type_picker()
        chart_type = ChartType[st.session_state.get("chart_type")]
        query = st.text_area("Enter question:")
        if st.button("Submit Query", type="primary"):
            if query:
                method = AIMethod[st.session_state.openai_method]
                df, attributes, metrics = agent_process(self.agent, method, query)
                if chart_type == ChartType.TABLE:
                    st.dataframe(df)
                else:
                    print(df)
                    df.set_index(df.columns.values[0], inplace=True)
                    # TODO - what the fuck? Looks like a Streamlit bug
                    df.index.name = None
                    print(df)
                    if chart_type == ChartType.BAR_CHART:
                        st.bar_chart(df)
                    elif chart_type == ChartType.LINE_CHART:
                        st.line_chart(df)
                    else:
                        st.error(f"Unsupported chart type {chart_type}")
        if st.checkbox("Show model entities"):
            self.show_model_entities()


@st.cache_data
def agent_process(_agent: ReportAgent, openai_method: AIMethod, query: str) -> tuple[pd.DataFrame, list, list]:
    return _agent.process(openai_method, query)

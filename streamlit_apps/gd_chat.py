import asyncio
from time import sleep

import streamlit as st
from gooddata_api_client.model.chat_history_result import ChatHistoryResult
from gooddata_sdk import Attribute, ExecutionDefinition, ObjId, SimpleMetric, TableDimension
from streamlit_chat import message

from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper

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

    @property
    def interaction_finished(self) -> bool:
        return st.session_state[CHAT_HISTORY] and st.session_state[CHAT_HISTORY][-1]["interactionFinished"]

    @property
    def last_interaction_id(self) -> bool:
        return st.session_state[LAST_INTERACTION_ID]

    def cache_chat_history(self) -> bool:
        last_interaction_id = st.session_state[LAST_INTERACTION_ID]
        chat_history = self.get_chat_history(last_interaction_id)
        last_changed = True
        for interaction in chat_history["interactions"]:
            if self.interaction_finished:
                st.session_state[CHAT_HISTORY].append(interaction)
                st.session_state[LAST_INTERACTION_ID] = int(interaction["chatHistoryInteractionId"])
            else:
                if self.interaction_finished:
                    st.session_state[CHAT_HISTORY].append(interaction)
                else:
                    # Replace last (unfinished) interaction with the new one
                    if st.session_state[CHAT_HISTORY]:
                        if st.session_state[CHAT_HISTORY][-1] == interaction:
                            last_changed = False
                        st.session_state[CHAT_HISTORY][-1] = interaction
                    else:
                        st.session_state[CHAT_HISTORY].append(interaction)
        return last_changed

    def poll_chat_history(self):
        finished = False
        while not finished:
            if self.cache_chat_history():
                self.render_chat_history()
            finished = self.interaction_finished
            sleep(1)

    @staticmethod
    def render_text_response(index: int, interaction: dict, text_response: str, use_case: str) -> None:
        chat_history_interaction_id = interaction["chatHistoryInteractionId"]
        message(text_response, key=f"{chat_history_interaction_id}_ai_{index}_{use_case}")

    def render_visualization(self, created_visualizations_response: dict) -> None:
        metrics = []
        for metric in created_visualizations_response.get("metrics", []):
            if metric["type"] == "metric":
                metrics.append(SimpleMetric(local_id=metric["id"], item=ObjId(id=metric["id"], type=metric["type"])))
            else:
                metrics.append(
                    SimpleMetric(
                        local_id=metric["id"],
                        item=ObjId(id=metric["id"], type=metric["type"]),
                        aggregation=metric["aggFunction"],
                    )
                )
        dimension_ids = []
        dimension_titles = []
        attributes = []
        for ai_dimension in created_visualizations_response.get("dimensionality", []):
            dimension_ids.append(ai_dimension["id"])
            dimension_titles.append(ai_dimension["title"])
            attributes.append(Attribute(local_id=ai_dimension["id"], label=ai_dimension["id"]))
        # TODO - pivoting
        dimensions = [TableDimension(item_ids=dimension_ids), TableDimension(item_ids=["measureGroup"])]
        exec_def = ExecutionDefinition(
            attributes=attributes,
            metrics=metrics,
            filters=[],
            dimensions=dimensions,
        )
        print(f"exec_def: {exec_def.as_api_model()}")
        frames = self.gd_sdk.pandas.data_frames(self.workspace_id)
        df, df_metadata = frames.for_exec_def(exec_def)
        df_from_result_id, df_metadata_from_result_id = frames.for_exec_result_id(
            result_id=df_metadata.execution_response.result_id,
        )
        df_from_result_id.columns = df_from_result_id.columns.map("".join)
        visualization_type = created_visualizations_response.get("visualizationType")
        if visualization_type == "BAR":
            df_from_result_id.reset_index(inplace=True)
            df_from_result_id.set_index(dimension_titles[0], inplace=True)
            st.bar_chart(df_from_result_id)
        elif visualization_type == "LINE":
            df_from_result_id.reset_index(inplace=True)
            df_from_result_id.set_index(dimension_titles[0], inplace=True)
            st.line_chart(df_from_result_id)
        else:
            st.dataframe(df_from_result_id)

    def render_chat_history(self):
        if st.session_state[CHAT_HISTORY]:
            for i, interaction in enumerate(reversed(st.session_state[CHAT_HISTORY])):
                text_response = interaction.get("textResponse")
                route_response = interaction.get("routing")
                found_objects_response = interaction.get("foundObjects")
                created_visualizations_response = interaction.get("createdVisualizations")
                # Render all steps of the reasoning
                if created_visualizations_response:
                    print(f"created_visualizations_response: {created_visualizations_response}")
                    self.render_text_response(
                        i, interaction, created_visualizations_response["reasoning"], "visualization"
                    )
                    if "objects" in created_visualizations_response:
                        # TODO - display more than one visualization
                        self.render_visualization(created_visualizations_response["objects"][0])
                if found_objects_response:
                    self.render_text_response(i, interaction, found_objects_response["reasoning"], "found objects")
                if text_response:
                    self.render_text_response(i, interaction, text_response, "text response")
                if route_response:
                    self.render_text_response(i, interaction, route_response["reasoning"], "routing")
                chat_history_interaction_id = interaction["chatHistoryInteractionId"]
                message(interaction["question"], is_user=True, key=f"{chat_history_interaction_id}_user_{i}")

    def render(self) -> None:
        self.init_chat_history()
        self.cache_chat_history()
        user_input = st.text_area("Ask GoodData a question:")
        columns = st.columns(2)
        with columns[0]:
            submit = st.button("Submit Query", type="primary")
        with columns[1]:
            if st.button("Clear chat history", type="primary"):
                st.session_state[CHAT_HISTORY] = []
                self.gd_sdk.sdk.compute.ai_chat_history_reset(self.workspace_id)

        if submit:
            asyncio.run(self.ask_question(user_input))
            self.poll_chat_history()
        else:
            self.render_chat_history()

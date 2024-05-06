from typing import Optional

import attr
import json
import re
from pathlib import Path
from time import time
from enum import Enum

import openai
import streamlit as st
from langchain_core.documents import Document

from gooddata.agents.libs.utils import debug_to_file, replace_in_string
from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from gooddata.agents.libs.rag_langchain import timeit, GoodDataRAGSimple, PRODUCT_NAME, VectorDB
from gooddata.tools import get_org_id_from_host
from streamlit_apps.gooddata.catalog import GoodDataCatalog, get_gooddata_full_catalog

DISTANCE_KEY = "_distance"
SCORE_KEY = "_similarity"
DOCUMENT_DEBUG_PATH = Path("RAG")
SEARCH_SYSTEM_TEMPLATE = """
You are expert for {PRODUCT_NAME}.
You can only search in a context and return relevant objects.
You return only JSON containing all found objects.
Example:
{{
    "objects": [
        {{"id": "start_date.month", "type": "attribute", "title": "Start date(month)"}},
        {{"id": "product_type", "type": "attribute", "title": "Product type"}},
        {{"id": "amount", "type": "fact", "title": "Amount" }},
        {{"id": "revenue", "type": "metric", "title": "Revenue"}},
        {{"id": "revenue_by_month", "type": "visualization", "title": "Revenue by month"}},
        {{"id": "revenue_overview", "type": "dashboard", "title": "Revenue overview"}}
    ]
}}

You answer the question based only on the following context:
----------------------------------------------------------------------------------------
{context}
----------------------------------------------------------------------------------------
"""
SEARCH_QUESTION_TEMPLATE = """
Question:
Find {PRODUCT_NAME} objects in the above context which ID or title relate most to this keyword "{input}".
Less but still relevant is when the keyword relate to objects mentioned in the context as related to the main object.
"""

SEARCH_TEMPLATE = """
You are expert for {PRODUCT_NAME}.
You can only search in a context and return relevant objects.
You return only JSON containing all found objects.
Example:
{{
    "objects": [
        {{"id": "start_date.month", "type": "attribute", "title": "Start date(month)"}},
        {{"id": "product_type", "type": "attribute", "title": "Product type"}},
        {{"id": "amount", "type": "fact", "title": "Amount" }},
        {{"id": "revenue", "type": "metric", "title": "Revenue"}},
        {{"id": "revenue_by_month", "type": "visualization", "title": "Revenue by month"}},
        {{"id": "revenue_overview", "type": "dashboard", "title": "Revenue overview"}}
    ]
}}

You answer the question based only on the following context:
----------------------------------------------------------------------------------------
{context}
----------------------------------------------------------------------------------------
Question:
{question}."""


class RAGUseCase(Enum):
    NAIVE = "Naive"
    VECTOR_SEARCH = "Vector search"
    RAG = "RAG"


@attr.s(auto_attribs=True, kw_only=True)
class Result:
    use_case: RAGUseCase
    raw_response: any
    json_response: Optional[dict]
    duration: int
    vector_duration: Optional[int] = None
    contains_distance: Optional[bool] = None
    contains_score: Optional[bool] = None


class GoodDataRAGApp:
    def __init__(self, gd_sdk: GoodDataSdkWrapper) -> None:
        self.gd_sdk = gd_sdk
        # TODO - can connect to GD with host:token. Where to get ORG_ID?
        self.org_id = gd_sdk.profile or get_org_id_from_host(gd_sdk.host)
        self.workspace_id = st.session_state.workspace_id

    @staticmethod
    def init_session_state():
        if "rag_enabled" not in st.session_state:
            st.session_state["rag_enabled"] = True
        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

    def _get_agent(self, max_search_results: int) -> GoodDataRAGSimple:
        return GoodDataRAGSimple(
            org_id=self.org_id,
            workspace_id=self.workspace_id,
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            max_search_results=max_search_results,
            vector_db=VectorDB[st.session_state.vector_db],
        )

    @staticmethod
    def render_header(catalog: GoodDataCatalog) -> None:
        columns = st.columns(7)
        with columns[0]:
            st.write(f"Datasets: {len(catalog.ldm.ldm.datasets)}")
        with columns[1]:
            st.write(f"Date datasets: {len(catalog.ldm.ldm.date_instances)}")
        with columns[2]:
            st.write(f"Attributes: {sum([len(d.attributes) for d in catalog.ldm.ldm.datasets])}")
        with columns[3]:
            st.write(f"Facts: {sum([len(d.facts) for d in catalog.ldm.ldm.datasets])}")
        with columns[4]:
            st.write(f"Metrics: {len(catalog.adm.analytics.metrics)}")
        with columns[5]:
            st.write(f"Visualizations: {len(catalog.adm.analytics.visualization_objects)}")
        with columns[6]:
            st.write(f"Dashboards: {len(catalog.adm.analytics.analytical_dashboards)}")

    @staticmethod
    def render_rag_enabled_checkbox():
        st.checkbox(
            label="RAG enabled:", value=True, key="rag_enabled",
        )

    @staticmethod
    def render_use_case_dropdown():
        st.selectbox(
            label="Use cases:",
            options=[use_case.value for use_case in RAGUseCase],
            key="rag_use_case",
        )

    @staticmethod
    def render_reset_db_button(agent: GoodDataRAGSimple):
        if st.button("Reset DB"):
            conn = agent.connect_to_db()
            agent.drop_table(conn)

    @staticmethod
    def render_vector_db_dropdown():
        st.selectbox(
            label="VectorDB:",
            options=[db.name for db in VectorDB],
            key="vector_db",
        )

    @staticmethod
    def extract_json(text: str):
        # Regular expression pattern to find text enclosed in {}
        pattern = r'(\{.*\})'
        # Search the text for the pattern
        if match := re.search(pattern, text, re.S):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def get_response_header(duration_answer):
        return (
                f"Generation took {duration_answer} ms, rag_enabled = {st.session_state.rag_enabled}, " +
                f"use_case={st.session_state.rag_use_case}, model={st.session_state.openai_model}"
        )

    def get_gooddata_full_catalog(self) -> GoodDataCatalog:
        catalog = get_gooddata_full_catalog(self.gd_sdk, self.workspace_id, DOCUMENT_DEBUG_PATH)
        # Without RAG we send all documents as one string to OpenAI.
        # So we mark every section as paragraph, not document.
        document_type = "document" if st.session_state.rag_enabled else "paragraph"
        for d in catalog.documents:
            d.page_content = d.page_content.format(document=document_type)
        return catalog

    @staticmethod
    def init_vector_store(agent: GoodDataRAGSimple, documents: list[Document]):
        start_vector = time()
        vector_store = agent.init_vector_store(documents)
        duration_vector = int((time() - start_vector) * 1000)
        return vector_store, duration_vector

    @staticmethod
    def report_result(result: Result):
        columns = st.columns(3)
        with columns[0]:
            st.info(f"Search duration: {result.duration} ms")
        with columns[1]:
            if result.vector_duration is None:
                st.info(f"Vector store irrelevant in this case")
            else:
                st.info(f"Vector store load duration: {result.vector_duration} ms")
        with columns[2]:
            if result.contains_distance is True or result.contains_score is True:
                st.info(f"Result contains cosine distance or score")
            elif result.contains_distance is False and result.contains_score is False:
                st.warning(f"Result does NOT contain cosine distance or score")
            else:
                st.info(f"Cosine distance/score irrelevant in this case")

        if result.json_response:
            st.json(result.json_response)
        elif result.use_case == RAGUseCase.VECTOR_SEARCH:
            objects = [r.metadata for r in result.raw_response]

            if result.contains_distance:
                sorted_objects = sorted(objects, key=lambda o: o[DISTANCE_KEY])
            elif result.contains_score:
                sorted_objects = sorted(objects, key=lambda o: o[SCORE_KEY], reverse=True)
            else:
                sorted_objects = objects
            result_json = {
                # Do not display the content. st.json() does not provide an option to make it a tooltip ;-(
                "objects": sorted_objects
            }
            st.json(result_json)
        else:
            # Fallback if response is not a valid JSON
            st.warning(f"Result is not a valid JSON:\n{result.raw_response}")

    @timeit
    def render(self) -> None:
        try:
            self.init_session_state()
            catalog = get_gooddata_full_catalog(self.gd_sdk, self.workspace_id, DOCUMENT_DEBUG_PATH)
            columns = st.columns(4)
            with columns[0]:
                self.render_use_case_dropdown()
            with columns[1]:
                result_count = st.number_input("Max search results", min_value=1, max_value=100, value=10)
            with columns[2]:
                self.render_vector_db_dropdown()
                agent = self._get_agent(result_count)
            with columns[3]:
                self.render_reset_db_button(agent)
            self.render_header(catalog)

            if (input := st.text_input("Search: ", type="default")):
                if st.session_state.rag_use_case == RAGUseCase.NAIVE.value:
                    context = "\n".join([d.page_content for d in catalog.documents])
                    system_prompt = replace_in_string(SEARCH_SYSTEM_TEMPLATE, {"PRODUCT_NAME": PRODUCT_NAME, "context": context})
                    user_prompt = replace_in_string(SEARCH_QUESTION_TEMPLATE, {"PRODUCT_NAME": PRODUCT_NAME, "input": input})
                    debug_to_file("rag_naive.txt", system_prompt + user_prompt, DOCUMENT_DEBUG_PATH)
                    start_answer = time()
                    response_choices = agent.gd_openai.ask_chat_completion(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                    response = response_choices.choices[0].message.content
                    result = Result(
                        use_case=RAGUseCase.NAIVE,
                        raw_response=response,
                        json_response=self.extract_json(response),
                        duration=int((time() - start_answer) * 1000),
                    )
                else:
                    vector_store, vector_duration = self.init_vector_store(agent, catalog.documents)
                    if st.session_state.rag_use_case == RAGUseCase.VECTOR_SEARCH.value:
                        start_answer = time()
                        response = agent.similarity_search(vector_store, input)
                        # Vector stores in LangChain do not provide distance or score in an uniform way
                        contains_distance = response[0].metadata.get(DISTANCE_KEY, None) is not None  # LanceDB
                        contains_score = response[0].metadata.get(SCORE_KEY, None) is not None   # DuckDB
                        result = Result(
                            use_case=RAGUseCase.VECTOR_SEARCH,
                            raw_response=response,
                            json_response=None,
                            duration=int((time() - start_answer) * 1000),
                            vector_duration=vector_duration,
                            contains_distance=contains_distance,
                            contains_score=contains_score,
                        )
                    elif st.session_state.rag_use_case == RAGUseCase.RAG.value:
                        start_answer = time()
                        question = f"""Find {PRODUCT_NAME} objects in the above context related to "{input}"."""
                        response = agent.rag_chain_invoke(
                            rag_retriever=agent.get_rag_retriever(vector_store),
                            question=question,
                            answer_prompt=replace_in_string(SEARCH_TEMPLATE, {"PRODUCT_NAME": PRODUCT_NAME})
                        )
                        result = Result(
                            use_case=RAGUseCase.RAG,
                            raw_response=response,
                            json_response=self.extract_json(response),
                            duration=int((time() - start_answer) * 1000),
                            vector_duration=vector_duration,
                        )
                    else:
                        raise ValueError(f"Unknown use case: {st.session_state.rag_use_case}")

                self.report_result(result)

        except openai.AuthenticationError as e:
            st.write("OpenAI unknown authentication error")
            st.write(e)

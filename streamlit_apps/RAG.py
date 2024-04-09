import os
import attr

import openai
import streamlit as st
from streamlit_chat import message

from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from gooddata_sdk import CatalogDeclarativeModel, CatalogDeclarativeAnalytics
from gooddata.agents.libs.rag_langchain import timeit, GoodDataRAGSimple


class GoodDataRAGApp:
    def __init__(self, gd_sdk: GoodDataSdkWrapper) -> None:
        self.gd_sdk = gd_sdk
        self.workspace_id = st.session_state.workspace_id

    def _get_agent(self) -> GoodDataRAGSimple:
        return GoodDataRAGSimple(
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            workspace_id=self.workspace_id,
        )

    @timeit
    def render(self) -> None:
        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        catalog = get_gooddata_full_catalog(self.gd_sdk, self.workspace_id)
        if catalog.documents is None or len(catalog.documents) == 0:
            st.error(
                f"Picked GoodData workspace with ID={self.workspace_id} does not contain any objects. "
                "Please select another workspace."
            )
        else:
            st.markdown(f"""
- Dataset count: {len(catalog.ldm.ldm.datasets)}
- Date Dataset count: {len(catalog.ldm.ldm.date_instances)}
- Attribute count: {sum([len(d.attributes) for d in catalog.ldm.ldm.datasets])}
- Fact count: {sum([len(d.facts) for d in catalog.ldm.ldm.datasets])}
""")
            try:
                agent = self._get_agent()
                rag_retriever = load_vector_store(agent, catalog.documents)

                question = st.text_area("Ask GoodData a question:")

                columns = st.columns(2)
                with columns[0]:
                    if st.button("Submit Query", type="primary"):
                        output = agent.rag_chain_invoke(rag_retriever, question)
                        st.session_state.past.append(question)
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
                st.write("OpenAI unknown authentication error")
                st.write(e)


@st.cache_resource
def load_vector_store(_agent, rag_docs: list[str]):
    return _agent.get_rag_retriever(rag_docs)


@attr.s(auto_attribs=True, kw_only=True)
class GoodDataCatalog:
    documents: list[str]
    ldm: CatalogDeclarativeModel
    adm: CatalogDeclarativeAnalytics


@st.cache_data
@timeit
def get_gooddata_full_catalog(_gd_sdk: GoodDataSdkWrapper, workspace_id: str) -> GoodDataCatalog:
    ldm = _gd_sdk.sdk.catalog_workspace_content.get_declarative_ldm(workspace_id)
    adm = _gd_sdk.sdk.catalog_workspace_content.get_declarative_analytics_model(workspace_id)
    metrics = adm.analytics.metrics
    datasets = ldm.ldm.datasets
    date_datasets = ldm.ldm.date_instances
    documents = []
    for dataset in datasets:
        fact_list = "\n".join([f"Fact with ID {f.id} has title {f.title}" for f in dataset.facts])
        attribute_list = "\n".join([f"Attribute with ID {a.id} has title {a.title}" for a in dataset.attributes])
        document = f"""
        This document describes GoodData Phoenix dataset with ID={dataset.id} and title={dataset.title}.
        
        Dataset contains the following facts:
        {fact_list}
        
        Dataset contains the following attributes:
        {attribute_list}
        """
        # TODO - labels
        documents.append(document)
    for metric in metrics:
        documents.append(f"""
        This document describes GoodData Phoenix metric.
        
        Metric with ID {metric.id} has title {metric.title}.
        """)

    for date_dataset in date_datasets:
        granularities = "\n- ".join(date_dataset.granularities)
        documents.append(f"""
        This document describes GoodData Phoenix date dataset.
        
        Date dataset with ID {date_dataset.id} has title {date_dataset.title}.
        The dataset has the following granularities: 
        {granularities}
        """)
    # DEBUG documents
    if not os.path.isdir("tmp/RAG"):
        os.makedirs("tmp/RAG")
    for i, document in enumerate(documents):
        with open(f"tmp/RAG/{i}.txt", "w") as f:
            f.write(document)
    return GoodDataCatalog(documents=documents, ldm=ldm, adm=adm)

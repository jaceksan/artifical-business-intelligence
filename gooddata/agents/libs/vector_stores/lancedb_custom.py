from typing import Any, ClassVar, Collection
from langchain_core.pydantic_v1 import Field
from langchain_community.vectorstores import LanceDB
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

EXCLUDED_METADATA = ["vector"]


class LanceDBRetriever(VectorStoreRetriever):
    """
    LanceDB custom retriever is required because by default LanceDB returns vectors,
    which we do not want to include into LLM prompt as a context.
    """
    vectorstore: VectorStore
    """VectorStore to use for retrieval."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

        for doc in docs:
            for excl in EXCLUDED_METADATA:
                del doc.metadata[excl]

        return docs


class CustomLanceDB(LanceDB):
    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return LanceDBRetriever(vectorstore=self, **kwargs, tags=tags)

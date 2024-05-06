import os.path
from typing import Any, ClassVar, Collection
from langchain_core.pydantic_v1 import Field
from langchain_community.vectorstores import LanceDB
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from dotenv import load_dotenv
import lancedb
from langchain_openai import OpenAIEmbeddings
from time import time
from faker import Faker
from faker.providers import job

TABLE_NAME = "embeddings"
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

        # raise Exception(f"JACEK ERROR, vole!: {docs}")
        for doc in docs:
            for excl in EXCLUDED_METADATA:
                del doc.metadata[excl]
        return docs


class CustomLanceDB(LanceDB):
    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        # raise Exception(f"JACEK ERROR, vole 222222!: {query}")
        return LanceDBRetriever(vectorstore=self, **kwargs, tags=tags)


load_dotenv()
fake = Faker()
Faker.seed(0)
fake.add_provider(job)
documents = []
for i in range(1000):
    documents.append(
        Document(
            page_content=f"{fake.name()} works as a {fake.job()}.",
            metadata={"id": str(i)}
        )
    )

if os.path.isdir('./tmp') is False:
    os.makedirs('./tmp')
db_conn = lancedb.connect('./tmp/test.LANCENDB')

try:
    start_exists = time()
    print("Checking table exists")
    print(f"Existing table names: {db_conn.table_names()}")
    table = db_conn.open_table(TABLE_NAME)
    print(table.to_arrow().to_pandas())
    vector_store = CustomLanceDB(connection=db_conn, table_name=TABLE_NAME, vector_key="vector", embedding=OpenAIEmbeddings())
    print(f"Table exists check took {time() - start_exists} seconds")
except Exception as e:
    print(f"Table does not exist, create it from documents: {e}")
    start_not_exists = time()
    CustomLanceDB.from_documents(documents, connection=db_conn, table_name=TABLE_NAME, vector_key="vector", embedding=OpenAIEmbeddings())
    vector_store = CustomLanceDB(connection=db_conn, table_name=TABLE_NAME, embedding=OpenAIEmbeddings())
    print(f"Table does not exist, took {time() - start_not_exists} seconds")

start_search = time()
query = "Geologist"
# docs = vector_store.similarity_search(query)
docs = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
).invoke(query)

print(f"Search result: {docs}")
print(f"Search took {time() - start_search} seconds")

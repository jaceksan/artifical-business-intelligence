from enum import Enum
from operator import itemgetter
import lancedb
import duckdb
import attr

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from gooddata.agents.libs.vector_stores.lancedb_custom import CustomLanceDB
from langchain_community.vectorstores import DuckDB
from langchain.globals import set_debug
from langchain_core.documents import Document


from gooddata.agents.libs.gd_openai import GoodDataOpenAICommon
from gooddata.agents.libs.utils import timeit, debug_to_file

PRODUCT_NAME = "GoodData Cloud"
DEFAULT_MAX_SEARCH_RESULTS = 5
DB_URL_TEMPLATE = "tmp/{org_id}.{db_type}"


@attr.s(auto_attribs=True, kw_only=True)
class DBParams:
    name: str
    db_library: any
    langchain_library: any


class VectorDB(Enum):
    LANCEDB = DBParams(name="LanceDB", db_library=lancedb, langchain_library=CustomLanceDB)
    DUCKDB = DBParams(name="DuckDB", db_library=duckdb, langchain_library=DuckDB)


class GoodDataRAGCommon:
    def __init__(
        self,
        org_id: str,
        workspace_id: str,
        vector_db: VectorDB,
        openai_api_key: str,
        openai_organization: str,
        openai_model: str = "gpt-3.5-turbo-0613",
        temperature: int = 0,
        max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    ) -> None:
        self.gd_openai = GoodDataOpenAICommon(
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_organization=openai_organization,
            temperature=temperature,
            workspace_id=workspace_id,
            org_id=org_id,
        )
        self.max_search_results = max_search_results
        self.openai_chat_model = self.gd_openai.get_chat_llm_model()
        self.vector_db = vector_db
        # Add prefix to prevent issues with DBs which do not support table names starting with numbers
        self.vector_db_table_name = f"ws_{workspace_id}"
        self.openai_embedding = self.gd_openai.get_llm_embeddings()

    # TODO - other databases. QDrant, Milvus, Weaviate, PostgreSQL
    # TODO - add any indexes?
    def connect_to_db(self):
        return self.vector_db.value.db_library.connect(
            DB_URL_TEMPLATE.format(
                org_id=self.gd_openai.org_id,
                db_type=self.vector_db.name
            )
        )

    def connect_to_table(self, db_conn, table_name: str) -> bool:
        # TODO - try to load docs everytime, update only changed rows
        if self.vector_db == VectorDB.LANCEDB:
            open_func = db_conn.open_table
        elif self.vector_db == VectorDB.DUCKDB:
            open_func = db_conn.table
        else:
            raise NotImplementedError(f"Database {self.vector_db.name} is not supported")
        try:
            print(f"Opening table {table_name}")
            open_func(table_name)
            return True
        except Exception as e:
            # TODO - better not that broad Exception
            print(f"Opening table {table_name} failed: {e}")
            return False

    @staticmethod
    def debug_documents(documents: list[Document]):
        docs_str = "\n".join(
            [
                f"""Metadata:\n{str(r.metadata)}\n\nContent:\n{r.page_content}"""
                for r in documents
            ]
        )
        debug_to_file("rag_docs.txt", docs_str)

    def init_vector_store_duckdb(self, documents: list[Document], db_conn):
        _, is_new = self.connect_to_table(db_conn, self.vector_db_table_name)
        if is_new:
            return self.vector_db.value.langchain_library.from_documents(
                documents,
                connection=db_conn,
                table_name=self.vector_db_table_name,
                embedding=self.openai_embedding,
                vector_key="embedding",
            )
        else:
            return self.vector_db.value.langchain_library(
                connection=db_conn,
                table_name=self.vector_db_table_name,
                embedding=self.openai_embedding,
                vector_key="embedding",
            )

    def init_vector_store_lancedb(self, documents: list[Document], db_conn):
        table_exists = self.connect_to_table(db_conn, self.vector_db_table_name)
        if table_exists:
            return self.vector_db.value.langchain_library(
                embedding=self.openai_embedding,
                table_name=self.vector_db_table_name,
                connection=db_conn,
            )
        else:
            return self.vector_db.value.langchain_library.from_documents(
                documents=documents,
                embedding=self.openai_embedding,
                table_name=self.vector_db_table_name,
                connection=db_conn,
            )

    @timeit
    def init_vector_store(self, documents: list[Document]):
        self.debug_documents(documents)
        db_conn = self.connect_to_db()
        if self.vector_db == VectorDB.DUCKDB:
            return self.init_vector_store_duckdb(documents, db_conn)
        elif self.vector_db == VectorDB.LANCEDB:
            return self.init_vector_store_lancedb(documents, db_conn)
        else:
            raise NotImplementedError(f"Database {self.vector_db.name} is not supported")

    def drop_table(self, db_conn):
        if self.vector_db == VectorDB.DUCKDB:
            db_conn.sql(f"""DROP TABLE "{self.vector_db_table_name}";""")
        elif self.vector_db == VectorDB.LANCEDB:
            db_conn.drop_table(self.vector_db_table_name)
        else:
            raise NotImplementedError(f"Database {self.vector_db.name} is not supported")

    @timeit
    def get_rag_retriever(self, vector_store):
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.max_search_results}
        )

    def similarity_search(self, vector_store, query: str):
        # TODO - use direct select from the table? How to embed correct vectors?
        #  Would allow to use filters before doing similarity search!
        #  Note: some DBs do not support filtering on metadata columns yet, e.g. DuckDB
        #  Some support it only with LlamaIndex
        # TODO: using retriever here to accept custom Retriever implementations for both RAG and pure similarity search
        return self.get_rag_retriever(vector_store).get_relevant_documents(query)
        # return vector_store.similarity_search(query, k=self.max_search_results)

    @staticmethod
    def _combine_documents(docs, document_prompt, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)


class GoodDataRAGSimple(GoodDataRAGCommon):
    @timeit
    def get_rag_chain(
        self,
        rag_retriever,
        answer_prompt: str,
    ):
        debug_to_file("answer_prompt.txt", answer_prompt)
        return (
            {"context": rag_retriever, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(answer_prompt)
            | self.openai_chat_model
            | StrOutputParser()
        )

    @timeit
    def rag_chain_invoke(
        self,
        rag_retriever,
        question: str,
        answer_prompt: str,
    ):
        set_debug(True)
        return self.get_rag_chain(rag_retriever, answer_prompt).invoke(question)


class GoodDataRAGHistory(GoodDataRAGCommon):
    @timeit
    def get_rag_chain_with_history(self, rag_retriever, condense_question_prompt, chat_template):
        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(chat_history=lambda x: get_buffer_string(x["chat_history"]))
            | ChatPromptTemplate.from_template(condense_question_prompt)
            | self.openai_chat_model
            | StrOutputParser(),
        )
        _context = {
            "context": itemgetter("standalone_question") | rag_retriever | self._combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        conversational_qa_chain = (
            _inputs
            | _context
            | ChatPromptTemplate.from_template(chat_template)
            | self.openai_chat_model
        )
        return conversational_qa_chain

    @timeit
    def rag_chain_with_history_invoke(self, rag_retriever, question: str, chat_history: list[tuple[str, str]]):
        chat_history_ai = [(HumanMessage(content=human), AIMessage(content=ai)) for human, ai in chat_history]
        self.get_rag_chain_with_history(rag_retriever).invoke(
            {
                "question": question,
                "chat_history": chat_history_ai,
                # "language": self.rag_language,
            }
        )

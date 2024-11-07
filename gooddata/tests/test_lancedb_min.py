import lancedb
from dotenv import load_dotenv
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

TABLE_NAME = "test"


# For OpenAIEmbeddings
load_dotenv()
documents = [Document(page_content="Test document.", metadata={"id": "1", "title": "Test document"})]

db_conn = lancedb.connect("test.LANCENDB")
LanceDB.from_documents(
    documents, connection=db_conn, table_name=TABLE_NAME, vector_key="vector", embedding=OpenAIEmbeddings()
)
table = db_conn.open_table(TABLE_NAME)

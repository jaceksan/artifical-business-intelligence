import os.path

from dotenv import load_dotenv
import duckdb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DuckDB
from langchain_core.documents import Document
from time import time
from faker import Faker
from faker.providers import job

TABLE_NAME = "embeddings"

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
db_conn = duckdb.connect('./tmp/test.DUCKDB')

try:
    start_exists = time()
    print("Checking table exists")
    table = db_conn.table(TABLE_NAME)
    table.show()
    vector_store = DuckDB(connection=db_conn, table_name=TABLE_NAME, embedding=OpenAIEmbeddings(), vector_key="embedding")
    print(f"Table exists check took {time() - start_exists} seconds")
except Exception as e:
    start_not_exists = time()
    print(f"Table does not exist, create it from documents")
    vector_store = DuckDB.from_documents(documents, connection=db_conn, table_name=TABLE_NAME, embedding=OpenAIEmbeddings(), vector_key="embedding")
    print(f"Table does not exist, took {time() - start_not_exists} seconds")

start_search = time()
query = "Geologist"
# docs = vector_store.similarity_search(query)
docs = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
).get_relevant_documents(query)

print(f"Search result: {docs}")
print(f"Search took {time() - start_search} seconds")

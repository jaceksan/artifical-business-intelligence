from typing import Optional
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory

from gooddata.agents.libs.gd_openai import GoodDataOpenAICommon
from gooddata.agents.libs.utils import timeit

_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """
You are expert for GoodData Phoenix.
You answer only questions related to GoodData Phoenix.
You answer "I can answer only questions related to GoodData" if you get question not related to GoodData Phoenix. 

Answer the question based only on the following context:
{context}

Question: 
{question}
"""
# Answer in the following language: {language}
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


class GoodDataRAGCommon:
    def __init__(
        self,
        openai_model: str = "gpt-3.5-turbo-0613",
        openai_api_key: Optional[str] = None,
        openai_organization: Optional[str] = None,
        temperature: int = 0,
        workspace_id: str = None,
    ) -> None:
        self.gd_openai = GoodDataOpenAICommon(
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_organization=openai_organization,
            temperature=temperature,
            workspace_id=workspace_id,
        )
        self.openai_chat_model = self.gd_openai.get_chat_llm_model()

    @staticmethod
    def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    @timeit
    def get_rag_retriever(self, rag_docs: list[str]):
        vectorstore = FAISS.from_texts(
            rag_docs, embedding=OpenAIEmbeddings()
        )
        return vectorstore.as_retriever()


class GoodDataRAGSimple(GoodDataRAGCommon):

    @timeit
    def get_rag_chain(self, rag_retriever):
        return (
            {"context": rag_retriever, "question": RunnablePassthrough()}
            | ANSWER_PROMPT
            | self.openai_chat_model
            | StrOutputParser()
        )

    @timeit
    def rag_chain_invoke(self, rag_retriever, question: str):
        return self.get_rag_chain(rag_retriever).invoke(question)


class GoodDataRAGHistory(GoodDataRAGCommon):
    @timeit
    def get_rag_chain_with_history(self, rag_retriever):
        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | self.openai_chat_model
            | StrOutputParser(),
        )
        _context = {
            "context": itemgetter("standalone_question") | rag_retriever | self._combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
        return conversational_qa_chain

    @timeit
    def rag_chain_with_history_invoke(self, rag_retriever, question: str, chat_history: list[tuple[str, str]]):
        chat_history_ai = [
            (HumanMessage(content=human), AIMessage(content=ai))
            for human, ai in chat_history
        ]
        self.get_rag_chain_with_history(rag_retriever).invoke({
            "question": question,
            "chat_history": chat_history_ai,
            # "language": self.rag_language,
        })

    @timeit
    def reset_rag_chain_context(self, docs: list[str]):
        return self.get_rag_retriever(docs)


class GoodDataRAGHistoryMemory(GoodDataRAGCommon):
    def __init__(
        self,
        openai_model: str = "gpt-3.5-turbo-0613",
        openai_api_key: Optional[str] = None,
        openai_organization: Optional[str] = None,
        temperature: int = 0,
        workspace_id: str = None,
    ) -> None:
        super().__init__(
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_organization=openai_organization,
            temperature=temperature,
            workspace_id=workspace_id,
        )
        self.rag_conversational_memory = self.conversational_memory

    @timeit
    @property
    def conversational_memory(self):
        return ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )

    @timeit
    def rag_chain_with_memory(self, rag_retriever):
        # First we add a step to load memory
        # This adds a "memory" key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.rag_conversational_memory.load_memory_variables) | itemgetter("history"),
        )
        # Now we calculate the standalone question
        standalone_question = {
            "standalone_question": {
               "question": lambda x: x["question"],
               "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | CONDENSE_QUESTION_PROMPT
            | self.openai_chat_model
            | StrOutputParser(),
        }
        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | rag_retriever,
            "question": lambda x: x["standalone_question"],
        }
        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: self._combine_documents(x["docs"]),
            "question": itemgetter("question"),
            # "language": itemgetter("language"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | self.openai_chat_model,
            "docs": itemgetter("docs"),
        }
        # And now we put it all together!
        final_chain = loaded_memory | standalone_question | retrieved_documents | answer
        return final_chain

    @timeit
    def rag_chain_with_memory_invoke(self, question: str):
        inputs = {"question": question}
        result = self.rag_chain_with_memory.invoke({
            "question": question,
            # "language": self.rag_language,
        })
        answer = result["answer"].content
        self.rag_conversational_memory.save_context(inputs, {"answer": answer})
        return answer

    @timeit
    def rag_chain_with_memory_reset(self):
        self.rag_conversational_memory.clear()

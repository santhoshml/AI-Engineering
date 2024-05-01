from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from utils import *
import os
import getpass
from langchain.globals import set_debug


class RAGMeta10K:
    
    def __init__(self) -> None:       
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
        
        # set_debug(True)
        
        self.UtilsObject = Utils()
        self.rag_prompt_template = self.UtilsObject.init_prompt()
        self.UtilsObject.split_into_chunks()
        self.qdrant_retriever = self.UtilsObject.get_vector_store().as_retriever()

    def ask_question(self, question: str):
        retrieval_augmented_qa_chain = (
            {"context": itemgetter("question") | self.qdrant_retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": self.rag_prompt_template | self.UtilsObject.get_llm_model(), "context": itemgetter("context")}
        )

        response = retrieval_augmented_qa_chain.invoke({"question" : question})
        print("response :"+ response["response"].content)
        # print("*******")
        # for context in response["context"]:
        #     print("Context:")
        #     print(context)
        #     print("----")


ragObject = RAGMeta10K()
ragObject.ask_question("Who are Directors?")    #works
ragObject.ask_question("what is the value of Total cash and cash equivalents ?") #works

#ragObject.ask_question("What is the value of total cash and cash equivalents?")
# ragObject.ask_question("Who are the is the Board Chair and Chief Executive Officer ?")
#ragObject.ask_question("Who is the Board Chair and Chief Executive Officer ?")


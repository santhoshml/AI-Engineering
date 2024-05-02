from dotenv import load_dotenv
load_dotenv() 

from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from utils import *
import os
import getpass
from langchain.globals import set_debug
import chainlit as cl 
from langchain_openai import ChatOpenAI, OpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


class RAGMeta10K:
    
    def __init__(self) -> None:       
        # os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
        
        # set_debug(True)
        
        self.UtilsObject = Utils()
        self.rag_prompt_template = self.UtilsObject.init_prompt()
        self.UtilsObject.split_into_chunks()
        # normail retriever
        self.qdrant_retriever = self.UtilsObject.get_vector_store().as_retriever()
        
        # MultiQuery Retriever
        # self.mqr_retriever = MultiQueryRetriever.from_llm(
        #     retriever=self.qdrant_retriever, llm=ChatOpenAI(temperature=0)
        # )
        
        #Contexttual Compression
        # gives correct answer for board question but messup on Cash question
        compressor = LLMChainExtractor.from_llm(OpenAI(temperature=0))
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.qdrant_retriever
        )
        

    def ask_question(self, question: str):
        retrieval_augmented_qa_chain = (
            {"context": itemgetter("question") | self.compression_retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": self.rag_prompt_template | self.UtilsObject.get_llm_model(), "context": itemgetter("context")}
        )

        response = retrieval_augmented_qa_chain.invoke({"question" : question})
        print("response :"+ response["response"].content)
        return response["response"].content
        # print("*******")
        # for context in response["context"]:
        #     print("Context:")
        #     print(context)
        #     print("----")


# ragObject = RAGMeta10K()
# ragObject.ask_question("Who are Directors?")    #works
# ragObject.ask_question("what is the value of Total cash and cash equivalents ?") #works


@cl.on_chat_start
async def start_chat():
    ragObject = RAGMeta10K()
    # ragObject.UtilsObject.generate_test_set()
    cl.user_session.set("ragObject", ragObject)

@cl.on_message
async def main(message: cl.Message):
    ragObject = cl.user_session.get("ragObject")
    answer=ragObject.ask_question(message.content)
    await cl.Message(content=answer).send()
                           
        
#ragObject.ask_question("Who are Directors?")    #works
#ragObject.ask_question("what is the value of Total cash and cash equivalents ?") #works

#ragObject.ask_question("What is the value of total cash and cash equivalents?")
# ragObject.ask_question("Who are the is the Board Chair and Chief Executive Officer ?")
#ragObject.ask_question("Who is the Board Chair and Chief Executive Officer ?")


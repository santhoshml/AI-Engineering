from langchain_openai import ChatOpenAI
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from ragas.testset.generator import TestsetGenerator


class Utils:
    def __init__(
        self,
        llm_name: str = "gpt-3.5-turbo",
        pdf_name: str = "meta-10k.pdf",
        embedding_model: str = "text-embedding-3-small",
        generator_llm: str = "gpt-3.5-turbo-16k",
        critic_llm: str= "gpt-4-turbo"
    ) -> None:
        self.openai_chat_model = ChatOpenAI(model=llm_name)
        self.enc = tiktoken.encoding_for_model(llm_name)
        self.docs = PyMuPDFLoader(pdf_name).load()
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.generator_llm = ChatOpenAI(model=generator_llm)
        self.critic_llm = ChatOpenAI(model=critic_llm)
        self.test_generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            OpenAIEmbeddings()
        )

    
    #semanticTextSplitter
    #tokenRTextSplitter
    def split_into_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=self.tiktoken_len
        )
        self.split_chunks = text_splitter.split_documents(self.docs)
        
        # semantic splitter
        # text_splitter = SemanticChunker(OpenAIEmbeddings())
        # self.split_chunks = text_splitter.split_documents([self.docs])
        return self.split_chunks

    def get_llm_model(self):
        return self.openai_chat_model

    def init_prompt(self) -> ChatPromptTemplate:
        RAG_PROMPT = """
            ###Instruction###:
            Answer the question based only on the following context. If you cannot answer the question with the context, please respond with "I don't know":
            
            CONTEXT:
            {context}

            QUERY:
            {question}            
            """
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        return rag_prompt

    def tiktoken_len(self, text) -> int:
        self.tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
            text,
        )
        return len(self.tokens)

    def get_vector_store(self):
        self.qdrant_vectorstore = Qdrant.from_documents(
            self.split_chunks,
            self.embedding_model,
            location=":memory:",
            collection_name="meta-10k",
        )
        return self.qdrant_vectorstore
    
    def generate_test_set(self)-> None:
        text_splitter_eval = RecursiveCharacterTextSplitter(
            chunk_size = 600,
            chunk_overlap = 50
        )
        eval_documents = text_splitter_eval.split_documents(self.docs)
        distributions = {
            "simple": 0.5,
            "multi_context": 0.4,
            "reasoning": 0.1
        }
        testset = self.test_generator.generate_with_langchain_docs(eval_documents, 20, distributions, is_async = False)
        print("santhosh:"+len(testset.to_pandas()))

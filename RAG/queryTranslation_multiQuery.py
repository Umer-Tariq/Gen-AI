from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.schema import Document
import os
import bs4
import warnings
import json
from operator import itemgetter

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# LOAD THE DOCUMENTS #

loader = WebBaseLoader(
    web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(
            class_ = ('post-content', 'post-title', 'post-header')
        )
    )
)

# docs is a list of relevant docs

docs = loader.load()

# SPLIT THE DOCUMENTS AND STORE THEM IN A VECTOR DB #

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
spilts = text_splitter.split_documents(documents=docs)

vectorstore = Chroma.from_documents(documents=spilts, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

retriever = vectorstore.as_retriever(searhc_kwargs = {"k" : 1})

# MAKE MULTIPLE VERSIONS OF THE USER QUERY #

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

multiple_query_prompt = ChatPromptTemplate.from_template(template)

llm = ChatMistralAI(temperature=0)

# dictionary not passed in this chain as their is only one input variable

generate_queries = (
    multiple_query_prompt
    | llm
    | StrOutputParser()
    | (lambda x : x.split('\n'))
)

queries = generate_queries.invoke("What are LLMs?")
# queries is a list of queries for which we will retrieve relevant docs

# RETRIEVE RELEVANT DOCUMENTS FOR EACH QUERY AND COMBINE THEM TO GET A CONTEXT #

def get_unique_union(list_of_documents):
    flattened_docs = [dumps(doc) for documents in list_of_documents for doc in documents]
    unique_docs = list(set(flattened_docs))
    unique_docs = [loads(doc) for doc in unique_docs]

    return unique_docs

# retriever.map() extracts relevant docs for each query from the list of queries. As a result we get a list of lists

retrieval_chain = generate_queries | retriever.map() | get_unique_union

# gets a list of docs for each query. so we have a list of list of relevant docs. from this list we find the unique ones and take their union

relevant_docs = retrieval_chain.invoke("What are LLMs?")

# USE THE RETRIEVED DOCUMENTS IN RAG #

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# the retriever chain is passed as the input to context. itemgetter gets the question from the input dictionary. A dict is passed to the chain first up as multiple input variables are present in the prompt.

rag_chain = (
    {"context" : retrieval_chain, "question" : itemgetter('question')}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke({"question" : "What are LLMs"})

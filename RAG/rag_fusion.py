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


def rrf(list_of_documents):
    rank_scores = {}
    result = []
    k = 60
    for docs in list_of_documents:
        for rank, doc in enumerate(docs):
            doc_dump = dumps(doc)
            if doc_dump not in rank_scores:
                rank_scores[doc_dump] = 0
            rank_scores[doc_dump] += (1/rank + k)
    
    sorted_docs = sorted(rank_scores.items(), key=lambda x : x[1], reverse= True)
    for doc, rank in sorted_docs:
        result.append((loads(doc), rank))

    return result






template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

prompt = ChatPromptTemplate.from_template(template=template)

llm = ChatMistralAI(temperature=0)

prompt_chain = (
    prompt 
    | llm
    | StrOutputParser()
    | (lambda x : x.split('\n'))
)


loader = WebBaseLoader(
    web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
    bs_kwargs= dict(
        parse_only = bs4.SoupStrainer(
            class_ = ('post-content', 'post-title', 'post-header')
        )
    )
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(documents=docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

retriever = vectorstore.as_retriever(search_kwargs = {"k" : 2})

retrieval_chain = prompt_chain | retriever.map() | rrf

template = """Answer the following question based on this context:

{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context" : retrieval_chain,
     "question" : itemgetter('question')}
    | prompt
    | llm
    | StrOutputParser()
)

output = rag_chain.invoke({"question" : "What are LLMs"})

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.schema import Document
from langchain.schema.runnable import RunnableMap
import os
import bs4
import warnings
import json
from operator import itemgetter



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

# GENERATE A MORE ABSTRACT QUESTION FROM THE ORIGINAL QUESTION USING THE FEW SHOT MESSAGE TEMPLATE IN WHICH WE PASS A SET OF EXAMPLE AS WELL TO THE LLM SO THAT IT USES THEM TO ANSWER ACCORDINGLY IN THE REQUIRED FORMAT #

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ('human', '{input}'),
        ('ai', '{output}'),
    ]
)


# FEW SHOT PROMPT NEEDS AN EXAMPLE LIST(A LIST OF DICTS EACH HAVING THE INPUT AND THE VALID OUTPUT) AND AN EXAMPLE PROMPT.

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples = examples, example_prompt=example_prompt
)

# THIS FEW SHOT PROMPT WILL BE USED IN MAKING THE FINAL PROMPT TO GET THE STEP BACK QUESTION. THE FINAL PROMPT WILL BE MAKE USING THE PROMPT FORM MESSAGES FUNCTION #

# MAKE THE PROMPT AND 

stepback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """"You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",),
        few_shot_prompt,
        ("human", "{question}")
    ]
)

# GET THE STEPBACK QUESTION #

llm = ChatMistralAI(temperature=0)

stepback_question_chain = (
    stepback_prompt
    | llm
    | StrOutputParser()
)




# USE THE STEPBACK QUESTION TO RETRIEVE THE BROADER CONTEXT. USE THE ORIGINAL QUESTION TO GET IT'S CONTEXT AND PASS THE QUESTION AS WELL #

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{normal_context}
{step_back_context}

Original Question: {question} """

question = "What are LLMs"

# RunnableMap is used because we want to use the chaining opeartor and that cannot be run iside a dictionary without RunnableMap #

# Normal string cannot be chained #

prompt = ChatPromptTemplate.from_template(response_prompt_template)

rag_chain = (
    RunnableMap(
        {"normal_context" : itemgetter("question") | retriever,
        "step_back_context" : itemgetter("question") | stepback_question_chain | retriever,
        "question" : itemgetter("question")}
    )
    | prompt
    | llm 
    | StrOutputParser()
)

answer = rag_chain.invoke({"question" : question})

print(answer)
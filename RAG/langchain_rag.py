from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import bs4
import torch
import os
import requests
import json
from pydantic import BaseModel


""" 
class DistilBERTEmbeddings(Embeddings):
    def __init__(self, model_name = "distilbert-base-uncased"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            tokens = self.tokenizer(text=text, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**tokens)
                embedding = model_output.last_hidden_state.mean(dim = 1).squeeze().tolist()
                embeddings.append(embedding)

        return embeddings

    def embed_query(self, query):
        tokens = self.tokenizer(text = query, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            model_output = self.model(**tokens)
            embedding = model_output.last_hidden_state.mean(dim = 1).squeeze().tolist()
        
        return embedding
"""
from langchain.llms.base import LLM
from pydantic import BaseModel
import requests

class MistralLLM(LLM, BaseModel):
    api_endpoint: str
    api_key: str

    @property
    def _llm_type(self):
        return 'mistral'

    def _call(self, prompt, stop = None):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            "Content-Type": "application/json"
        }
        
        # Correct payload to match the expected format for chat completions
        payload = {
            'model': 'codestral-latest',
            'messages': [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(url=self.api_endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            generated_content = response_data['choices'][0]['message']['content']
            return generated_content
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
            
# SET ENVIRONMENT VARIABLES #

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["USER_AGENT"] = "my-langchain-app/1.0"


# LOAD API KEY FORM OS #

mistral_api_key = os.getenv('MISTRAL_API_KEY')

# LOAD DOCUMENTS #

loader = WebBaseLoader(
    web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(
            class_ = ("post-title", "post-content", "post-header") 
        )
    )
)

docs = loader.load()

# MAKE CHUNCKS OF DOCUMENT #

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(documents=docs)

# CREATE EMBEDDINGS AND STORE IN VECTOR DB #

vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
retriever = vectorstore.as_retriever(search_kwargs = {'k' : 3})

# RETRIEVE RELEVANT DOCUMENT AS CONTEXT #

#relevant_documents = retriever.invoke(input="What is task decomposition")

# MAKE A PROMPT USING PROMPT TEMPLATE #

prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Please answer the question using the context.\n\nQuestion: {query}\n\n Context: {context}"
)

# INITIALIZE AN LLM - MISTRAL #

llm = MistralLLM(api_endpoint="https://api.mistral.ai/v1/chat/completions", api_key=mistral_api_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# MAKE A CHAIN #
rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


rag_chain.invoke("What is Task Decomposition?")
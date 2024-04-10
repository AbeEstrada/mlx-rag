from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

document = PyPDFLoader("./2312.03103.pdf").load()
documents = RecursiveCharacterTextSplitter().split_documents(document)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

llm = MLXPipeline.from_model_id(
    "mlx-community/gemma-1.1-7b-it-4bit",
    pipeline_kwargs={"max_tokens": 2048, "temp": 0.9},
)


template = """
You are an assistant for question-answering tasks.
Use the provided context only to write an answer:

<context>
{context}
</context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)

response = chain.invoke(
    {"input": "Give me a summary of this paper"})

print(response["answer"])

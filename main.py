from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

document = PyPDFLoader("./2312.03103.pdf").load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
).split_documents(document)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = MLXPipeline.from_model_id(
    "mlx-community/gemma-3n-E2B-it-lm-4bit",
    pipeline_kwargs={"max_tokens": 2000, "temp": 1.0},
)

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three paragraphs maximum and keep the answer concise.
Question: {input}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)

while True:
    question = input("\nEnter your question (or 'quit' to exit): ")
    if question.lower() == "quit":
        break
    response = chain.invoke({"input": question})
    print("\nAnswer:", response["answer"])

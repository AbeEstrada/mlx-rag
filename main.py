from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

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

model, tokenizer = load("mlx-community/gemma-3n-E2B-it-lm-4bit")

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three paragraphs maximum and keep the answer concise.
Question: {input}
Context: {context}
Answer:"""

prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | RunnableLambda(lambda p: generate(model, tokenizer, p.text, max_tokens=2000, sampler=make_sampler(temp=0.2)))
)

while True:
    question = input("\nEnter your question (or 'quit' to exit): ")
    if question.lower() == "quit":
        break
    print("\nAnswer:", chain.invoke(question))

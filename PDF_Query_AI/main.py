from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma 
from langchain.chains import ChatVectorDBChain
from langchain.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings

llm = Ollama(
    model="zephyr", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

loader = PyPDFLoader("paper.pdf")
pages = loader.load_and_split()

embeddings = OllamaEmbeddings(model="zephyr")

vector_db = Chroma.from_documents(pages, embedding=embeddings , persist_directory='.')
vector_db.persist()


qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())


question = "what is about the article?"
result = qa_chain({"query": question})



question = "Make a summary about the BClean using simple words"
result = qa_chain({"query": question})


context = []
context.append({"question": result['query'], "answer": result["result"]})




#context = f"Question: {result['query']} | Answer: {result['result']}"




question = "Make a summary in portuguese"
result = qa_chain({"context": context , "query": question})



context.append({"question": result['query'], "answer": result["result"]})

result = qa_chain({"context": context , "query": question})


qa_chain2 = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=vector_db.as_retriever(),  return_source_documents=True)



question = "Make a summary about the BClean using simple words, show as a solution"
result = qa_chain2({"question": question})

context = f"Question: {result['question']} | Answer: {result['answer']}"
result = qa_chain2({"context": context , "question": "Make a longer summary"})


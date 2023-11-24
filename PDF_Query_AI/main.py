from langchain.vectorstores import Chroma 
from langchain.chains import ChatVectorDBChain
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings

loader = PyPDFLoader("paper.pdf")
pages = loader.load_and_split()
print('Lenght Pages List: ', len(pages))

print('Pages 0 Content: ')
print(pages[0].page_content)
embeddings = OllamaEmbeddings(model="mistrallite")

vector_db = Chroma.from_documents(pages, embedding=embeddings , persist_directory='.')
vector_db.persist()
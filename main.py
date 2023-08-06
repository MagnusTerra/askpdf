from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

loader = UnstructuredPDFLoader("C.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

while True:
    query = str(input("Escribe tu pregunta: "))
    if query == "salir":
        break
    
    docs = docsearch.get_relevant_documents(query)

    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    output = chain.run(input_documents=docs, question=query)
    print(output)

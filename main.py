from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


LLM = None
VECTOR_STORE = None
VECTOR_STORE_DIR = Path(__file__).parent / "Vectorstore"

def initialize_components():
  
  global LLM, VECTOR_STORE
  
  if LLM is None:
    LLM = ChatGoogleGenerativeAI(
                                  model="gemini-3.8-flash",
                                  temperature=0
                                )



  if VECTOR_STORE is None:

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


    VECTOR_STORE = Chroma(
                            collection_name="example_collection",
                            embedding_function=embeddings,
                            persist_directory=VECTOR_STORE_DIR,  # Where to save data locally, remove if not necessary
                          )



def process_pdf(file):

  global LLM, VECTOR_STORE
  yield "Initializing Components"
  print( "Initializing Components")
  initialize_components()


  yield "Resetting Vector Store"
  print("Resetting Vector Store")
  VECTOR_STORE.reset_collection()

  yield "Loading Data from PDFs"
  print("Loading Data from PDFs")
  loader = PyPDFLoader(file)
  data = loader.load()
  yield "Splitting Documents"
  print("Splitting docs")
  

  # Initialize the text splitter
  text_splitter = RecursiveCharacterTextSplitter(
                                                      separators=[
                                                                        "\n\n",
                                                                        "\n",
                                                                        " ",
                                                                        ".",
                                                                        ",",
                                                                    ],
                                                      chunk_size=100,
                                                      chunk_overlap=20,
                                                      
                                                  )

  chunks =  text_splitter.split_documents(data)
  from uuid import uuid4

  uuids= [str(uuid4()) for _ in range(len(chunks))]
  VECTOR_STORE.add_documents(
                              chunks, ids=uuids
                            )
  yield "Done Adding Docs to vector db"



prompt_tm = """ 
            Answer the Question Only based on the context provided !
            No Preamble
            context: {context}
            Question : {Question}
"""

prompt = ChatPromptTemplate.from_template(prompt_tm)



def generate_answer(query):

    
    global LLM, VECTOR_STORE, prompt
    if VECTOR_STORE is None:
        raise RuntimeError("Vector database isn't initialized!")

    retriever = VECTOR_STORE.as_retriever()
    documents = retriever.invoke(query)

    content = "\n\n".join(
        doc.page_content for doc in documents
    )

    chain = prompt | LLM | StrOutputParser()

    answer = chain.invoke({
        "context": content,
        "Question": query
    })

    sources = list(set(
        doc.metadata.get("source", "Unknown")
        for doc in documents
    ))

    return answer, sources



def main():
    file_path = Path(
        r"D:\Resume_cv\DS\Trijesh_Kondapuram_Resume.pdf"
    )

    for status in process_pdf(file_path):
        print(status)

    generate_answer("what skills does he have?")
if __name__ == "__main__":
    main()
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

import os

class rag:
    def __init__(self, doclist=None, openai_key=None):
        if doclist==None or openai_key==None:
            raise ValueError("Please check if you have specified the doclist or openai_key before using it ") 
        
        self.doclist = doclist
        self.openai_key=openai_key
        self.docs = []
        self.chunks=[]
        self.vectorstore=None
        self.agent=None


    # Load documents
    def load_doc(self):
        docs=[]
        for d in self.doclist: 
            if os.path.exists(d):
                extension=d.split(".")[-1]
                #print(extension)
                match extension:
                    case "pdf":
                        pages = PyPDFLoader(d).load()
                        # for page in pages:
                        #     #print(f"Page: {page.metadata["page"]}: {page.page_content[:100]} ...")
                        #     self.text += page.page_content
                        docs +=pages
                    case "txt":
                        pages =TextLoader(d).load()
                        #print(page)
                        # print(f".txt: {page[0].page_content} ...")
                        # print("step1")
                        # self.text += page[0].page_content
                        # print("step2")
                        docs +=pages
                    case _:
                        print(f"Doesn't support {d} to be loaded, so skipping it.")
                 
            else:
                print(f"Your file {d} was not available.")
                continue  
            self.docs = docs      
        return self.docs

    def chunk(self):
        if self.docs!=[]:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            self.chunks = splitter.split_documents(self.docs)
            i=1
            # for chunk in chunks:
            #     print(f"**** chunk {i} ****")
            #     print(chunk.page_content[500:-1])
            #     i+=1

            return self.chunks
        else:
            raise ValueError("docs is empty, cannot be chunked it")
        
    def embed_index(self):
        if self.chunks!=[]:
            emb = OpenAIEmbeddings(openai_api_key=self.openai_key)
            #print(type(emb))
            # print(f"emb= {emb}")
            # print(f"emb.__dict__ = {emb.__dict__}")
            # vec = emb.embed_query("Hello world")
            # print(f"len(vec)={len(vec)}")       # Dimension of embedding (e.g., 1536)
            # print(f"vec[:5]={vec[:5]}")        # First 5 numbers

            # print("*"*150)

            vectorstore = FAISS.from_documents(self.chunks, emb)
            #print(type(vectorstore))
            # Number of vectors
            # print(f"vectorstore.index.ntotal= {vectorstore.index.ntotal}")
            # # Inspect stored document metadata
            # # print(f"vectorstore.docstore._dict= {vectorstore.docstore._dict}")   # Dictionary of stored documents

            # print("*"*150)
            # # Get the FAISS document store dictionary
            # doc_dict = vectorstore.docstore._dict

            # # Loop through each ID and document
            # for vector_id, doc in doc_dict.items():
            #     print(f"Vector ID: {vector_id}")
            #     print(f"Chunk text: {doc.page_content[:200]}...")  # first 200 characters
            #     print(f"Metadata: {doc.metadata}")
            #     print("-" * 50)        
            self.vectorstore=vectorstore
            return self.vectorstore
        else:
            raise ValueError("chunks is empty, cannot be embed and indexing")

    def rag_chain(self):

        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized, Please embed your docs first")
        
        #By default, if you don’t pass model or api_key, it will use OPENAI_API_KEY from your environment variables and a default model like "gpt-3.5-turbo-instruct" (depending on your LangChain version).
        llm = OpenAI(temperature=0) #temperature=0 → makes the output deterministic (model will try to give the same answer for the same input every time)
        self.agent = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", #"stuff" → put all retrieved chunks together into one big prompt. Simple, but can run into token limits if many chunks
                                # "map_reduce", "refine", etc. → more complex multi-step processing
            retriever=self.vectorstore.as_retriever(search_kwargs={"k":10}) #retrieve 3 most relevant chunks for each question
        )
        return self.agent
    
    def ask_me(self,question):
        
        print(f"Rag+LLM Response: {self.agent.invoke(question)}")
        print()

    



    


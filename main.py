from dotenv import load_dotenv
import os
from core.rag import rag

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
doclist = ['.\\Docs\\Savina.Resume.SrQualityEngineer-BigData.pdf','.\\Docs\\portfolio_notes.txt']

try:
    #load_doc → chunk → embed → rag_chain → run query
    rag_pipeline=rag(doclist, openai_key)

    rag_pipeline.load_doc()
    rag_pipeline.chunk()
    rag_pipeline.embed_index()
    rag_pipeline.rag_chain()

    rag_pipeline.ask_me("What is the name on the resume?")
    rag_pipeline.ask_me("Summerize her python skills")
    rag_pipeline.ask_me("Summerize her strenth")
    rag_pipeline.ask_me("What is included in the portfolio?")
    rag_pipeline.ask_me("does Savina like watching movies?")
    rag_pipeline.ask_me("What job do you recommand Savina to apply?")
    rag_pipeline.ask_me("What is her AI experience?")
    

except Exception as e:
    print(f"{e}")







from dotenv import load_dotenv
load_dotenv()  # ✅ Load .env variables into environment

from app.ingest import ingest_document

ingest_document(r"C:\Users\hp\hackrx-api\Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf")  # Replace with your actual PDF file name

# common.py
import datetime
import os
import modal

# Create the Modal app
app = modal.App(name="civil-engineer-assistant")

# Define the image with necessary dependencies
pip_image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1-mesa-glx",  # For PyMuPDF
        "poppler-utils",    # For tabula-py
        "tesseract-ocr",    # For pytesseract
    )
    .pip_install(
        [
            "openai",
            "PyPDF2",
            "reportlab",
            "pymupdf",  # PyMuPDF
            "sentence-transformers",
            "faiss-cpu",
            "numpy",
            "pandas",
            "tabula-py",
            "pytesseract",
            "openpyxl",
            "python-dotenv",
            "torch",
            "nltk",
            "tqdm",
            "spacy",
            "argparse",
            "python-magic",
            "pdf2image",
            "Pillow",
            "pdfminer.six",
            "openai",
            "pinecone-client[grpc]",
            "aiohttp",
            "tenacity",
            "newrelic_telemetry_sdk"
        ]
    )
)

with pip_image.imports():
    from openai import AsyncOpenAI
    from pinecone.grpc import PineconeGRPC as Pinecone
    from newrelic_telemetry_sdk import Log, LogClient

def get_openai_client():
    openai_org = os.environ["OPENAI_ORG"]
    openai_api_key = os.environ["OPENAI_API_KEY"]

    client = AsyncOpenAI(
    organization=openai_org,
    timeout=600
    )

    client.api_key = openai_api_key
    
    return client


def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index = pc.Index(host="https://cae-xllq9u8.svc.aped-4627-b74a.pinecone.io")
    return index


def log_event(message, attributes, log_type="Info"):
    log_client = LogClient(os.environ["NEW_RELIC_LICENSE_KEY"])
    log_entry = Log(
        message=message,
        attributes=attributes,
        timestamp=int(datetime.datetime.now().timestamp())
    )
    log_client.send(log_entry)
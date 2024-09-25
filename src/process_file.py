# process_file.py
from itertools import islice
import os
import modal
from common import app, pip_image, get_openai_client, get_pinecone_index
import asyncio
import aiohttp
from tqdm import tqdm
import json
from typing import List, Dict

with pip_image.imports():
    import fitz  # PyMuPDF

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

async def process_page(page: fitz.Page, file_key: str, page_num: int) -> List[Dict]:
    text = page.get_text()
    chunks = chunk_text(text)
    
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": f"{file_key}-p{page_num}-c{i}",
            "text": chunk,
            "page_num": page_num,
            "chunk_index": i
        })
    
    return chunk_data

async def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    openai_client = get_openai_client()
    texts = [chunk["text"] for chunk in chunks]
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [e.embedding for e in response.data]
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return chunks
    except Exception as e:
        print(f"Error in embedding chunks: {str(e)}")
        raise

def chunker(seq, batch_size):
    iterator = iter(seq)
    return iter(lambda: list(islice(iterator, batch_size)), [])

async def upsert_to_pinecone(chunks: List[Dict], user_id: str, file_url: str, file_name: str):
    vectors = [{
        "id": chunk["id"],
        "values": chunk["embedding"],
        "metadata": {
            "userId": user_id,
            "fileId": chunk["id"].split("-")[0],
            "text": chunk["text"],
            "pdf_path": file_url,
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
            "file_name": file_name
        }
    } for chunk in chunks]
    
    try:
        pinecone_index = get_pinecone_index()
        async_results = [
            pinecone_index.upsert(vectors=chunk, async_req=True)
            for chunk in chunker(vectors, batch_size=100)  # Adjust batch_size as needed
        ]

        # Wait for and retrieve responses
        responses = [async_result.result() for async_result in async_results]
        
        total_upserted = sum(response.upserted_count for response in responses)
        print(f"Upserted {total_upserted} vectors to Pinecone in {len(responses)} batches.")
    except Exception as e:
        print(f"Error in upserting to Pinecone: {str(e)}")
        raise

async def update_file_status(file_id: str, status: str, chunk_count: int):
    async with aiohttp.ClientSession() as session:
        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/File?id=eq.{file_id}"
        headers = {
            "apikey": os.getenv("SUPABASE_API_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_API_KEY')}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        data = json.dumps({"status": status, "chunkCount": chunk_count})
        
        async with session.patch(url, headers=headers, data=data) as response:
            if response.status != 204:
                print(f"Error updating file status: {response.status}")

@app.function(secrets=[modal.Secret.from_name("cea-secret")], image=pip_image, timeout=3600, keep_warm=1)
async def process_file(file_key: str, user_id: str, file_url: str, file_name: str):
    try:
        # Download the PDF file
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                pdf_content = await response.read()
        
        # Process the PDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        all_chunks = []
        for page_num in tqdm(range(len(doc)), desc="Processing pages"):
            page = doc.load_page(page_num)
            chunks = await process_page(page, file_key, page_num + 1)
            all_chunks.extend(chunks)
        
        # Embed chunks in parallel with rate limiting
        chunk_batches = [all_chunks[i:i+100] for i in range(0, len(all_chunks), 100)]
        embedded_chunks = []
        for batch in tqdm(chunk_batches, desc="Embedding chunks"):
            embedded_batch = await embed_chunks(batch)
            embedded_chunks.extend(embedded_batch)
            await asyncio.sleep(0.1)  # Rate limiting: 10 requests per second
        
        # Upsert to Pinecone
        await upsert_to_pinecone(embedded_chunks, user_id, file_url, file_name)
        
        # Update file status in Supabase
        await update_file_status(file_key, "SUCCESS", len(embedded_chunks))
        
        print(f"File processing completed successfully: {file_key}")
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        await update_file_status(file_key, "ERROR", 0)

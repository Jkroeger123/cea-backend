import math
import os
import traceback
import modal
from common import app, log_event, pip_image, get_openai_client, get_pinecone_index
import asyncio
import aiohttp
import json
from typing import List, Dict, Tuple
import base64
from io import BytesIO
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from process_file import create_file

with pip_image.imports():
    import fitz  # PyMuPDF
    from PIL import Image

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def analyze_image_with_gpt4(image: Image.Image, prompt: str, user_id: str) -> str:
    openai_client = get_openai_client()
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ],
                }
            ],
            max_tokens=4096,
        )
        tokens = response.usage.total_tokens
        log_event(message='Tokens logged', attributes={'tokens': tokens, 'user_id': user_id})
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analyzing image with GPT-4: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def upload_image_chunk(image: Image.Image, file_name: str) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    file_content = buffered.getvalue()
    file_size = len(file_content)

    async with aiohttp.ClientSession() as session:
        presign_url = "https://api.uploadthing.com/v6/uploadFiles"
        presign_payload = json.dumps({
            "files": [{"name": file_name, "size": file_size, "type": 'png'}],
            "acl": "public-read",
            "contentDisposition": "inline"
        })
        presign_headers = {
            "Content-Type": "application/json",
            "X-Uploadthing-Api-Key": os.environ["UPLOADTHING_API_KEY"]
        }
        
        # Ensure all headers are strings and remove any None values
        presign_headers = {str(k): str(v) for k, v in presign_headers.items() if k is not None and v is not None}
        
        try:
            async with session.post(presign_url, data=presign_payload, headers=presign_headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get presigned URL. Status: {response.status}, Response: {error_text}")
                
                presign_data = await response.json()
                
                if "data" not in presign_data or not presign_data["data"]:
                    raise Exception(f"Invalid presign data received: {presign_data}")
                
                upload_url = presign_data["data"][0]["url"]
                upload_fields = presign_data["data"][0]["fields"]

            form_data = aiohttp.FormData()
            for key, value in upload_fields.items():
                form_data.add_field(str(key), str(value))
            form_data.add_field("file", file_content, filename=file_name, content_type="image/png")
            
            async with session.post(upload_url, data=form_data) as upload_response:
                if upload_response.status != 204:
                    error_text = await upload_response.text()
                    raise Exception(f"Failed to upload file. Status: {upload_response.status}, Response: {error_text}")

            if "fileUrl" not in presign_data["data"][0]:
                raise Exception(f"File URL not found in response: {presign_data}")

            return (presign_data["data"][0]["fileUrl"], presign_data["data"][0]["key"])
        
        except Exception as e:
            print(f"Error in upload_image_chunk: {str(e)}")
            raise

async def process_chunk(chunk: Image.Image, page_metadata: str, file_key: str, page_num: int, chunk_index: int, chunk_position: str, user_id: str, file_name: str):
    chunk_description = await analyze_image_with_gpt4(chunk, "Describe the content of this section of a construction document. Focus on visible elements, measurements, and any text present. Any text visible should be included in the output, alongside descriptions of the visual content. If anything is cut off, you can assume / make a best guess at the rest of the text. Be concise - pure information, no extra words or grammar needed.", user_id)
    
    full_text = f"{page_metadata}\n\nChunk Description: {chunk_description}"
    
    openai_client = get_openai_client()
    embedding_response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[full_text]
    )
    embedding = embedding_response.data[0].embedding

    chunk_name  = f"{file_key}-p{page_num}-c{chunk_index}"

    (chunk_image_url, fileKey) = await upload_image_chunk(chunk, chunk_name)
    await create_file(user_id, chunk_image_url, file_name, fileKey)

    return {
        "id": chunk_name,
        "text": full_text,
        "page_num": page_num + 1,
        "chunk_index": chunk_index,
        "chunk_position": chunk_position,
        "embedding": embedding,
        "chunk_image_url": chunk_image_url
    }

@app.function(secrets=[modal.Secret.from_name("cea-secret")], image=pip_image, timeout=86400)
async def process_page(page_input: Tuple[int, fitz.Page], file_key: str, user_id: str, file_url: str, file_name: str):
    page_num, total_pages = page_input
    
    # Download the PDF content in each worker
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            pdf_content = await response.read()
    
    # Open the PDF and load the specific page
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    page = doc.load_page(page_num)
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    page_metadata = await analyze_image_with_gpt4(img, f"Analyze this construction document. This is page {page_num + 1} of {total_pages}. Provide the page ID if visible (make sure to always clarify this content is on page {page_num + 1}, no matter what the page ID says), and a high-level description of the page content. Focus on key elements and their purpose. Assume we know this is a construction document, the high level description should be the scope of the page provided within the full project. Be concise - pure information, no extra words or grammar needed.", user_id)

    chunk_size = 1024
    overlap = 200
    width, height = img.size
    
    # Calculate the number of chunks in each dimension
    num_chunks_x = math.ceil((width - overlap) / (chunk_size - overlap))
    num_chunks_y = math.ceil((height - overlap) / (chunk_size - overlap))
    
    chunks = []
    
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            # Calculate the coordinates for this chunk
            left = x * (chunk_size - overlap)
            top = y * (chunk_size - overlap)
            right = min(left + chunk_size, width)
            bottom = min(top + chunk_size, height)
            
            # Create a new image with a white background
            chunk = Image.new('RGB', (chunk_size, chunk_size), (255, 255, 255))
            
            # Paste the portion of the original image onto the new image
            chunk.paste(img.crop((left, top, right, bottom)), (0, 0))
            
            chunk_position = f"({x}, {y})"
            chunks.append((chunk, chunk_position))

    chunk_results = await asyncio.gather(*[
        process_chunk(chunk[0], page_metadata, file_key, page_num, i, chunk[1], user_id, file_name)
        for i, chunk in enumerate(chunks)
    ])

    pinecone_index = get_pinecone_index()
    vectors = [{
        "id": chunk["id"]+'-image',
        "values": chunk["embedding"],
        "metadata": {
            "userId": user_id,
            "fileId": file_key,
            "text": chunk["text"],
            "pdf_path": file_url,
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
            "chunk_position": chunk["chunk_position"],
            "file_name": file_name,
            "chunk_image_url": chunk["chunk_image_url"]
        }
    } for chunk in chunk_results]

    try:
        pinecone_index.upsert(vectors=vectors, namespace=user_id)
        print(f"Upserted {len(vectors)} vectors to Pinecone for page {page_num}.")
    except Exception as e:
        print(f"Error in upserting to Pinecone for page {page_num}: {str(e)}")
        raise

    return len(vectors)

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

@app.function(secrets=[modal.Secret.from_name("cea-secret")], image=pip_image, timeout=86400, keep_warm=1)
async def process_construction(file_key: str, user_id: str, file_url: str, file_name: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                pdf_content = await response.read()
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Prepare input for for_each: (page_number, total_pages)
        page_inputs: List[Tuple[int, int]] = [(i, len(doc)) for i in range(len(doc))]
        
        # Use for_each to process pages in parallel
        await process_page.for_each.aio(page_inputs, kwargs={
            'file_key': file_key,
            'user_id': user_id,
            'file_url': file_url,
            'file_name': file_name
        })
        
        print(f"Image embedding processing completed successfully: {file_key}")
    except Exception as e:
        print(f"Error image processing file: {str(e)}")
        raise e
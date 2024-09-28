#import all functions to deploy
from modal import web_endpoint
from fastapi import responses, HTTPException, status
from process_contruction import process_construction
from process_file import *
from common import *

@app.function(secrets=[modal.Secret.from_name("cea-secret")], image=pip_image, timeout=86400)
async def handle_file(file_key: str, user_id: str, file_url: str, file_name: str):

    try:
        # Create tasks for both processes
        process_file_task = process_file.remote.aio(file_key, user_id, file_url, file_name)
        process_construction_task = process_construction.remote.aio(file_key, user_id, file_url, file_name)
        
        # Await both tasks concurrently
        await asyncio.gather(process_file_task,process_construction_task)
        
        # Update file status after both processes complete
        await update_file_status(file_key, "SUCCESS", 0)
        
        print(f"File processing completed successfully: {file_key}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        await update_file_status(file_key, "ERROR", 0)


@app.function(secrets=[modal.Secret.from_name("cea-secret")], image=pip_image, timeout=86400, keep_warm=1)
@modal.web_endpoint(method="POST")
async def embed_file(item: Dict):
    try:
        file_key = item["file_key"]
        user_id = item["user_id"]
        file_url = item["file_url"]
        file_name = item["file_name"]
        
        # Check if file already exists
        file_exists = await check_file_exists(user_id, file_name)

        if file_exists:
            return responses.JSONResponse(content="This file has already been processed", status_code=status.HTTP_409_CONFLICT)
            
        # Spawn the process_file functions
        handle_file.spawn(file_key, user_id, file_url, file_name)
        
        return {
            "status": "processing_started",
            "message": "File processing has been initiated",
            "file_key": file_key
        }
    except KeyError as e:
        return responses.JSONResponse(content=f"Missing required field: {str(e)}", status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return responses.JSONResponse(content=f"An unexpected error occurred: {str(e)}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
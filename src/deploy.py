#import all functions to deploy
from modal import web_endpoint
from process_file import *
from common import *

@app.function(secrets=[modal.Secret.from_name("cea-secret")], image=pip_image, timeout=86400, keep_warm=1)
@modal.web_endpoint(method="POST")
async def embed_file(item: Dict):
    try:
        file_key = item["file_key"]
        user_id = item["user_id"]
        file_url = item["file_url"]
        file_name = item["file_name"]
        
        # Spawn the process_file function
        process_file.spawn(file_key, user_id, file_url, file_name)
        
        return {
            "status": "processing_started",
            "message": "File processing has been initiated",
            "file_key": file_key
        }
    except KeyError as e:
        return {
            "status": "error",
            "message": f"Missing required field: {str(e)}",
        }, 400
    except Exception as e:
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
        }, 500
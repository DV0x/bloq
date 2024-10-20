import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.services.file import PrivateFileService

file_upload_router = r = APIRouter()

logger = logging.getLogger("uvicorn")


class FileUploadRequest(BaseModel):
    base64: str
    filename: str
    params: Any = None


@r.post("")
def upload_file(request: FileUploadRequest) -> Dict[str, Any]:
    """
    To upload a private file from the chat UI.
    Returns:
        The metadata of the uploaded file.
    """
    try:
        logger.info(f"Processing file: {request.filename}")
        file_meta = PrivateFileService.process_file(
            request.filename, request.base64, request.params
        )
        return file_meta.to_upload_response()
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing file")

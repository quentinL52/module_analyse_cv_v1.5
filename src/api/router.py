import uuid
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from src.services.pipeline import execute_cv_pipeline, TASKS_STORE
from src.core.db import get_db
from src.core.config import settings

router = APIRouter()

@router.post("/async", status_code=202)
async def analyze_cv_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_description: str = Form(None),
    db: Session = Depends(get_db)
):
    if file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds max size of {settings.MAX_FILE_SIZE_MB}MB")

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    task_id = str(uuid.uuid4())
    
    # Exécution en tâche de fond pour répondre < 60s
    background_tasks.add_task(execute_cv_pipeline, task_id, pdf_bytes, file.filename, job_description, db)

    return {"task_id": task_id, "status": "ACCEPTED", "message": "Traitement en cours"}

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task = TASKS_STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
        
    return task

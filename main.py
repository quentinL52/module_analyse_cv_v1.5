from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.router import router as cv_router
from src.core.db import engine, Base
from src.core.config import settings
from fastapi import Depends
from src.core.security import verify_api_key

# Création des tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API d'Analyse de CV (SaaS Plug & Play)"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion du routeur
app.include_router(
    cv_router, 
    prefix="/api/v1/cv", 
    tags=["CV Analysis"],
    dependencies=[Depends(verify_api_key)]
)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": settings.VERSION}

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from src.core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CVMetrics(Base):
    __tablename__ = "cv_metrics"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    timestamp_execution = Column(DateTime, default=datetime.utcnow)
    version_module = Column(String, default=settings.VERSION)
    model_used = Column(String, index=True)
    duree_etape_1_ocr = Column(Float)
    duree_etape_2_extraction = Column(Float)
    duree_etape_4_agents = Column(Float)
    duree_totale = Column(Float)
    tokens_prompt = Column(Integer, default=0)
    tokens_completion = Column(Integer, default=0)
    fallback_ocr_triggered = Column(Boolean, default=False)
    statut_final = Column(String)  # COMPLETED, DEGRADED_MODE, FAILED

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

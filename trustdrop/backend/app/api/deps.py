from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from app.services.assessment import TrustEngine


def get_db(request: Request):
    db: Session = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_engine(request: Request) -> TrustEngine:
    eng = request.app.state.engine
    if eng is None:
        raise HTTPException(503, detail="Model not loaded. Train first: python scripts/train.py")
    return eng

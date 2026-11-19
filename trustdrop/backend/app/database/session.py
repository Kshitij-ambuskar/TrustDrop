from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import Base


def make_session_factory(url: str) -> sessionmaker[Session]:
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    if ":memory:" in url:  # one shared connection, otherwise every connection would see its own empty DB
        kwargs["poolclass"] = StaticPool
    elif url.startswith("sqlite:///"):
        from pathlib import Path
        Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)

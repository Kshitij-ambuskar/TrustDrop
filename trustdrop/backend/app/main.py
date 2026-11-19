"""FastAPI application factory. Routes are thin; all logic lives in services/ and ml/."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_assess, routes_counseling, routes_students, routes_system
from app.config import settings
from app.database.session import make_session_factory
from app.services.assessment import TrustEngine
from app.services.seed import seed_demo_students

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("trustdrop")


def create_app(models_dir: Path | None = None, reports_dir: Path | None = None,
               database_url: str | None = None, seed: bool = True) -> FastAPI:
    models_dir, reports_dir = Path(models_dir or settings.models_dir), Path(reports_dir or settings.reports_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.session_factory = make_session_factory(database_url or settings.database_url)
        try:
            app.state.engine = TrustEngine.load(models_dir, reports_dir)
            log.info("Model loaded: %s (%s)", app.state.engine.bundle["model_name"], app.state.engine.label)
            if seed:
                with app.state.session_factory() as db:
                    log.info("Seeded %s demo students", seed_demo_students(db, app.state.engine.is_synthetic))
        except FileNotFoundError as e:
            app.state.engine = None
            log.error("Model not loaded: %s", e)
        yield

    app = FastAPI(title="TrustDrop API", version="1.0.0", lifespan=lifespan,
                  description="Trustworthy-AI decision support for student dropout risk. "
                              "Research/educational prototype - not for automated decisions.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
    for r in (routes_system.router, routes_assess.router, routes_students.router, routes_counseling.router):
        app.include_router(r)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")  # message only; request bodies (student data) are never logged
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()

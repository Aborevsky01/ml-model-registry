import json
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import time
from datetime import datetime

from . import models, schemas
from .database import engine, SessionLocal, init_db
from .redis_client import redis_client

from fastapi.templating import Jinja2Templates
from fastapi import Request

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

REQUEST_COUNT = Counter("registry_requests_total", "Total requests", ["endpoint", "method"])
LATEST_HIST = Histogram("registry_latest_seconds", "Time for get_latest_version", ["model_name"])

templates = Jinja2Templates(directory="app/templates")
init_db()

app = FastAPI(
    title="ML Model Registry",
    description="A simple Model Registry for tracking ML models and versions",
    version="1.0.0"
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _cache_key_latest(model_name: str, stage: str) -> str:
    return f"latest:{model_name}:{stage}"


@app.post("/models", response_model=schemas.ModelRead, status_code=status.HTTP_201_CREATED)
def create_model(model: schemas.ModelCreate, db: Session = Depends(get_db)):
    db_model = db.query(models.RegisteredModel).filter(models.RegisteredModel.name == model.name).first()
    if db_model:
        raise HTTPException(status_code=400, detail="Model with this name already exists")
    
    new_model = models.RegisteredModel(**model.model_dump())
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


@app.get("/models", response_model=List[schemas.ModelRead])
def list_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.RegisteredModel).offset(skip).limit(limit).all()


@app.post("/models/{model_name}/versions", response_model=schemas.ModelVersionRead)
def create_model_version(model_name: str, version_in: schemas.ModelVersionCreate, db: Session = Depends(get_db)):
    db_model = db.query(models.RegisteredModel).filter(models.RegisteredModel.name == model_name).first()
    if not db_model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Вычисляем номер следующей версии
    latest_version = db.query(models.ModelVersion).filter(
        models.ModelVersion.model_id == db_model.id
    ).order_by(models.ModelVersion.version.desc()).first()
    
    next_version_num = (latest_version.version + 1) if latest_version else 1

    # Превращаем словари params и metrics в JSON-строки для БД
    params_str = json.dumps(version_in.params) if version_in.params else None
    metrics_str = json.dumps(version_in.metrics) if version_in.metrics else None

    new_version = models.ModelVersion(
        model_id=db_model.id,
        version=next_version_num,
        stage="DEV", # По умолчанию всегда DEV
        artifact_path=version_in.artifact_path,
        git_commit=version_in.git_commit,
        data_ref=version_in.data_ref,
        params_json=params_str,
        metrics_json=metrics_str,
        created_by=version_in.created_by,
        training_env=version_in.training_env,
        pipeline_version=version_in.pipeline_version,
        run_id=version_in.run_id,
    )
    
    db.add(new_version)
    db.commit()
    db.refresh(new_version)
    
    # Возвращаем в формате, который ждет Pydantic (распаковываем JSON обратно)
    return prepare_version_response(new_version)


@app.post("/models/{model_name}/versions/{version}/stage", response_model=schemas.ModelVersionRead)
def update_version_stage(
    model_name: str,
    version: int,
    stage_update: schemas.StageUpdate,
    db: Session = Depends(get_db),
):
    db_model = db.query(models.RegisteredModel).filter(
        models.RegisteredModel.name == model_name
    ).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    db_version = db.query(models.ModelVersion).filter(
        models.ModelVersion.model_id == db_model.id,
        models.ModelVersion.version == version,
    ).first()
    if not db_version:
        raise HTTPException(status_code=404, detail="Version not found")

    # Если переводим в PRODUCTION — снимаем старую PROD в ARCHIVED
    if stage_update.stage == "PRODUCTION":
        old_prod = db.query(models.ModelVersion).filter(
            models.ModelVersion.model_id == db_model.id,
            models.ModelVersion.stage == "PRODUCTION",
        ).first()
        if old_prod and old_prod.id != db_version.id:
            old_prod.stage = "ARCHIVED"

    db_version.stage = stage_update.stage
    db.commit()
    db.refresh(db_version)

    for s in ["DEV", "STAGING", "PRODUCTION"]:
        redis_client.delete(_cache_key_latest(model_name, s))

    return prepare_version_response(db_version)


@app.get("/models/{model_name}/latest", response_model=schemas.ModelVersionRead)
def get_latest_version(model_name: str, stage: str = "PRODUCTION", db: Session = Depends(get_db)):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/models/{model_name}/latest", method="GET").inc()

    cache_key = _cache_key_latest(model_name, stage)
    cached = redis_client.get(cache_key)
    if cached:
        LATEST_HIST.labels(model_name=model_name).observe(time.time() - start)
        return json.loads(cached)

    db_model = db.query(models.RegisteredModel).filter(models.RegisteredModel.name == model_name).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    db_version = db.query(models.ModelVersion).filter(
        models.ModelVersion.model_id == db_model.id,
        models.ModelVersion.stage == stage
    ).order_by(models.ModelVersion.version.desc()).first()

    if not db_version:
        raise HTTPException(status_code=404, detail=f"No version found in stage '{stage}'")

    result = prepare_version_response(db_version)
    redis_client.set(cache_key, json.dumps(result, default=str))
    LATEST_HIST.labels(model_name=model_name).observe(time.time() - start)
    return result


def prepare_version_response(db_version: models.ModelVersion):
    version_dict = {c.name: getattr(db_version, c.name) for c in db_version.__table__.columns}

    if isinstance(version_dict.get("created_at"), datetime):
        version_dict["created_at"] = version_dict["created_at"].isoformat()
    
    version_dict["params"] = json.loads(db_version.params_json) if db_version.params_json else None
    version_dict["metrics"] = json.loads(db_version.metrics_json) if db_version.metrics_json else None
    
    return version_dict


@app.get("/models/{model_name}", response_model=schemas.ModelWithVersions)
def get_model_with_versions(model_name: str, db: Session = Depends(get_db)):
    db_model = db.query(models.RegisteredModel).filter(models.RegisteredModel.name == model_name).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    db_versions = db.query(models.ModelVersion).filter(
        models.ModelVersion.model_id == db_model.id
    ).order_by(models.ModelVersion.version.asc()).all()

    versions = [prepare_version_response(v) for v in db_versions]
    model_dict = schemas.ModelRead.model_validate(db_model).model_dump()
    return {**model_dict, "versions": versions}


from fastapi.responses import RedirectResponse

@app.get("/ui/models")
def ui_list_models(request: Request, db: Session = Depends(get_db)):
    db_models = db.query(models.RegisteredModel).all()
    return templates.TemplateResponse(
        "models_list.html",
        {"request": request, "models": db_models},
    )


@app.get("/ui/models/{model_name}")
def ui_model_detail(model_name: str, request: Request, db: Session = Depends(get_db)):
    db_model = db.query(models.RegisteredModel).filter(models.RegisteredModel.name == model_name).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    db_versions = db.query(models.ModelVersion).filter(
        models.ModelVersion.model_id == db_model.id
    ).order_by(models.ModelVersion.version.desc()).all()

    versions = []
    for v in db_versions:
        d = prepare_version_response(v)
        versions.append(d)

    return templates.TemplateResponse(
        "model_detail.html",
        {"request": request, "model": db_model, "versions": versions},
    )


@app.post("/ui/models/{model_name}/versions/{version}/promote")
def ui_promote_version(model_name: str, version: int, db: Session = Depends(get_db)):
    _ = update_version_stage(model_name, version, schemas.StageUpdate(stage="PRODUCTION"), db)
    return RedirectResponse(url=f"/ui/models/{model_name}", status_code=303)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
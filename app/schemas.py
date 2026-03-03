from datetime import datetime
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field


class ModelBase(BaseModel):
    name: str = Field(..., example="fraud-detection-xgb")
    description: Optional[str] = Field(None, example="Model for credit card fraud detection")
    domain: Optional[str] = Field(None, example="fraud_detection")
    owner: Optional[str] = Field(None, example="mlds_180_team")


class ModelCreate(ModelBase):
    """Тело запроса для POST /models"""
    pass


class ModelRead(ModelBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  

class ModelVersionBase(BaseModel):
    artifact_path: str = Field(..., example="models/mlds_180/fraud-detection-xgb/v1")
    git_commit: Optional[str] = Field(None, example="abc123def")
    data_ref: Optional[str] = Field(None, example="s3://bucket/datasets/fraud/v3")
    params: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = Field(None, example="andrej.borevskiy")
    training_env: Optional[str]    
    pipeline_version: Optional[str]
    run_id: Optional[str]


class ModelVersionCreate(ModelVersionBase):
    """Тело запроса для POST /models/{name}/versions"""
    pass


class ModelVersionRead(BaseModel):
    id: int
    model_id: int
    version: int
    stage: str
    artifact_path: str
    git_commit: Optional[str]
    data_ref: Optional[str]
    params: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    created_at: datetime
    created_by: Optional[str]
    training_env: Optional[str]    
    pipeline_version: Optional[str]
    run_id: Optional[str]

    class Config:
        from_attributes = True


class StageUpdate(BaseModel):
    """Тело для смены стадии версии"""
    stage: str = Field(..., example="PRODUCTION")



class ModelWithVersions(ModelRead):
    versions: List[ModelVersionRead] = []

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class RegisteredModel(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    domain = Column(String(255), nullable=True)        
    owner = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    versions = relationship(
        "ModelVersion",
        back_populates="model",
        cascade="all, delete-orphan",
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)

    version = Column(Integer, nullable=False)

    # DEV / STAGING / PRODUCTION / ARCHIVED
    stage = Column(String(32), nullable=False, default="DEV", index=True)

    artifact_path = Column(Text, nullable=False)

    git_commit = Column(String(64), nullable=True)
    data_ref = Column(Text, nullable=True)      

    params_json = Column(Text, nullable=True)    
    metrics_json = Column(Text, nullable=True)    

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255), nullable=True)

    model = relationship("RegisteredModel", back_populates="versions")

    training_env = Column(Text, nullable=True)      
    pipeline_version = Column(String(64), nullable=True)
    run_id = Column(String(64), nullable=True)


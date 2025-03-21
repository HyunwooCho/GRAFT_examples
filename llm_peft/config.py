''' backend/config.py (Pydantic 셋팅)
    환경 변수(.env) 로딩
'''
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    huggingface_token: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_s3_bucket: str
    database_url: str = "postgresql+asyncpg://graft_usr:graft_pw@db/graft_db"
    hf_home: str = "/shared/models"
    hf_datasets_cache: str = "/shared/datasets"

    class Config:
        env_file = ".env"
        env_file_encodeing = "utf-8"
        env_prefix = ""
        case_sensitive = False

settings = Settings()


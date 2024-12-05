from pydantic import BaseSettings

class CorsSettings(BaseSettings):
    allow_origins: list[str] = ["*"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]

    class Config():
        env_prefix = "cors"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

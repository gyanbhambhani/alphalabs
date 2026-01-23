from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App settings
    app_name: str = "AI Trading Lab"
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://localhost:5432/trading_lab"
    
    # Redis (optional)
    redis_url: str = "redis://localhost:6379"
    
    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    
    # LLM API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    
    # ChromaDB
    chroma_persist_directory: str = "./chroma_data"
    
    # Trading settings
    initial_capital: float = 100000.0
    trading_universe: list[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
        "META", "TSLA", "AMD", "INTC", "NFLX"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

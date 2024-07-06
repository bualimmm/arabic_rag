from pydantic import BaseModel


class Settings(BaseModel):
    """Settings for app urls"""
    embeddings_model: str = "https://Cohere-embed-v3-alnam-serverless.eastus2.inference.ai.azure.com"
    llm_model: str = "https://Cohere-command-r-plus-alnam-serverless.eastus2.inference.ai.azure.com"


settings = Settings()

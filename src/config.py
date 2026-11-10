"""
IBM Cloud / watsonx.ai configuration manager for the Interview Trainer Agent.
Loads credentials from environment variables and provides initialised clients.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()


@dataclass
class IBMConfig:
    # watsonx.ai
    api_key: str = field(default_factory=lambda: os.getenv("IBM_CLOUD_API_KEY", ""))
    cloud_url: str = field(default_factory=lambda: os.getenv("IBM_CLOUD_URL", "https://us-south.ml.cloud.ibm.com"))
    project_id: str = field(default_factory=lambda: os.getenv("IBM_WATSON_PROJECT_ID", ""))

    # Models confirmed available on this watsonx.ai project (verified from API)
    granite_chat_model: str = field(default_factory=lambda: os.getenv("IBM_GRANITE_MODEL_ID", "mistralai/mistral-small-3-1-24b-instruct-2503"))
    granite_instruct_model: str = field(default_factory=lambda: os.getenv("IBM_GRANITE_INSTRUCT_MODEL", "mistralai/mistral-small-3-1-24b-instruct-2503"))
    granite_embedding_model: str = field(default_factory=lambda: os.getenv("IBM_GRANITE_EMBEDDING_MODEL", "ibm/slate-125m-english-rtrvr-v2"))

    # Watson Discovery
    discovery_api_key: str = field(default_factory=lambda: os.getenv("WATSON_DISCOVERY_API_KEY", ""))
    discovery_url: str = field(default_factory=lambda: os.getenv("WATSON_DISCOVERY_URL", ""))
    discovery_instance_id: str = field(default_factory=lambda: os.getenv("WATSON_DISCOVERY_INSTANCE_ID", ""))
    discovery_project_id: str = field(default_factory=lambda: os.getenv("WATSON_DISCOVERY_PROJECT_ID", ""))
    discovery_collection_id: str = field(default_factory=lambda: os.getenv("WATSON_DISCOVERY_COLLECTION_ID", ""))

    # App settings
    vector_store_path: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))
    top_k_retrieval: int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "5")))
    max_questions: int = field(default_factory=lambda: int(os.getenv("MAX_QUESTIONS_PER_SESSION", "10")))

    def validate(self) -> bool:
        """Return True if mandatory IBM Cloud credentials are present."""
        return bool(self.api_key and self.project_id)

    def __repr__(self) -> str:
        masked = self.api_key[:6] + "****" if self.api_key else "<not set>"
        return (
            f"IBMConfig(api_key={masked}, cloud_url={self.cloud_url}, "
            f"project_id={self.project_id or '<not set>'})"
        )


# Singleton config instance
config = IBMConfig()

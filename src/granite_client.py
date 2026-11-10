"""
IBM Granite LLM wrapper using ibm-watsonx-ai SDK.
Uses models confirmed available on this watsonx.ai project:
  - Chat/instruct: ibm/granite-4-h-small
  - Fallback:      meta-llama/llama-3-1-8b
"""

import time
import logging
from typing import Optional
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from src.config import config

logger = logging.getLogger(__name__)


class GraniteClient:
    """Wrapper around IBM Granite / Llama models via watsonx.ai."""

    def __init__(self):
        self._client: Optional[APIClient] = None
        self._model: Optional[ModelInference] = None
        self._initialised = False

    def _ensure_init(self):
        if self._initialised:
            return
        if not config.validate():
            raise EnvironmentError(
                "IBM Cloud credentials missing. "
                "Set IBM_CLOUD_API_KEY and IBM_WATSON_PROJECT_ID in your .env file."
            )
        credentials = Credentials(url=config.cloud_url, api_key=config.api_key)
        self._client = APIClient(credentials=credentials, project_id=config.project_id)

        # Use the instruct model (supports text generation on this project)
        self._model = ModelInference(
            model_id=config.granite_instruct_model,
            api_client=self._client,
            project_id=config.project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.MIN_NEW_TOKENS: 30,
                GenParams.TEMPERATURE: 0.7,
                GenParams.REPETITION_PENALTY: 1.05,
                GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
            },
        )
        self._initialised = True
        logger.info("IBM watsonx.ai client initialised. Model: %s", config.granite_instruct_model)

    def generate(self, prompt: str, instruct: bool = False, max_retries: int = 3) -> str:
        """
        Generate text from a prompt.
        Retries up to max_retries times on 429 rate-limit errors.
        """
        self._ensure_init()
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._model.generate_text(prompt=prompt)
                return response.strip()
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "429" in err_str or "consumption_limit_reached" in err_str:
                    wait = attempt * 10  # 10s, 20s, 30s
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). Waiting %ds...",
                        attempt, max_retries, wait
                    )
                    time.sleep(wait)
                elif "model_no_support" in err_str or "not supported" in err_str:
                    logger.error("Model not supported: %s", exc)
                    raise
                else:
                    logger.error("Granite generation failed: %s", exc)
                    raise
        raise last_exc

    def generate_with_context(self, system_prompt: str, user_message: str) -> str:
        """
        Format a system + user prompt using the correct template per model family.
        """
        model_id = config.granite_instruct_model.lower()
        if "llama" in model_id:
            # Llama 3 chat template
            prompt = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif "mistral" in model_id or "mistralai" in model_id:
            # Mistral instruct template
            prompt = (
                f"[INST] {system_prompt}\n\n{user_message} [/INST]"
            )
        else:
            # Granite 3+ instruct template
            prompt = (
                f"<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
                f"<|start_of_role|>user<|end_of_role|>{user_message}<|end_of_text|>\n"
                f"<|start_of_role|>assistant<|end_of_role|>"
            )
        return self.generate(prompt)


# Singleton
granite = GraniteClient()

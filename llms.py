# %% [markdown]
# # How to write a custom LLM wrapper
#
# This notebook goes over how to create a custom LLM wrapper, in case you want to use your own LLM or a different wrapper than one that is supported in LangChain.
#
# There is only one required thing that a custom LLM needs to implement:
#
# 1. A `_call` method that takes in a string, some optional stop words, and returns a string
#
# There is a second optional thing it can implement:
#
# 1. An `_identifying_params` property that is used to help with printing of this class. Should return a dictionary.
#
# Let's implement a very simple custom LLM that just returns the first N characters of the input.

# %%
from typing import Any, Union, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# %%


class GPT4AllJ(LLM):
    seed: int = -1
    n_threads: int = -1
    n_predict: int = 200
    top_k: int = 40
    top_p: float = 0.9
    temperature: float = 0.9
    repeat_penalty: int = 1
    repeat_last_n: int = 64
    n_batch: int = 8

    @property
    def _llm_type(self) -> str:
        return "GPT4AllJ"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        api_url = os.environ.get("API_URL")
        payload: Mapping[str, Union[str, Mapping[str, Any]]] = {
            "prompt": prompt,
            "params": self._get_params()
        }
        data = json.dumps(payload)
        print('data', data)
        headers: Mapping[str, Any] = {"Content-Type": "application/json"}
        response = requests.request(
            "POST", api_url, headers=headers, data=data)
        completion = json.loads(response.content.decode("utf-8"))
        return completion['message']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return self._get_params()

    def _get_params(self) -> Mapping[str, Any]:
        return {
            "seed": self.seed,
            "n_threads": self.n_threads,
            "n_predict": self.n_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "n_batch":  self.n_batch
        }

# %% [markdown]
# We can now use this as an any other LLM.

from typing import List

import httpx
from langchain_core.embeddings import Embeddings


class CustomEmbeddings(Embeddings):

    def __init__(self, base_url, api_key):
        self.base_url = base_url + "/v1/embeddings"
        self.api_key = api_key
        super().__init__()

    def _get_header(self):
        return {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = {
            "input_type": 'query',
            "input": texts
        }

        response = httpx.post(self.base_url, json=data, headers=self._get_header()).json()
        return [x['embedding'] for x in response['data']]

    def embed_query(self, text: str) -> List[float]:
        data = {
            "input_type": 'query',
            "input": [text]
        }

        response = httpx.post(self.base_url, json=data, headers=self._get_header()).json()
        return [x['embedding'] for x in response['data']][0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        data = {
            "input_type": 'query',
            "input": texts
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, json=data, headers=self._get_header())
            response.raise_for_status()
            response = response.json()
        return [x['embedding'] for x in response['data']]

    async def aembed_query(self, text: str) -> List[float]:
        data = {
            "input_type": 'query',
            "input": [text]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, json=data, headers=self._get_header())
            response.raise_for_status()
            response = response.json()
        return [x['embedding'] for x in response['data']][0]

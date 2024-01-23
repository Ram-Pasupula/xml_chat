
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from typing import List
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class HuggingFaceEmbeddingServer(Embeddings):
    """
    This class is used to get embeddings for a list of texts using the Text Embedding server
    """

    def __init__(self, url: str):

        try:
            import requests
        except ImportError:
            raise ValueError(
                "The requests python package is not installed. Please install it with `pip install requests`"
            )
        self._api_url = f"{url}"
        self._session = requests.Session()

    def __call__(self, input: List[Document]) -> Embeddings:
        print("I got called List[Document]")
        # Call HuggingFace Embedding Server API for each document
        return self._session.post(  # type: ignore
            self._api_url, json={"inputs": input}
        ).json()

    def __call2__(self, input: Document) -> Embeddings:
        import json

        accept = "application/json"
        content_type = "application/json"
        embeddings = []
        for text in input:
            input_body = {"inputText": text}
            body = json.dumps(input_body)
            response = self._client.invoke_model(
                body=body,
                modelId=self._model_name,
                accept=accept,
                contentType=content_type,
            )
            embedding = json.load(response.get("body")).get("embedding")
            embeddings.append(embedding)
        return embeddings


class HuggingFaceRedisEmbeddings(HuggingFaceEmbeddingServer):
    score_threshold: float = 0.5
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._session.post(
            self._api_url, json={"inputs": texts}
        ).json()

        embeddings = response

        if embeddings:
            dim = len(embeddings)
            print(f"doc dim : {dim}")
        else:
            print("Embeddings list is empty.")
        return (embeddings)

    def embed_query(self, text: str) -> List[float]:
        """
        Get the embeddings for a single text.

        Args:
            text (str): The text to get embeddings for.

        Returns:
            List[float]: The embeddings for the text.
        """
        response = self._session.post(
            self._api_url, json={"inputs": [text]}
        ).json()
        embedding = response[0]
        return embedding

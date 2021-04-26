from typing import Iterable

from azure.storage.blob import ContainerClient


class ContainerService:
    def __init__(self, connection_string, container_name) -> None:
        self.container_client = ContainerClient.from_connection_string(
            connection_string, container_name
        )

    def download_blob(self, blob_name) -> bytes:
        blob = self.container_client.get_blob_client(blob_name)
        blob_downloader = blob.download_blob()
        return blob_downloader.readall()

    def upload_blob(self, blob_name, data) -> None:
        self.container_client.upload_blob(blob_name, data)

    def list_blobs(self, prefix) -> Iterable:
        return self.container_client.list_blobs(prefix)

    def get_blob_client(self, blob_name):
        return self.container_client.get_blob_client(blob_name)

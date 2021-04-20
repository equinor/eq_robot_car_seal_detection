import os
from dotenv import load_dotenv

load_dotenv()

STORAGE_CONNECTION_STRING = os.getenv('STORAGE_CONNECTION_STRING')
STORAGE_CONTAINER_NAME = "testdata"

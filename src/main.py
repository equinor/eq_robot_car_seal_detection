from azure.storage.blob import ContainerClient
import cv2
import time
import numpy as np

from config import STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME


if __name__ == "__main__":
    container_client = ContainerClient.from_connection_string(
        STORAGE_CONNECTION_STRING,
        container_name=STORAGE_CONTAINER_NAME
    )
    blob = container_client.get_blob_client("images/green1.jpg")
    blob_downloader = blob.download_blob()

    bytes = blob_downloader.readall()

    nparr = np.frombuffer(bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    
    while True:
        key = cv2.waitKey(0)
        ESCAPE_KEY_CODE = 27
        if key == ESCAPE_KEY_CODE:
            break
        else:
            time.sleep(0.1)
            continue

    cv2.destroyAllWindows()

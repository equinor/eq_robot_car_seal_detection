import cv2
import numpy as np
from azure.storage.blob import ContainerClient

from config import STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME
from image import (
    draw_bounding_box,
    find_color_in_image,
    get_biggest_contour,
    get_image_from_bytes_string,
    show_image,
)

lower_green_bound = np.array([65, 50, 50])
upper_green_bound = np.array([87, 255, 255])

if __name__ == "__main__":
    container_client = ContainerClient.from_connection_string(
        STORAGE_CONNECTION_STRING, container_name=STORAGE_CONTAINER_NAME
    )
    blob = container_client.get_blob_client("images/green1.jpg")
    blob_downloader = blob.download_blob()
    bytes = blob_downloader.readall()
    image = get_image_from_bytes_string(bytes)

    image_binary_where_green = find_color_in_image(
        image, lower_green_bound, upper_green_bound
    )
    biggest_contour = get_biggest_contour(image_binary_where_green)
    bounding_box_xywh = cv2.boundingRect(biggest_contour)
    draw_bounding_box(image, bounding_box_xywh, "green ziptie", [0, 255, 0])

    image_2 = cv2.cvtColor(image_binary_where_green, cv2.COLOR_GRAY2BGR)
    image_to_show = np.concatenate((image, image_2), axis=1)
    show_image(image_to_show)

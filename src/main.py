import cv2
import numpy as np
from azure.storage.blob import ContainerClient
import argparse

from config import STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME
from image import (
    draw_bounding_box,
    find_color_in_image,
    get_biggest_contour,
    get_bytes_string_from_image,
    get_image_from_bytes_string,
    show_image,
)

LOWER_GREEN_BOUND = np.array([65, 50, 50])
UPPER_GREEN_BOUND = np.array([87, 255, 255])
IMAGE_NAME = "green1.jpg"


def main():
    color_name, medium, should_upload = get_arguments()
    show_one_detected_image(should_upload)


def get_arguments():
    args = build_arg_parser()
    color_name = args.color
    medium = args.medium
    should_upload = args.upload

    return color_name, medium, should_upload


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--color", help="color of ziptie detected (red | green)", required=True
    )
    ap.add_argument(
        "--medium", help="what medium to detect in (image | video)", required=True
    )
    ap.add_argument(
        "--upload",
        help="flag for whether or not do uplaod result (default: False)",
        action="store_true",
    )

    return ap.parse_args()


def show_one_detected_image(should_upload):
    container_client = ContainerClient.from_connection_string(
        STORAGE_CONNECTION_STRING, container_name=STORAGE_CONTAINER_NAME
    )
    blob = container_client.get_blob_client(f"images/{IMAGE_NAME}")
    blob_downloader = blob.download_blob()
    bytes = blob_downloader.readall()
    image = get_image_from_bytes_string(bytes)

    image_binary_where_green = find_color_in_image(
        image, LOWER_GREEN_BOUND, UPPER_GREEN_BOUND
    )
    biggest_contour = get_biggest_contour(image_binary_where_green)
    bounding_box_xywh = cv2.boundingRect(biggest_contour)
    draw_bounding_box(image, bounding_box_xywh, "green ziptie", [0, 255, 0])

    if should_upload:
        image_file_extension = f".{IMAGE_NAME.split('.')[-1]}"
        bytes_string = get_bytes_string_from_image(image, image_file_extension)
        container_client.upload_blob(f"processed_images/{IMAGE_NAME}", bytes_string)
    else:
        image_2 = cv2.cvtColor(image_binary_where_green, cv2.COLOR_GRAY2BGR)
        image_to_show = np.concatenate((image, image_2), axis=1)
        show_image(image_to_show)


if __name__ == "__main__":
    main()

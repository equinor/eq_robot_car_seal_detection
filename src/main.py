import argparse

import cv2
import numpy as np

from color import Bound, Color
from config import STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME
from container_service import ContainerService
from image import (
    draw_bounding_box,
    find_color_in_image,
    get_biggest_contour_in_range,
    get_bytes_string_from_image,
    get_image_from_bytes_string,
    show_image,
)

COLORS = [
    Color(
        name="green",
        bounds=[Bound([65, 50, 50], [87, 255, 255])],
        default_color=[0, 255, 0],
    ),
    Color(
        name="red",
        bounds=[
            Bound([177, 50, 50], [180, 255, 255]),
            Bound([0, 50, 50], [8, 255, 255]),
        ],
        default_color=[0, 0, 255],
    ),
]


def main():
    file, should_upload = get_arguments()

    container_service = ContainerService(
        STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME
    )

    if file:
        detect_image_in_container(file, should_upload, container_service)
    else:
        detect_all_images_in_container(should_upload, container_service)


def get_arguments():
    args = build_arg_parser()
    file = args.file
    should_upload = args.upload

    return file, should_upload


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--upload",
        help="flag for whether or not to move image(s) and upload result(s)",
        action="store_true",
    )
    ap.add_argument(
        "--file", help="run detection on a single file, instead of full container"
    )

    return ap.parse_args()


def detect_all_images_in_container(should_upload, container_service):
    for blob in container_service.list_blobs("images/"):
        detect_image_in_container(blob.name, should_upload, container_service)


def detect_image_in_container(blob_name, should_upload, container_service):
    bytes = container_service.download_blob(blob_name)
    image = get_image_from_bytes_string(bytes)

    for color in COLORS:
        image_binary_where_color = np.zeros(image.shape[0:2], dtype=np.uint8)
        for bound in color.bounds:
            image_binary_where_color_bound = find_color_in_image(
                image, np.array(bound.lower), np.array(bound.upper)
            )
            image_binary_where_color = np.bitwise_or(
                image_binary_where_color, image_binary_where_color_bound
            )

        biggest_contour = get_biggest_contour_in_range(image_binary_where_color)
        if len(biggest_contour) > 0:
            bounding_box_xywh = cv2.boundingRect(biggest_contour)
            draw_bounding_box(
                image,
                bounding_box_xywh,
                f"{color.name} ziptie",
                color.default_color,
            )

    if should_upload:
        image_file_extension = f".{blob_name.split('.')[-1]}"
        image_file_name = blob_name.split("/")[-1]
        bytes_string = get_bytes_string_from_image(image, image_file_extension)
        blob_client = container_service.get_blob_client(
            f"processed_images/{image_file_name}"
        )
        blob_client.upload_blob(bytes_string, overwrite=True)
    else:
        show_image(image)


if __name__ == "__main__":
    main()

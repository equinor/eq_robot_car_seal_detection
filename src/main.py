import cv2
import numpy as np
from azure.storage.blob import ContainerClient

from config import STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME

kernel = np.ones((5 * 4, 5 * 4), np.uint8)
lower_green_bound = np.array([65, 50, 50])
upper_green_bound = np.array([87, 255, 255])


def show_image(image):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    while cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) > 0:
        key = cv2.waitKey(100)
        ESCAPE_KEY_CODE = 27
        if key == ESCAPE_KEY_CODE:
            break
        else:
            continue

    cv2.destroyAllWindows()


def resize_image(image, scaling_factor=0.25):
    height = image.shape[0]
    width = image.shape[1]
    new_width = int(scaling_factor * width)
    new_heigth = int(scaling_factor * height)
    return cv2.resize(image, (new_width, new_heigth))


def find_color_in_image(image, lower_color_bound, upper_color_bound):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image, lower_color_bound, upper_color_bound)
    return image


def clean_image(image):
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)
    return image


def display_bounding_box(self, contours, image, color):
    for con in contours:
        M = cv2.moments(con)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        x, y, w, h = cv2.boundingRect(con)

        cv2.rectangle(image, (x, y), (x + w, y + h), self.color[color])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.8
        cv2.putText(
            image,
            color + " ziptie",
            (x, y),
            font,
            font_size,
            self.color[color],
            2,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    container_client = ContainerClient.from_connection_string(
        STORAGE_CONNECTION_STRING, container_name=STORAGE_CONTAINER_NAME
    )
    blob = container_client.get_blob_client(
        "images/Carseal open ventiler oppe på ett rapo.jpg"
    )
    blob_downloader = blob.download_blob()

    bytes = blob_downloader.readall()

    nparr = np.frombuffer(bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # image = resize_image(image)
    green_image = find_color_in_image(image, lower_green_bound, upper_green_bound)
    grey_3_channel = cv2.cvtColor(green_image, cv2.COLOR_GRAY2BGR)
    show_image(np.concatenate((image, grey_3_channel), axis=1))
    # show_image(image)

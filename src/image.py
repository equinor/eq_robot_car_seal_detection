import cv2
import numpy as np


def get_image_from_bytes_string(bytes):
    nparr = np.frombuffer(bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def get_bytes_string_from_image(image, file_extension):
    return cv2.imencode(file_extension, image)[1].tostring()


def find_color_in_image(image, lower_color_bound, upper_color_bound):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image, lower_color_bound, upper_color_bound)
    return image


def get_biggest_contour_in_range(binary_image, min=0.001, max=0.20):
    cont, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min = min * binary_image.size
    max = max * binary_image.size
    biggest_contour = []
    biggest_area = 0
    for con in cont:
        area = cv2.contourArea(con)
        if max > area > min:
            if area > biggest_area:
                biggest_area = area
                biggest_contour = con
    return biggest_contour


def draw_bounding_box(image, bounding_box_xywh, text, color):
    x, y, w, h = bounding_box_xywh

    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness=3)

    font_size = 0.8
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        color,
        2,
        cv2.LINE_AA,
    )


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

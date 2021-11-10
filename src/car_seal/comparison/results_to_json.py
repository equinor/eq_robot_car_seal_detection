import json
import os

import cv2
from car_seal.bounding_box import BoundingBox

PREDICTION_THRESHOLD = 50


def normalize_bounding_box(img_name, left, top, width, height):
    def limit(desimal):
        if desimal < 0:
            return 0
        elif desimal > 1:
            return 1
        return desimal

    image_path = os.path.join(
        os.path.dirname(__file__), "../../../dataset/images/" + img_name + ".JPG"
    )
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_height, img_width, _ = image.shape
    left = limit(left / img_width)
    top = limit(top / img_height)
    width = limit(width / img_width)
    height = limit(height / img_height)
    return left, top, width, height


def parse_txt(file_path):
    results = {}
    current_img = None
    with open(file_path, "r") as f:
        for line in f:
            if "/home" in line:
                img_name = line.split("obj/")[1].split(".JPG")[0]
                current_img = img_name
                results[current_img] = []
            elif "Car Seal" in line:
                percentage = line.split("%")[0].split(" ")[-1]
                if int(percentage) < PREDICTION_THRESHOLD:
                    continue
                box = line.split("(")[1].split(")")[0].split(" ")
                _, left, _, top, _, width, _, height = [word for word in box if word]
                if "Green" in line:
                    label = "green_car_seal"
                elif "Red" in line:
                    label = "red_car_seal"
                left, top, width, height = normalize_bounding_box(
                    img_name=current_img,
                    left=int(left),
                    top=int(top),
                    width=int(width),
                    height=int(height),
                )
                box = BoundingBox(
                    label=label, left=left, top=top, width=width, height=height
                )
                results[current_img].append(box)
    return results


if __name__ == "__main__":

    file_path = os.path.join(os.path.dirname(__file__), "results/darknet_results.txt")
    results = parse_txt(file_path=file_path)
    file_path = os.path.join(
        os.path.dirname(__file__),
        f"results/darknet_results_{PREDICTION_THRESHOLD}.json",
    )
    with open(file_path, "w") as f:
        json.dump(results, f)

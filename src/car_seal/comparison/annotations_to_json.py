import json
import os

from car_seal.bounding_box import BoundingBox


def read_annotations(folder_path):
    results = {}
    annotation_file_pahts = os.listdir(folder_path)
    for annotation_file_name in annotation_file_pahts:
        name, _ = annotation_file_name.split(".")
        annotation_file_path = os.path.join(folder_path, annotation_file_name)
        with open(annotation_file_path, "r") as f:
            results[name] = []
            for line in f:
                line = line.strip()
                label, x, y, width, height = line.split(" ")
                if int(label) == 0:
                    label = "red_car_seal"
                elif int(label) == 1:
                    label = "green_car_seal"
                x, y, width, height = [float(p) for p in [x, y, width, height]]
                box = BoundingBox(
                    label=label,
                    left=(x - width / 2),
                    top=(y - height / 2),
                    width=width,
                    height=height,
                )
                results[name].append(box)

    return results


if __name__ == "__main__":
    folder_path_annotations = os.path.join(
        os.path.dirname(__file__), "../../../dataset/annotations/"
    )
    results = read_annotations(folder_path=folder_path_annotations)
    file_path = os.path.join(
        os.path.dirname(__file__), "results/annotations_results.json"
    )
    with open(file_path, "w") as f:
        json.dump(results, f)

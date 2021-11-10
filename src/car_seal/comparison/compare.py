import json
import os

import cv2

IOU_THRESHOLD = 0.5


def read_json_results(file_path):
    with open(file_path, "r") as f:
        return json.loads(f.read())


def compare_results(ground_truth, result1, result2):
    for image in result1:
        show_results(image, ground_truth[image], result1[image], result2[image])


def draw_bounding_boxes(img, annotations, img_width, img_height, color, line_size):
    for annotation in annotations:
        # Scale up normalized coordinates for drawing
        left = int(annotation[1] * img_width)
        top = int(annotation[2] * img_height)
        width = int(annotation[3] * img_width)
        height = int(annotation[4] * img_height)
        cv2.rectangle(img, (left, top), (left + width, top + height), color, line_size)
        cv2.putText(
            img,
            annotation[0].split("_")[0],
            (left, max(top - int(line_size / 2), 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
        )


def show_results(img_name, ground_truth, annotations1, annotations2):
    RED = (0, 0, 255)
    BBOX_LINE_SIZE_RED = 5
    GREEN = (0, 255, 0)
    BBOX_LINE_SIZE_GREEN = 4
    BLUE = (255, 0, 0)
    BBOX_LINE_SIZE_BLUE = 3

    image_path = os.path.join(
        os.path.dirname(__file__), "../../../dataset/images/" + img_name + ".JPG"
    )
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_height, img_width, _ = img.shape

    draw_bounding_boxes(
        img, ground_truth, img_width, img_height, RED, BBOX_LINE_SIZE_RED
    )
    draw_bounding_boxes(
        img, annotations1, img_width, img_height, GREEN, BBOX_LINE_SIZE_GREEN
    )
    draw_bounding_boxes(
        img, annotations2, img_width, img_height, BLUE, BBOX_LINE_SIZE_BLUE
    )
    cv2.imshow("image", img)

    key = cv2.waitKey(0)
    if key == 113:
        raise Exception


def calculate_iou(box1, box2):
    left_1, top_1, width_1, height_1 = box1[1:]
    left_2, top_2, width_2, height_2 = box2[1:]
    area_1 = width_1 * height_1
    area_2 = width_2 * height_2

    intersection_right = min(left_1 + width_1, left_2 + width_2)
    intersection_left = max(left_1, left_2)
    intersection_bottom = min(top_1 + height_1, top_2 + height_2)
    intersection_top = max(top_1, top_2)
    intersection_width = intersection_right - intersection_left
    intersection_height = intersection_bottom - intersection_top
    if intersection_width < 0 or intersection_height < 0:
        return 0
    intersection_area = intersection_height * intersection_width
    union_area = area_1 + area_2 - intersection_area

    return intersection_area / union_area


def compare_bounding_boxes(ground_truth: list, prediction: list):
    true_pos = 0
    false_neg = 0
    used_predictions = set()
    for gt_box in ground_truth:
        gt_label = gt_box[0]
        max_score = 0
        max_score_index = None
        for i, pred_box in enumerate(prediction):
            if gt_label != pred_box[0]:
                continue
            score = calculate_iou(gt_box, pred_box)
            if score > max(max_score, IOU_THRESHOLD):
                max_score = score
                max_score_index = i
        if max_score_index is not None:
            used_predictions.add(max_score_index)
            true_pos += 1
        else:
            false_neg += 1
    false_pos = len(prediction) - len(used_predictions)
    return true_pos, false_pos, false_neg


def check_performance(ground_truths: dict, predictions: dict):
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for image in predictions:
        result = compare_bounding_boxes(ground_truths[image], predictions[image])
        true_pos += result[0]
        false_pos += result[1]
        false_neg += result[2]
    return true_pos, false_pos, false_neg


if __name__ == "__main__":
    file_path_darknet = os.path.join(
        os.path.dirname(__file__), "results/darknet_results_80.json"
    )
    file_path_custom_vision = os.path.join(
        os.path.dirname(__file__), "results/custom_vision_results.json"
    )
    file_path_annotations = os.path.join(
        os.path.dirname(__file__), "results/annotations_results.json"
    )
    darknet_results = read_json_results(file_path=file_path_darknet)
    custom_vision_results = read_json_results(file_path=file_path_custom_vision)
    annotations_results = read_json_results(file_path=file_path_annotations)

    dn_true_pos, dn_false_pos, dn_false_neg = check_performance(
        annotations_results, darknet_results
    )
    dn_precision = dn_true_pos / (dn_true_pos + dn_false_pos)
    dn_recall = dn_true_pos / (dn_true_pos + dn_false_neg)

    cv_true_pos, cv_false_pos, cv_false_neg = check_performance(
        annotations_results, custom_vision_results
    )
    cv_precision = cv_true_pos / (cv_true_pos + cv_false_pos)
    cv_recall = cv_true_pos / (cv_true_pos + cv_false_neg)

    print("Darknet")
    print(
        f"True pos: {dn_true_pos}, False pos: {dn_false_pos}, False neg: {dn_false_neg}"
    )
    print(f"Precision: {dn_precision}, Recall: {dn_recall}")

    print("Custom vision")
    print(
        f"True pos: {cv_true_pos}, False pos: {cv_false_pos}, False neg: {cv_false_neg}"
    )
    print(f"Precision: {cv_precision}, Recall: {cv_recall}")

    print("Press q to quit, press any other key to continue")
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    compare_results(
        ground_truth=annotations_results,
        result1=darknet_results,
        result2=custom_vision_results,
    )

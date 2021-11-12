import json
import os

import cv2
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from car_seal.bounding_box import BoundingBox
from car_seal.config import (
    PREDICTION_ENDPOINT,
    PREDICTION_KEY,
    TRAINING_ENDPOINT,
    TRAINING_KEY,
)
from car_seal.custom_vision.reader import read_and_resize_image
from msrest.authentication import ApiKeyCredentials

PREDICTION_THRESHOLD = 0.5


class PredictImages:
    def __init__(self):
        # Load credentials from environment

        # Authenticate the training client
        credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
        trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)

        # Authenticate the prediction client
        prediction_credentials = ApiKeyCredentials(
            in_headers={"Prediction-key": PREDICTION_KEY}
        )
        self.predictor = CustomVisionPredictionClient(
            PREDICTION_ENDPOINT, prediction_credentials
        )

        project_name = "car_seal_train_and_validation"
        self.publish_iteration_name = "Iteration3"
        self.max_byte_size = 4000000

        projects = trainer.get_projects()
        project_id = next((p.id for p in projects if p.name == project_name), None)

        print("Connecting to existing project...")
        self.project = trainer.get_project(project_id)

    @staticmethod
    def show_prediction(image_path, predictions):
        GREEN = (0, 255, 0)
        BBOX_LINE_SIZE = 5
        image = cv2.imread(image_path)
        img_height = image.shape[0]
        img_width = image.shape[1]

        for prediction in predictions:
            if prediction.probability < PREDICTION_THRESHOLD:
                continue
            print(
                "\t",
                prediction.tag_name,
                ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(
                    prediction.probability * 100,
                    prediction.bounding_box.left,
                    prediction.bounding_box.top,
                    prediction.bounding_box.width,
                    prediction.bounding_box.height,
                ),
            )

            left = prediction.bounding_box.left * img_width
            top = prediction.bounding_box.top * img_height
            width = prediction.bounding_box.width * img_width
            height = prediction.bounding_box.height * img_height

            x0 = int(left)
            y0 = int(top)
            x1 = int(left + width)
            y1 = int(top + height)

            cv2.rectangle(image, (x0, y0), (x1, y1), GREEN, BBOX_LINE_SIZE)

        cv2.imshow("image", image)
        while True:
            key = cv2.waitKey(0)
            if key:
                break

    def predict_image(self, image_path: str):
        image_bytes: bytes = read_and_resize_image(
            image_path=image_path,
            max_byte_size=self.max_byte_size,
        )
        print(f"Predicting on image {image_path}")

        # Send image and get back the prediction results
        results = self.predictor.detect_image(
            self.project.id, self.publish_iteration_name, image_bytes
        )
        return results.predictions


if __name__ == "__main__":
    txt_file_paths = []
    with open(
        os.path.join(os.path.dirname(__file__), "../../../dataset/test.txt"), "r"
    ) as f:
        for line in f:
            line = line.strip()
            txt_file_paths.append(line)

    dataset_path = os.path.join(os.path.dirname(__file__), "../../../dataset")

    results = {}

    predict_images = PredictImages()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    for image_name in txt_file_paths:
        image_path = os.path.join(dataset_path, "images", f"{image_name}.JPG")
        if not os.path.exists(image_path):
            raise FileExistsError(f"File {image_path} does not exist")
        predictions = predict_images.predict_image(image_path=image_path)
        predict_images.show_prediction(image_path, predictions)
        results[image_name] = []
        for prediction in predictions:
            if prediction.probability < PREDICTION_THRESHOLD:
                continue
            results[image_name].append(
                BoundingBox(
                    label=prediction.tag_name,
                    left=prediction.bounding_box.left,
                    top=prediction.bounding_box.top,
                    width=prediction.bounding_box.width,
                    height=prediction.bounding_box.height,
                )
            )
    cv2.destroyAllWindows()
    file_path = os.path.join(
        os.path.dirname(__file__), "../comparison/results/custom_vision_results.json"
    )
    with open(file_path, "w") as f:
        json.dump(results, f)

    print(results)

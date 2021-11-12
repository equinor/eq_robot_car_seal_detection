import argparse
import os

from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
    Project,
    Region,
    Tag,
)
from car_seal.config import TRAINING_ENDPOINT, TRAINING_KEY
from car_seal.custom_vision.reader import read_and_resize_image
from msrest.authentication import ApiKeyCredentials

parser = argparse.ArgumentParser(description="Upload dataset to custom vision project.")
parser.add_argument(
    "--project_name",
    metavar="N",
    type=str,
    default="car_seal_test",
    help="Name of the custom vision project",
)


class UploadDataset:
    def __init__(self, files_to_upload: list, project_name: str) -> None:
        self.files_to_upload = files_to_upload

        credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
        self.trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)

        self.project_name = project_name

        self.max_byte_size = 4000000

        self.project: Project = self._connect_to_or_create_project(
            project_name=self.project_name
        )
        # Make two tags in the new project
        self.green_car_seal_tag = self._get_or_create_tag("green_car_seal")
        self.red_car_seal_tag = self._get_or_create_tag("red_car_seal")
        self.label_to_tag_id = {
            0: self.red_car_seal_tag.id,
            1: self.green_car_seal_tag.id,
        }

    def _connect_to_or_create_project(self, project_name: str) -> Project:
        projects = self.trainer.get_projects()
        project_id = next((p.id for p in projects if p.name == project_name), None)

        if project_id is not None:
            print("Connecting to existing project...")
            return self.trainer.get_project(project_id)

        print("Creating new project...")
        obj_detection_domain = next(
            domain
            for domain in self.trainer.get_domains()
            if domain.type == "ObjectDetection" and domain.name == "General"
        )
        return self.trainer.create_project(
            project_name, domain_id=obj_detection_domain.id
        )

    def _get_or_create_tag(self, tag_name) -> Tag:
        tags = self.trainer.get_tags(self.project.id)
        for tag in tags:
            if tag.name == tag_name:
                return self.trainer.get_tag(self.project.id, tag.id)

        return self.trainer.create_tag(self.project.id, tag_name)

    def _read_annotation_file(self, annotation_path: str) -> list:
        annotations = []
        with open(annotation_path, "r") as f:

            for line in f:
                line = line.strip()
                parameter_list = line.split(" ")
                label = int(parameter_list[0])
                x, y, w, h = list(map(float, parameter_list[1:]))

                left = x - w / 2
                if left < 0:  # Accounting for previous rounding error
                    left = 0
                top = y - h / 2
                if top < 0:  # Accounting for previous rounding error
                    top = 0

                if left + w > 1:  # Accounting for previous rounding error
                    w = 1 - left
                if top + h > 1:  # Accounting for previous rounding error
                    h = 1 - top

                try:
                    tag_id = self.label_to_tag_id[label]
                except:
                    raise ValueError(f"Wrong label {label} at {annotation_path}")

                annotations.append(
                    Region(
                        tag_id=tag_id,
                        left=left,
                        top=top,
                        width=w,
                        height=h,
                    )
                )
        return annotations

    def main(self) -> None:
        dataset_path = os.path.join(os.path.dirname(__file__), "../../../dataset")

        existing_image_count = self.trainer.get_image_count(project_id=self.project.id)
        file_number = existing_image_count
        self.files_to_upload = self.files_to_upload[file_number:]

        for file_name in self.files_to_upload:
            tagged_images_with_regions = []

            annotations: list = self._read_annotation_file(
                annotation_path=os.path.join(
                    dataset_path, "annotations", file_name + ".txt"
                ),
            )
            image_bytes: bytes = read_and_resize_image(
                image_path=os.path.join(dataset_path, "images", file_name + ".JPG"),
                max_byte_size=self.max_byte_size,
            )
            print(f"Image {file_name} is {len(image_bytes)} bytes")
            tagged_images_with_regions.append(
                ImageFileCreateEntry(
                    name=file_name, contents=image_bytes, regions=annotations
                )
            )
            print("Upload images...")
            upload_result = self.trainer.create_images_from_files(
                self.project.id, ImageFileCreateBatch(images=tagged_images_with_regions)
            )
            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status)
                exit(-1)
            print(
                f"Uploaded file number {file_number+1} of {len(self.files_to_upload)}"
            )
            file_number += 1


if __name__ == "__main__":
    args = parser.parse_args()
    project_name = args.project_name

    txt_file_paths = []
    with open(
        os.path.join(os.path.dirname(__file__), "../../../dataset/train.txt"), "r"
    ) as f:
        for line in f:
            line = line.strip()
            txt_file_paths.append(line)
    with open(
        os.path.join(os.path.dirname(__file__), "../../../dataset/validation.txt"), "r"
    ) as f:
        for line in f:
            line = line.strip()
            txt_file_paths.append(line)

    upload_dataset = UploadDataset(
        files_to_upload=txt_file_paths, project_name=project_name
    )
    upload_dataset.main()

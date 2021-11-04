from os import name

import pytest
from azure.cognitiveservices.vision.customvision.training.models import (
    Project,
    ProjectSettings,
    Tag,
)

from upload_dataset import UploadDataset

mock_project = Project(
    name="mock_project", description="A mock project", settings=ProjectSettings()
)

mock_tag = Tag(name="mock_tag", description="Tag for test", type="For test")


@pytest.mark.parametrize(
    "image_path, max_byte_size",
    [],
)
def test_read_and_resize_image(mocker, image_path, max_byte_size):
    mocker.patch.object(
        UploadDataset, "_connect_to_or_create_project", return_value=mock_project
    )
    mocker.patch.object(UploadDataset, "_get_or_create_tag", return_value=mock_tag)
    upload_dataset = UploadDataset(files_to_upload=[], project_name="test")
    image_bytes = upload_dataset._read_and_resize_image(image_path=image_path)
    assert len(image_bytes) < max_byte_size


@pytest.mark.parametrize(
    "annotation_path",
    [
        "/home/oysand/git/eq_robot_car_seal_detection/test_dataset/1.txt",
        "/home/oysand/git/eq_robot_car_seal_detection/test_dataset/4.txt",
        "/home/oysand/git/eq_robot_car_seal_detection/test_dataset/34.txt",
        "/home/oysand/git/eq_robot_car_seal_detection/test_dataset/459.txt",
        "/home/oysand/git/eq_robot_car_seal_detection/test_dataset/580.txt",
        "/home/oysand/git/eq_robot_car_seal_detection/test_dataset/611.txt",
        "/home/oysand/git/eq_robot_car_seal_detection/test_dataset/1189.txt",
    ],
)
def test_read_annotation_file(mocker, annotation_path):
    mocker.patch.object(UploadDataset, "_get_or_create_tag", return_value=mock_tag)
    upload_dataset = UploadDataset(files_to_upload=[], project_name="test")
    annotations: list = upload_dataset._read_annotation_file(
        annotation_path=annotation_path,
    )
    for annotation in annotations:
        assert annotation.left >= 0
        assert annotation.left + annotation.width <= 1
        assert annotation.top >= 0
        assert annotation.top + annotation.height <= 1

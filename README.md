# eq_robot_car_seal_detection

## Pre-requisites

- Python
- A `.env` file. Created by copying the template and filling in the values.

## Run

```
source venv/Scripts/activate # If using venv
pip install -r requirements.txt
cd src
python main.py --help
```

## Custom vision

- Place the dataset to be uploaded into dataset folder. The images needs to be placed in a
folder called images and the annotations in a folder called annotations.
- Update the test, train and validation files if desired.
- Install the repository using `pip install -e .`
- Run the upload_dataset.py file to upload the dataset to a custom vision project.

## Darknet setup

- Follow the setup from https://github.com/AlexeyAB/darknet
- Use the files from src/cars_seal/darknet
- Test, train and validation is found in the dataset folder

### Train

```
./darknet detector train data/car-seal.data cfg/yolo-car-seal.cfg yolov4.conv.137 -map
```

### Test

```
./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights path_to_image
```

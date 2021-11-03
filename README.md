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
- Run the upload_dataset.py file to upload the dataset to a custom vision project.

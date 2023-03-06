# Productionized Titanic Classification Model Package

## Run with Tox (Recommended)
- Download the data from: https://www.openml.org/data/get_csv/16826755/phpMYEkMl
- Save the file as `raw.csv` in the classification_model/datasets directory
- `pip install tox`
- Make sure you are in the assignment-section-05 directory (where the tox.ini file is) then run the command: `tox` (this runs the tests and typechecks, trains the model under the hood). The first time you run this it creates a virtual env and installs
dependencies, so takes a few minutes.

## Run without Tox
- Download the data from: https://www.openml.org/data/get_csv/16826755/phpMYEkMl
- Save the file as `raw.csv` in the classification_model/datasets directory
- Add assignment-section-05 *and* classification_model paths to your system PYTHONPATH
- `pip3 install -r requirements/test_requirements`
- Train the model: `python classification_model/train_pipeline.py`
- Run the tests `pytest tests`


## Commands 
  https://packaging.python.org/en/latest/tutorials/packaging-projects/

  - For Windows
    - py -m build

  - For Macos
    python3 -m pip install --upgrade build
    python3 -m build
    



`tox` is used to run 
`tox -e test_package` to run the project.
[Output](./output/output.png)


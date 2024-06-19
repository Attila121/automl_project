# AutoML Exam - SS24 (Tabular Data)
This repo serves as a template for the exam assignment of the AutoML SS24 course
at the university of Freiburg.

The aim of this repo is to provide a minimal installable template to help you get up and running.

## Installation

To install the repository, first create an environment of your choice and activate it. 

You can change the Python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-tabular-env
source automl-tabular-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-tabular-env python=3.11
conda activate automl-tabular-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:
```bash
python -c "import automl"
```

We make no restrictions on the Python library or version you use, but we recommend using Python 3.8 or higher.

## Code
We provide the following:

* `download-openml.py`: A script to download the dataset from openml given a `--task`, corresponding to an OpenML Task ID. We wil provide those suggested as training datasets, prior to us releasing the test dataset.

* `run.py`: A script that loads in a downloaded dataset, trains an _AutoML-System_ and then generates predictions for
`X_test`, saving those predictions to a file. For the training datasets, you will also have access to `y_test` which
is present in the `./data` folder, however you **will not** have access to `y_test` for the test dataset we provide later.
Instead you will generate the predictions for `X_test` and submit those to us through github classrooms.

* `./src/automl`: This is a python package that will be installed above and contain your source code for whatever
system you would like to build. We have provided a dummy `AutoML` class to serve as an example.

**You are completely free to modify, install new libraries, make changes and in general do whatever you want with the
code.** The only requirement for the exam will be that you can generate predictions for `X_test` in a `.npy` file
that we can then use to give you a test score through github classrooms.

## Data

### Practice datasets:

* [y_prop_4_1 (361092)](https://www.openml.org/search?type=task&id=361092&collections.id=299&sort=runs)
* [Bike_Sharing_Demand (361099)](https://www.openml.org/search?type=task&id=361099&collections.id=299&sort=runs)
* [Brazilian Houses (361098)](https://www.openml.org/search?type=task&id=361098&collections.id=299&sort=runs)

### Final Test dataset:

```
TBA
```

You can download the required OpenML data using:
```bash
python download-openml.py --task <task_id>
```

This will by default, download the data to the `/data` folder with the following structure.
The fold numbers, `1, ..., n` are **outer folds**, meaning you can treat each one of them as
a seperate dataset for training and validation. You can use the `--fold` argument to specify which fold you would like.

```bash
./data
├── 361092
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 2
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 3
    ...
├── 361098
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
    ...
```

When we provide you the test data, you can unzip it and place it in the `/data` folder with the following structure.
There will be only one fold, `1`, for the test dataset.
```bash
./data
├── exam-dataset
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   └── y_train.parquet
```

**Note:** You will not have access to `y_test` for the test dataset, only `X_test`.


## Running an initial test
This will train a dummy AutoML system and generate predictions for `X_test`:
```bash
python run.py --task brazilian_houses --seed 42 --output-path preds-42-brazil.npy
```

You are free to modify these files and command line arguments as you see fit.

## Final submission

The final test predictions should be uploaded in a file `final_test_preds.npy`, with each line containing the predictions for the input in the exact order of `X_test` given.

Upload your poster as a PDF file named as `final_poster_tabular_<team-name>.pdf`, following the template given [here](https://docs.google.com/presentation/d/1lyE-iLGXIKi31CLFwueGhjfcsR_8r7_L/edit?usp=sharing&ouid=107220015291298974152&rtpof=true&sd=true).

## Reference performance

| Dataset | Final Test performance |
| -- | -- |
| y_prop | 0.87 |
| bike_sharing | 0.935 |
| brazilian_houses | 0.969 |
| final test dataset | TBA |

## Tips
* If you need to add dependencies that you and your teammates are all on the same page, you can modify the
`pyproject.toml` file and add the dependencies there. This will ensure that everyone has the same dependencies

* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
predictions, etc. Also, be friendly teammate and ignore your virtual environment and any additional folders/files
created by your IDE.

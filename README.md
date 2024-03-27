# Terrain Generator ML

A suite of ML tools and models for generating stylish earth-like terrain heightmaps using real earth data sets

# Run

https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944

`pip install -e src`

.
├── .data
│ ├── processed
│ │ ├── test.csv
│ │ └── train.csv
│ └── raw
│ └── data.csv
|
├── .experiments
│ └── model1
│ ├── version_0
| | └── ...
| └── ...
└── src
|
└── ml
├── data
│ ├── make_dataset.py
│ └── preprocessing.py
|
├── datasets
│ ├── dataset1
│ | ├── datamodule.py
│ | └── dataset.py
| └── ...
|
├── engines
│ └── system.py
|
├── models
│ ├── model1.py
│ └── model2.py
|
├── scripts
│ ├── predict.py
│ ├── test.py
│ └── train.py
|
└── utils
├── constants.py
└── helpers.py

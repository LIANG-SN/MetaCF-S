# MetaCF & Score-aware MetaCF

***Some of the codes are adopted from [Reference](https://weitianxin.github.io/)***

## Requirements

### Packages

- python==3.6.3
- pytorch==1.1.0
- numpy==1.17.0
- scikit-learn==0.24.2

### Environment

- Ubuntu, VERSION="16.04.7 LTS (Xenial Xerus)

## How to run

### Training

- `train.sh`
  - directly run to train our score-aware MetaCF model
  - uncomment `--original_model`,  `--dot_prod` to run the original model
  - change `--epoch` to load saved model
  - other configs' discriptions are in help information

### Evaluation

- `evaluate.sh`:
  - directly run to evaluate our score-aware MetaCF model
  - uncomment `--original_model`,  `--dot_prod` to evaluate the original model
  - change `--epoch` to load saved model
  - other configs' discriptions are in help information
  - we early stop after 2 epoches, and we fix the rank prediction after the first epoch

## Details of each script

### `dataset.py`

- Initialize and load the original kindle dataset and our movielens dataset
- Subgraph sampling with/without probability

### `main.py`

- Training and evaluation procedure

### `model.py`

- The model of GCN

### self-use test scripts

- `read_log.py`
- `testcode.py`

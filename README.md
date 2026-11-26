This repository contains the material for the study on: **Automatic Generation of Fine-Grained Source Code Templates using Language Models**.


### Setup

Pre-requests:

python == 3.9.2\
tqdm == 4.64.0\
treelib == 1.6.1\
numpy == 1.21.6\
torch == 1.12.0+cu113\
pytorch_lightning == 1.6.3


### Get dataset & prepare folds
Download the preprocessed java dataset from: [Link](https://www.sciencedirect.com/science/article/abs/pii/S0164121222002291)

To prepare train, dev and test sets run:

`python data_splits.py -i data/java -o data/splits/java`


### Model Training 
To train the model run:

`python trainer.py -i data/splits/java -o saveModels/java -m ActiveTrans or ActiveTransPlus`

### Model Testing
To evaluate the model run:

`python evaluator.py -d data/splits/java -i saveModels/java -m ActiveTrans or ActiveTransPlus`





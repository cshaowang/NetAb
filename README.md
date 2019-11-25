## NetAb
### This repo hosts the code for paper "Learning with Noisy Labels for Sentence-level Sentiment Classification" (in EMNLP-2019).


## Python Development Environment
- Python 3.6.7
- MongoDb
- pymongo
- tensorflow-gpu 1.9.0

## Installation
1. Download the project NetAb;
2. Unzip the downloaded project;

Then the project is organized as follows

    ├── .idea                 <- IntelliJ’s project specific settings files
    ├── Data
    │   ├── TestSens          <- Clean-labeled test sentences
    │   ├── TrainingSens      <- Noisy-labeled train sentences
    │   ├── ValSens           <- Clean-labeled validation sentences
    │   └── word2id           <- Word to index
    │
    ├── model                 <- Network functions
    ├── config.py             <- Configuration information
    ├── create_w2v_mongo.py   <- To create a word2vectors with mongodb
    ├── data_helper.py        <- Utilities
    ├── main.py               <- Main function
    ├── README.md             <- Guide for user(s) to perform this project.

## Usage
1. Download the pre-trained word vectors GloVe.840B.300d; and then place it to the folder ./data/;
2. Run create_w2v_mongo.py to create a mongodb version for the GloVe.840B.300d;
3. Run main.py to produce the sentention classification results on each dataset, (e.g., python -m main -dataset 'movie').

Any questions, please let me know. Thanks!

Hao WANG
Email: cshaowang@gmail.com

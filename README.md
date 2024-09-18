# A Comparative Analysis of Text-Based Explainable Recommender Systems

This is our Pytorch implementation of the paper:

Alejandro Ariza-Casabona, Ludovico Boratto, and Maria Salamó. 2024. A
Comparative Analysis of Text-Based Explainable Recommender Systems. In
18th ACM Conference on Recommender Systems (RecSys ’24), October 14–18,
2024, Bari, Italy.

🚧🚧 **We are currently working on an extended Python library that is quickly installable through pip with more SOTA models, pipeline configurations, and easily extensible to new external models** 🚧🚧


## Datasets


The preprocessed and original datasets used in this project can be downloaded at the following [link](https://drive.google.com/file/d/1Xmn0RVx1bXRNkNNeSBON-8WoIBE1KytC/view?usp=sharing).

The selected datasets were Yelp, TripAdvisor and RateBeer, but the code to extract explanations `extract_exps.py` and process the datasets `process_data.py` is also provided.

Additionally, `process_data.py` contains the source code to obtain the auxiliary data required by GREENer which is really time consuming and must be run beforehand.

## Usage


The main source code is under the `src/` package, including model implementations, trainer, datasets, utilities, etc.

0. Under `src/utils/constants.py` make sure the paths are correct.
1. Update the `scripts/single.sh` arguments to run a single experiment.
2. Give necessary permissions to the script file: `chmod +x scripts/single.sh`
3. Run in background: `nohup ./scripts/single.sh > log.out &`

The `main.py` python script will be executed and there is a variety of available arguments to configure the experiment, such as resuming the training or evaluating an already trained model.

The configuration of each model that was used in our comparison is under the `config` folder and each model was run for 4 different seeds (1111, 24, 53, and 126).

## Dependencies


The Python (3.10.13) packages we used to run the code can be installed via `conda env create -f environment.yml`. Next, activate the new conda enviroment `conda activate text_exp`.

# Dynamic Conditional Optimal Transport through Simulation-Free Flows

This repo contains an implementation of COT-FM, as described in our publication [Dynamic Conditional Optimal Transport through Simulation-Free Flows](https://arxiv.org/abs/2404.04240).

Our code is heavily based on the (very nice) [torchcfm](https://github.com/atong01/conditional-flow-matching) package, which is required as a dependency.

## Environment

Clone the repo and create a conda environment:

`conda env create -f environment.yml`

which can be activated with

`conda activate cotfm`.

## Usage

The file `cotfm/trainer.py` contains a basic script for training and tuning models on the datasets considered in the paper.

To launch training (using a random search over hyperparameters), run e.g.

`python ./cotfm/trainer.py --savedir ./2moons/ --dataset 2moons --cotmode batch`

changing the `--dataset` as necessary. 

## Citation

If you found our paper or code useful, please consider citing our work as follows:

```
@misc{kerrigan2024dynamic,
      title={Dynamic Conditional Optimal Transport through Simulation-Free Flows}, 
      author={Gavin Kerrigan and Giosue Migliorini and Padhraic Smyth},
      year={2024},
      eprint={2404.04240},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
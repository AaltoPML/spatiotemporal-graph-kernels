# Non-separable Spatio-temporal Graph Kernels via SPDEs

This repository is the official implementation of the methods in the publication
* Alexander Nikitin, ST John, Arno Solin, and Samuel Kaski (2022). **Non-separable spatio-temporal graph kernels via SPDEs**. In *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)*. [[arXiv]](https://arxiv.org/abs/2111.08524) 

<p align="center">
  <img src="data/gp_on_graphs_teaser.png" />
</p>


We leverage an explicit link between stochastic partial differential equations (SPDEs) and Gaussian processes on graphs and derive non-separable spatio-temporal graph kernels that capture interaction across space and time. We formulate the graph kernels for the stochastic heat equation and wave equation. We show that by providing novel tools for spatio-temporal GP modelling on graphs, we outperform pre-existing graph kernels in real-world applications that feature diffusion, oscillation, and other complicated interactions.

## Use
The repo uses [git-lfs](https://git-lfs.github.com/) to store datasets. To fetch the data use:
```bash
git lfs fetch
```

The code was tested with `python==3.6` and should work for `python>=3.6`.

To install the required packages, run:
```bash
pip install -r requirements.txt
pip install -e .
```

## Structure
The repository contains two sets of kernels for time-independent and temporal processes on graphs.
* Time-independent kernels are stored in `graph_kernels/kernels.py`.
* Temporal kernels are stored in `graph_kernels/time_kernels.py`.
* SHEK and SWEK are implemented in `graph_kernels/time_kernels.py:StochasticHeatEquation` and `graph_kernels/time_kernels.py:StochasticWaveEquationKernel`.

## Experiments.
We provide an experimental evaluation of the proposed kernels on several datasets.

### Heat Transfer Dataset
#### Interpolation:
```bash
python experiments/1d_experiments.py --interpolation --dump_directory=$PATH_TO_RESULTS
```

#### Extrapolation:
```bash
python experiments/1d_experiments.py --extrapolation --dump_directory=$PATH_TO_RESULTS
```

### Chickenpox experiments
#### Interpolation (103 + num_test_weeks):
```bash
python experiments/run_chicken_pox.py --num_test_weeks=2 --interpolation --dump_directory=$PATH_TO_RESULTS
```

#### Extrapolation:
```bash
python experiments/run_chicken_pox.py --num_test_weeks=2 --extrapolation --dump_directory=$PATH_TO_RESULTS
```


### Covid19 Experiments
#### Interpolation (33 + num_test_weeks):
```bash
python experiments/run_covid_experiments.py --log_target --no-use_flight_graph \
        --no-use_normalized_target --num_test_weeks=2 --interpolation --dump_directory=$PATH_TO_RESULTS
```

#### Extrapolation:
```bash
python experiments/run_covid_experiments.py --log_target --no-use_flight_graph \
        --no-use_normalized_target --num_test_weeks=2 --interpolation --dump_directory=$PATH_TO_RESULTS
```


### Wave Experiments:
Open with jupyter-notebook:
```bash
./experiments/1d_wave_experiments.ipynb
```

## Citation
If you use the code in this repository for your research, please cite the paper as follows:
```bibtex
@inproceedings{nikitin2022non,
  title={Non-separable spatio-temporal graph kernels via SPDEs},
  author={Nikitin, Alexander V and John, ST and Solin, Arno and Kaski, Samuel},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={10640--10660},
  year={2022},
  organization={PMLR}
}
```

## Contributing
For all correspondence, please contact alexander.nikitin@aalto.fi.

## License
This software is provided under the [Apache License 2.0](LICENSE).

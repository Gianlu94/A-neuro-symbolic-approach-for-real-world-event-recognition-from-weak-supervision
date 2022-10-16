# A neuro-symbolic approach for real-world event recognition from weak supervision
refactoring (ongoing)

<!-- GETTING STARTED -->
## Getting Started

TODO: update instructions

1) Install Minizinc:

    https://www.minizinc.org/doc-2.5.5/en/installation.html
    
2) Create the environment containing all the dependencies and activate it:
```sh
  conda env create -f environment.yml
  conda activate ns-env
  ```

3) Download train and test features and put them in datasets/MultiTHUMOS/features/

    https://drive.google.com/drive/folders/1txv4OyMd88ku3nzWAeYVhJ-9YR8NHE8w?usp=sharing

<!-- USAGE EXAMPLES -->
## Usage

#### Run Experiment 1 
###### Minizinc:
```python
python main_exps.py -path_to_conf configurations/conf_exp1_mnz.json
  ```
###### Neural (baseline):
```sh
python main_exps.py -path_to_conf configurations/conf_exp1_neural.json
  ```
#### See visual results of atomic actions predictions

1) Put `epochs_predictions.pickle` (created in *`/logs/<exp_id>/<exp_name>/`*) in *`/visual_results/`*

2) Move into *`/visual_results/`* and run:

```sh
python get_plots.py -pickle_file <path_to_epochs_prediction.pickle> -path_to_plots <path_where_to_save_figures> -mode <mode> exp <exp>
  ```
replace **mode** with val or test and **exp** with mnz or neural



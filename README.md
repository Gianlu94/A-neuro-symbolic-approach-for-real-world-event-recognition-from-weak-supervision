# NSA_framework
A neuro-symbolic framework for structured event recognition using Allen's interval algebra

<!-- GETTING STARTED -->
## Getting Started

1) Install Minizinc:

    https://www.minizinc.org/doc-2.5.5/en/installation.html
    
2) Create the environment containing all the dependencies and activate it:
```sh
  conda env create -f environment.yml
  conda activate ns-env
  ```
  
3) Download pretrained MLAD model on MultiTHUMOS dataset and put it in mlad/models/pretrained_models/:

    https://drive.google.com/file/d/1vXq-y68hC4Qe6N1PBk3DlqGjOWhP9Vsc/view?usp=sharing

4) Download train and test features and put them in datasets/MultiTHUMOS/features/

    https://drive.google.com/drive/folders/1txv4OyMd88ku3nzWAeYVhJ-9YR8NHE8w?usp=sharing

<!-- USAGE EXAMPLES -->
## Usage

To evaluate a pre-trained mlad model:
```sh
python evaluate_pretrained_mlad.py --path_to_model ./mlad/models/pretrained_models/model_to_load.pth --epoch epoch_of_the_model --path_to_conf ./configurations/conf_to_use.json
  ```
To start train:
```sh
python main.py --path_to_model ./mlad/models/pretrained_models/MultiTHUMOS_5Layers.pth --path_to_conf ./configurations/conf_1.json --path_to_mzn ./minizinc/models/ --path_to_data ./datasets/Multi-THUMOS/se_events/
  ```

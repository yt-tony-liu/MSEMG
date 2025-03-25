# MSEMG
This repository provides the source code for the implementation of "MSEMG: Surface Electromyography Denoising with a Mamba-based Efficient Network". 

<p align='center'>
<td style='text-align:center;'>
  <img src=img/Model.png >
</td>
</p>

We developed this repo with `python=3.10` and `pytorch=1.12`. You can reproduce our experiment by following the steps:
- Construct The Environment
- Prepare The Dataset
- Run Training
- Run Testing

## Construct The Environment
### Clone the repository
Clone our repository by running the following command and entering the directory. This will be your working directory.
```
git clone https://github.com/yt-tony-liu/MSEMG.git
cd MSEMG
```
### Install the Python packages
```
pip install -r requirements.txt
```
## Prepare The Dataset
### Download the sEMG signals
We use the surface electromyography signals from [Non-Invasive Adaptive Prosthetics (NINAPro)](https://ninapro.hevs.ch/instructions/DB2.html) DB2. The website doesn't provide an all-in-one compressed file of the database, so you might need to download data points from every subjects separately and unpack them into 1 folder shown as follows:
```
[EMG_corpus_dir]
|-- DB2_s1
|   |-- S1_E1_A1.mat
|   |-- S1_E2_A1.mat
|   `-- S1_E3_A1.mat
|-- DB2_s2
|   |-- S2_E1_A1.mat
|   |-- S2_E2_A1.mat
|   `-- S2_E3_A1.mat
...
```
### Download the ECG signals 
The ECG signals for simulating interference can be downloaded from [MIT-BIH Normal Sinus Rhythm Database](https://www.physionet.org/content/nsrdb/1.0.0/). You should specify the path to the folder containing all signals for the later implementation.
```
[ECG_corpus_dir]
|-- 16265.atr
|-- 16265.dat
|-- 16265.hea
|-- 16265.xws
|-- 16272.atr
|-- 16272.dat
|-- 16272.hea
|-- 16272.xws
...
```
### Prepare the config file
You can specify the paths to the corresponding directories in the `data_cfg.yaml` in `/cfg`. Note that the `ECG_storage_dir` is the directory you would like to store the process ECG signals, the `sEMG_dataset_dir` is the directory to the dataset for later use, and the `result_dir` is the directory for your experiment results.
```
EMG_corpus_dir: [directory to downloaded NINAPro DB2]
ECG_corpus_dir: [directory to downloaded MIT-BIH NSRD]
ECG_storage_dir: [directory to store ECG for simulating interference]
sEMG_dataset_dir: [sEMG dataset directory]
result_dir: [your result directory]
```
### Data preprocessing
To preprocess all downloaded data and prepare the dataset, you need to run the following after you have specified all paths in the `local_cfg.yaml`.
```
python preprocess.py
```
## Run Training
You need to train SDEMG before you perform denoising. You can train SDEMG on a single GPU or multiple GPUs. Note that you can specify the number of workers in `data_cfg.yaml`, the recommended number is half of your CPU cores on your device. Also, adjust the experiment setting and the hyperparameters (e.g. batch size) in `cfg/default.yaml`
### Single GPU Training
```
python main.py --train
```
### Multiple GPU Training
SDEMG supports multiple GPU training with [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). You need to install `accelerate` by running `pip install accelerate` and adjust the configuration for your device by running `accelerate config`. 
```
accelerate launch main.py --train
```
## Run Testing
You can run testing and find the denoise result in your project directory. 
```
python main.py --test
```
## Inference 
You can fill in the path to the files you would like to run inference in `line 72` of `main.py`.
```
72      file_paths = ['demo file paths']
```
SDEMG will run a single inference on the files.
```
python main.py --sample
```

## Citing our work
Y. -T. Liu, K. -C. Wang, R. Chao, S. M. Siniscalchi, P. -C. Yeh and Y. Tsao, "MSEMG: Surface Electromyography Denoising with a Mamba-based Efficient Network," ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Hyderabad, India, 2025, pp. 1-5, doi: 10.1109/ICASSP49660.2025.10887547.



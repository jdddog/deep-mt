# DeepMT
Prediction of outcome of Mechanical Thrombectomy in endovascular stroke patients.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://img.shields.io/badge/python-3.8-blue)
![Unit tests](https://github.com/jdddog/deep-mt/workflows/Unit%20Tests/badge.svg)

## 1. Installation
See below for installation instructions.

### 1.1 OS pre-requisites
* Ubuntu 22.04 or above.
* pip: https://pypi.org/project/pip/.
* Python 3.10. See https://www.linuxcapable.com/how-to-install-python-3-10-on-ubuntu-linux/ for instructions on how to
install Python 3.10 on Ubuntu 22.04.
* virtualenv 20 or greater.
  * To install: `pip install --upgrade virtualenv`.
  * To check your version run: `virtualenv --version`.
* PyTorch 1.13.1.
* CUDA 11.7.

### 1.2 Setup Instructions
Install deep-mt in a Python virtual environment to prevent conflicts with other packages.

Clone project.
```bash
git clone git@github.com:jdddog/deep-mt.git
```

Enter the deep-mt folder.
```
cd deep-mt
```

Create a virtual environment.
```
virtualenv -p python3.10 venv
```

Activate your virtual environment.
```
source venv/bin/activate
```

Install the deep-mt package and it's dependencies.
```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
```

## 2. Data Pre-processing
See below for data pre-processing instructions.

Your data folder should end up arranged like this:
* data
  * configs: config files.
  * experiments: files saved during training, e.g. weights.
  * nii
     * STKCentreA: NIfTI files for centre A.
     * STKCentreB: NIfTI files for centre B.
  * thrombectomy-example.csv: sample CSV file with fake data.

**Make sure that you copy your CSV data files into the data folder. `thrombectomy-example.csv` shows how
these file should be structured.**

### 2.1. Install dependencies
For pre-processing install the required dependencies:
* Install R: https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-22-04
* Install R Studio:
  * Install Ubuntu dependencies: `sudo apt install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev`
  * RStudio download location: https://www.rstudio.com/products/rstudio/download/
* Install FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
* In RStudio, run the command `bin/preprocess/install.R`
* Install the `deep-skull` Python project into a virtual environment: https://github.com/jdddog/deep-skull

### 2.2. Convert DICOMs to NIfTI format
Run the command convert a batch of DICOM CT scans to NIfTI:
```bash
./bin/preprocess/dcm2nii-batch.sh /path/to/dicoms /path/to/nii ax_CT
```

Run the command convert a batch of DICOM CT angiogram scans to NIfTI:
```bash
./bin/preprocess/dcm2nii-batch.sh /path/to/dicoms /path/to/nii ax_A
```

### 2.3. Pre-process NIfTI files: skull stripping and registration
Run the preprocess script from the `./bin/preprocess` directory:
```bash
cd ./bin/preprocess
./preprocess.sh -t ./templates -n ../data/nii -d ../../deep-skull -x 1.0 -y 1.0 -z 2.0
```

### 2.4. Calculate CSF
Calculate CSF of the scans:
```bash
deep-mt calc-csf ./data/thrombectomy-example.csv ./data/nii/ ./data/thrombectomy-example-csf.csv
```

Merge the `brain_volume`, `csf_volume` and `csf_ratio` columns into your CSV file.

## 3. Training & Evaluating PyTorch Models
To train a PyTorch model run the following command.
```bash
deep-mt train-pytorch ./data/configs/imaging/sex/sex-ct.yaml
```

To evaluate a batch of weights, run the following command. By default, the valid and test sets is used for evaluation:
```bash
deep-mt evaluate-pytorch ./data/configs/imaging/sex/sex-ct.yaml
```

To evaluate with the train, valid and test datasets, use the `--subset` option:
```bash
deep-mt evaluate-pytorch ./data/configs/imaging/sex/sex-ct.yaml --subset train --subset valid --subset test
```

To start tensorboard:
```bash
tensorboard --logdir runs/
```

## 4. Feature Selection & Training Scikit Learn Models
To train a scikit-learn model run the following command.
```bash
deep-mt train-sklearn ./data/configs/clinical/logistic-regression-mrs02-36.yaml
```

To run feature selection run the following command. This uses SequentialFeatureSelector with RepeatedStratifiedKFold
to select a subset of features and then trains and evaluates two models, one with the subset of features and one
with all features. The Sklearn model is saved to the experiment folder and evaluation results saved to CSV.
```bash
deep-mt feature-selection ./data/configs/clinical/logistic-regression-mrs02-36.yaml
```

## 5. Visualise
To visualise the transformed scans, run the following command. By default, all scans from all subsets are visualised.
```bash
deep-mt visualise ./data/configs/imaging/sex/sex-ct.yaml
```

To visualise the transformed scans for a specific subset and to only visualise a maximum number of cases, specify
the subset and n-cases option. Multiple subsets can be specified by repeating the --subset option.
```bash
deep-mt visualise ./data/configs/imaging/sex/sex-ct.yaml --subset valid --n-cases 5
```

By default, every third slice is visualised, to visualise them all, set `--every-n 1`.

The visualise command will output the location of the visualisations.

## 6. Visualise Salience
To visualise the salience of a model, run the following command. By default, all scans from all subsets are visualised.
```bash
deep-mt visualise-salience ./data/configs/imaging/sex/sex-ct.yaml ./data/experiments/sex-ct/sex-ct_epoch_1.pth --salience-type gradcam
```

To visualise the salience for a specific subset and to only visualise a maximum number of cases, specify
the subset and n-cases option. Multiple subsets can be specified by repeating the --subset option.
```bash
deep-mt visualise-salience ./data/configs/imaging/sex/sex-ct.yaml ./data/experiments/sex-ct/sex-ct_epoch_1.pth --salience-type gradcam  --subset valid --n-cases 5
```

By default, every third slice is visualised, to visualise them all, set `--every-n 1`.

The visualise-salience command will output the location of the visualisations.

## 7. Notes
### Data Cleaning (when adding new scans)
A couple of commands come in handy to perform data cleaning, when adding new scans to the dataset.

To merge new DICOM scans with an existing dataset, run the `merge-scans` command.
```bash
deep-mt merge-scans /path/to/src/dicoms /path/to/dst/dicoms
```

To check that DICOM scans are readable, run the `check-scans-readable` command and then fix any issues.
```bash
deep-mt check-scans-readable /path/to/dicoms
```

To find duplicate DICOM scans, even if the scans have different case ids or series ids, use the `find-duplicate-scans` command.
```bash
deep-mt find-duplicate-scans /path/to/dicoms
```

### Convert Scans to NIfTI
You may need to clean the DICOMs with gdcmanon first:
```bash
gdcmanon -r --continue --dumb --remove 0008,0020 --remove 0008,0030 -i /path/to/dicoms -o /path/to/output
```

### Crop and resample:
Example commands to crop scans:
```bash
deep-mt crop /path/to/nii/ "^STK(?:CH)?\d+[_]?\d*_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid.nii.gz$" 39 45 52 372 460 110
deep-mt crop /path/to/nii/ "^STK(?:CH)?\d+[_]?\d*_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid_combined_bet.nii.gz$" 39 45 52 372 460 110
deep-mt crop /path/to/nii/ "^STK(?:CH)?\d+[_]?\d*_ax_A_cropped_to_STK(?:CH)?\d+[_]?\d*_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid.nii.gz$" 39 45 52 372 460 110
```

Example commands to resample scans:
```bash
deep-mt resample /path/to/nii/ "^STK(?:CH)?\d+[_]?\d*_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid_combined_bet_crop.nii.gz$" --x 0.44 --y 0.44 --z 1.5
deep-mt resample /path/to/nii/ "^STK(?:CH)?\d+[_]?\d*_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid_crop.nii.gz$" --x 0.44 --y 0.44 --z 1.5
deep-mt resample /path/to/nii/ "^STK(?:CH)?\d+[_]?\d*_ax_A_cropped_to_STK(?:CH)?\d+[_]?\d*_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid_crop.nii.gz$" --x 0.44 --y 0.44 --z 1.5
```

### Predict
To make predictions for a given weights file:
```bash
deep-mt predict-pytorch ./data/configs/imaging/mrs/mrs-ct-1.0x1.0x2.0mm-152x182x76px.yaml ./data/experiments/mrs-ct-1.0x1.0x2.0mm-152x182x76px/mrs-ct-1.0x1.0x2.0mm-152x182x76px_epoch_33.pth
deep-mt predict-pytorch ./data/configs/imaging/mrs/mrs-ct-1.0x1.0x2.0mm-152x182x76px-no-basilars.yaml ./data/experiments/mrs-ct-1.0x1.0x2.0mm-152x182x76px-no-basilars/mrs-ct-1.0x1.0x2.0mm-152x182x76px-no-basilars_epoch_59.pth
```

## License
The Python code is licensed under Apache License Version 2.0.

The bash and R based pre-processing scripts in `./bin/preprocess` are licensed the GPLv3 due to dependencies
used in these scripts.
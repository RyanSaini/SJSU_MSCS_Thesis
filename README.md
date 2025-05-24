# SJSU MSCS Thesis Source Code
This repo contains the source code for Ryan Saini's masters thesis at SJSU. 

## Setup Instructions

### 1. Create Virtual Environment

First, create and activate a virtual environment using the provided requirements file:

```bash
# Create virtual environment
python -m venv thesis_env

# Activate virtual environment
# On Windows:
thesis_env\Scripts\activate
# On macOS/Linux:
source thesis_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Repository Structure

The repository is organized into two main experiment directories:

```
├── Experiment1/
│   ├── CIFAR10_resNet.py
│   ├── CIFAR10_csnn.py
│   └── data/                    # Created after first run of either file
├── Experiment2/
│   ├── DVS_cnn.py
│   ├── DVS_csnn.py
│   └── DVSGesturedataset/       # Created after first run of either file
├── requirements.txt
└── README.md
```

## Running Experiments

### Important: Directory Requirements

**All scripts must be run from within their respective experiment directories.** This ensures proper relative path handling and dataset management.

### Experiment 1

```bash
# Navigate to Experiment 1 directory
cd Experiment1

# Run the CIFAR-10 ResNet experiment
python CIFAR10_resNet.py

# Run the CIFAR-10 csnn experiment
python CIFAR10_csnn.py
```

### Experiment 2

```bash
# Navigate to Experiment 2 directory
cd Experiment2

# Run the DVSGesture cnn experiment 
python DVS_cnn.py

# Run the DVSGesture csnn experiment 
python DVS_csnn.py
```

## Dataset Information

- **Automatic Download**: Datasets will be automatically downloaded when running scripts for the first time
- **Directory Creation**: Upon first run, dataset directories will be created:
  - `Experiment1/data/` - Contains CIFAR-10 and other datasets for Experiment 1
  - `Experiment2/DVSGesturedataset/` - Contains DVS Gesture dataset for Experiment 2
- **Preprocessing**: Data preprocessing occurs during the initial run
- **Caching**: Subsequent runs will use cached datasets for faster execution
- **Important**: Do not move dataset files after initial download to maintain cache functionality


## Dataset Citations
This work uses the following datasets:

CIFAR-10
Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. 
Technical Report, University of Toronto.

DVS Gesture Dataset
Amir, A., Taba, B., Berg, D., Melano, T., McKinstry, J., Di Nolfo, C., ... & Modha, D. S. (2017). 
A low power, fully event-based gesture recognition system. 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 7243-7252).

## License
Copyright © 2025 Ryan Saini
This code is provided for academic and research purposes. Please cite this work if you use any part of it in your research.

## Notes

- Ensure your virtual environment is activated before running any experiments
- Each experiment should be run from its respective directory
- First-time execution may take longer due to dataset download and preprocessing
- Keep dataset files in their original locations to utilize caching

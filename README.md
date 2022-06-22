# COMP472-Project

Project done for COMP 472: Artificial Intelligence in the Summer 2022 semester

# Link to the Report

https://docs.google.com/document/d/1HZvHrTKvRzlIwEa-leeImMFPz4dsGyJqNpd1tgH4kl8/edit?usp=sharing

# Team Members:

| Name               | Student ID |
| ------------------ | ---------- |
| Laila Alhalabi     | 40106558   |
| Mohammad Ali Zahir | 40077619   |
| Cheikh Diagne      | 40094098   |
| Marita Brichan     | 40138194   |

# Instructions

## Dataset

- The dataset has two folders: Training and Testing
- Each folder has 4 classes (No Mask, N95 Mask, Surgical Mask and Cloth Mask). The training folder has 300 images in each one of its classes, while the testing folder has 100 images in each one of its classes
- For Phase 2, we further divided the dataset to introduce a bias for the atrributes age (Old or Young) and gender (Male or Female). From the 1200 initial that we had in the training folder, we reduced to 80 each for each attribute in the testing folder

## Running instructions

- The repository contains a main.py file, which is responsible about displaying the statistics of the dataset, training and evaluating the model.
- Clone the repository
- Download PyTorch
- To download torch, create the following environment using Anaconda:
  `$ conda create -n pytorch python=3.6`
- Activate the environment by running
  `$ conda activate pytorch`
- Install the libraries by doing 'pip install -r Requirements.txt'
- run main.py for the Phase 1 Dataset
  `$ python main.py`
- run TrainAge.py for the Age bias in phase 2
  `$ python TrainAge.py`
- run TrainGender.py for the Gender bias in phase 2
  `$ python TrainGender.py`

## Notes

- To run code from phase 1, uncomment the commented code in `$ main.py`, and do the command above
- To run the code without K-Fold for phase 2, uncomment the commented code in `$ TrainAge.py` & `$ TrainGender.py`

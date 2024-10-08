# Traffic Congestion Prediction Based on Classification and Regression Methods

## Project Overview
In this project, we aim to predict traffic congestion using two distinct approaches: classification-based and regression-based methods. For the classification aspect, we have employed Support Vector Classification (SVC) and Random Forest Classifier to categorize traffic conditions. On the other hand, our regression-based approach utilizes Support Vector Regression (SVR), Random Forest Regression, and Recurrent Neural Network with Long Short-Term Memory (RNN-LSTM) to predict the severity and extent of traffic congestion. This comprehensive approach allows for a nuanced analysis of traffic patterns. The entire process, from data preparation to training and testing the models, is meticulously documented and accessible on our GitHub page. This repository serves as a valuable resource for those interested in the intricacies of traffic congestion prediction, providing detailed insights into the methodologies and algorithms employed in our study.

## Repository Contents
- `Data Preparation`: Scripts and notebooks for data cleaning, labelling and preparation.
    - `st_dbscan.py`: Code to cluster data.
    - `coordinates.py`: Converting geographic coordinates (Longitude, Latitude) to UTM coordinates for clustering.
    - `label_data.ipynb`: The notebook to label data using ST_DBSCAN.
    - `label_data.py`: The python file used on `label_data.ipynb`.
- `Classifiers`: The folder includes classifier codes.
    - `RandomForestClassification_train.ipynb`: The notebook to train Random Forest Classification method.
    - `RandomForestClassification_train.ipynb`: The notebook to test Random Forest Classification method.
    - `SupportVectorClassification.ipynb`: The notebook to train and test SVC method.
- `RandomForestRegression.ipynb`: The notebook to train and test Random Forest Regression method.
- `RNN.ipynb`: The notebook to train and test Recurrent Neural Network with Long Short-Term Memory method.

  
## Getting Started
To use this repository for traffic congestion prediction:
1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed (listed in "requirements").
3. If you want to run code to clean data, you should run `Data Preperation/label_data.ipynb`.
4. If you want to run code to train Random Forest Classification, you should run `rfc_classification.ipynb`.
5. If you want to run code to test Random Forest Classification, you should run `rfc_test.ipynb`.
6. If you want to run code to train and test SVC, you should run `svc_classification.ipynb`.
7. If you want to run code to train and test Random Forest Regressor, you should run `rfr.ipynb`.
8. If you want to run code to train and test RNN-LSTM, you should run `RNN.ipynb`.

## Requirements
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.3.2
tensorflow==2.15.0
matplotlib==3.4.3
joblib==1.3.2
torch==2.1.0

## Contributing
Yaşar Mehmet Çelik 150200302
Öykü Eren 150200326


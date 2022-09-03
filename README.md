# Pneumonia Detection from Chest X-ray

We have trained multiple classification models to detect pneumonia from chest X-rays. 

## Dataset
The dataset is divided into 3 parts namely training, validation and testing dataset each of them with two categories of images - Chest X-ray images with Pneumonia and Normal Chest X- ray images. 
Dataset Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Methodology

### Data Preprocessing
All the images were resized and flattened to create a single dataset on which classical ML models can be trained. 
All the images were normalized in range of [0, 1]. 
Some images were flipped, rotated or cropped while feeding to CNN model to avoid overfitting.

### Dimension Reduction
#### PCA
Applied PCA to dimensionally reduce the datset from 4096 features (i.e.64*64) to 200 features capturing 95 percent variance in the dataset.

Percent Variance captured Vs Number of Principal Components:

<img width="512" alt="image" src="https://user-images.githubusercontent.com/70518850/188290426-e5de939b-5a89-4799-bc13-c35cfeec12c4.png">

#### Autoencoder

Autoencoder was trained to dimensionally reduce the datset from 65536 features (i.e.256*256) to 256 features.

Autoencoder Architecture:

 <img width="800" alt="image" src="https://user-images.githubusercontent.com/70518850/188290386-543edbe2-0a5c-42c0-a9c0-db30fb655109.png">

Reconstruction error versus number of epochs:

 <img width="512" alt="image" src="https://user-images.githubusercontent.com/70518850/188290364-5f4e0572-2535-46c7-b73c-c37ddb72aef6.png">

### Model Training

F1 Score graphs for training data and test data were plotted as below:


| Classical ML Model      | PCA Reduced Data | Autoencoder Reduced Data     |
| :---        |    :----:   |          ---: |
| Logistic Regression      | <img width="400" alt="image" src="https://user-images.githubusercontent.com/70518850/188289572-8e91f618-6745-467c-b1a6-638d4c479b0e.png">| <img width="400" alt="image" src="https://user-images.githubusercontent.com/70518850/188289667-057d0ea7-5211-478a-b467-93a088afc700.png">   |
| KNN      | <img width="800" alt="image" src="https://user-images.githubusercontent.com/70518850/188289737-1dac892f-d48f-4cc1-9bc7-23487f60e8ad.png">| <img width="800" alt="image" src="https://user-images.githubusercontent.com/70518850/188289730-ecc18f34-dd13-4147-96e7-53a4eabf5e11.png">  |
| SVM      | <img width="400" alt="image" src="https://user-images.githubusercontent.com/70518850/188289787-a03b33b4-fa09-408d-b23d-a5b57e65d81b.png">| <img width="400" alt="image" src="https://user-images.githubusercontent.com/70518850/188289925-af7708fb-71f8-4c38-ace5-a86768ed861f.png">  |


#### CNN
CNN Architecture:

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/70518850/188289968-a67b18bd-1d9d-4b7a-af6b-2fc5a1d5f508.png">

Cross entropy loss vs epochs:

<img width="512" alt="image" src="https://user-images.githubusercontent.com/70518850/188290018-7a145227-7851-47e9-ae7d-aa2025792808.png">

F1 score vs epochs:

<img width="512" alt="image" src="https://user-images.githubusercontent.com/70518850/188290087-15892dfe-6d8a-477b-afdb-fe80fe763ce3.png">


## Results

F1 score was used as the performance metrics as the data was imbalanced and there is a serious downside to predicting false negatives in such cases. 

| ML Model      | PCA Reduced Data | Autoencoder Reduced Data     |
| :---        |    :----:   |          ---: |
| Logistic Regression      | 0.8355 | 0.9411  |
| KNN      | 0.8437 | 0.8435  |
| SVM      | 0.835 | 0.825 |

| ML Model      | F1 Score |
| :---        |      ---: |
| CNN      | 0.8816 |


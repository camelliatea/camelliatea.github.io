---
title: "Parasite Image Classification"
excerpt: "This project presents a TensorFlow-based image classification model designed to identify three parasite classes (RBCs, Toxoplasma, and Trichomonad) from microscopic images. It showcases the application of deep learning techniques for automating image data analysis, encompassing the complete workflow from data preprocessing and model training to performance evaluation and real-world deployment."
date: 2025-04-21
author_profile: false
---

# Project Overview

This project implements an image classification model using TensorFlow to identify three parasite-related classes (**RBCs**, **Toxoplasma**, and **Trichomonad**) from microscopic images. Developed as part of the final project for the *"Belajar Pengembangan Machine Learning"* course, the project highlights how deep learning can assist in medical diagnostics by automating image analysis.

The development pipeline spans from data preparation and augmentation to model training, evaluation, and deployment across multiple platforms.

Key goals:
- Automate detection of parasitic structures in microscopic images.
- Explore convolutional neural networks (CNNs) for medical image classification.


**Project Details:** [Github](https://github.com/camelliatea/dicoding-image-classification)


### Languages and Tools

- **Language:** Python  
- **Data Handling:** Pandas, NumPy  
- **Image Processing:** Pillow  
- **Modeling & Training:** TensorFlow, Keras, scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Utilities:** TQDM, OS, Shutil  

### Dataset

The dataset used in this project is the [**Parasite Dataset: Leishmania, Plasmodium & Babesia**](https://www.kaggle.com/datasets/ahmedxc4/parasite-dataset), curated by Ahmed Alrefaei and hosted on Kaggle. It contains **34,298 high-resolution microscopic images** captured under 400X or 1000X magnification.

Given the original class imbalance, a **balanced subset** of three dominant categories was selected:
- **RBCs**
- **Toxoplasma**
- **Trichomonad**

Each class contains **5,000+ samples**, ensuring reliable training and generalization performance.


# Development Workflow

### 1. **Data Preparation**

- Loaded the dataset from Kaggle into the notebook environment.
- Merged all images from the original `train` and `test` folders into a single unified dataset.
- Displayed all available parasite class/categories within the dataset.
- Explored class distribution to visualize potential imbalance across categories.

### 2. **Data Preprocessing**

- Selected three target classes: Trichomonad, TBCs, and Toxoplasma, then stored them in a new variable to filter the dataset for use in the final DataFrame `final_df`.
- Split the data into features `X` and labels `y` with an 80% train / 20% test ratio.
- Reorganized and copied image files from their original locations into a new, clean directory structure, categorized by class and dataset split.
- Applied data augmentation and normalization using `ImageDataGenerator`, with some parameters, following: 
    - `rescale=1./255` for normalize pixel values to the [0, 1] range
    - `rotation_range=15` for randomly rotate images
    - `zoom_range=0.1` for randomly zoom in/out up to 10%
    - `fill_mode='nearest'` for filling empty pixels after transformation using nearest pixel values
    - `validation_split=0.2` for reverse 20% of training data for validation.


### 3. **Model Architecture**

- Built a Sequential Convolutional Neural Network (CNN) consisting of:
    - Three convolutional layers with ReLU activation  for feature extraction, Batch Normalization Layer for stabilizing training, and Max Pooling Layer for spatial downsampling.
    - Flatten, Dense, and Dropout Layers with ReLU activation for regularization.
    - A final Dense Layer with **softmax** activation for multi-class classification.
- Model compiled using:
    - Optimizer: Adam with a learning rate of 0.000001
    - Loss Function: Categorical Crossentropy
    - Evaluation Metric: Accuracy
- Model trained using `train_generator` for 10 epochs with a batch size of 64, incorporating class weights to handle imbalance, validating with `validation_generator`, and applying `EarlyStopping` to optimize training efficiency. Training is stopped if no improvement in validation loss is observed for 5 consecutive epochs, and the model automatically recovers the weights with the best performance.


### 4. **Model Evaluation**

>![Accuracy and Loss Trends](/images/projects/4/image.png)
>- Accuracy increased significantly from epoch 0 to 1, but showed marginal improvement afterward. By the final epoch, training accuracy reached ~99%, while validation accuracy stabilized around ~95%.
>- Training loss consistently decreased from epoch 0 to epoch 7, indicating good learning progress. However, the validation loss started to increase after epoch 2 and continued to increase slightly, showing early signs of overfitting. As a result, training was stopped using early stopping to maintain generalization performance.

![Confusion Matrix](/images/projects/4/image-1.png)

**Classification Report:**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| RBCs        | 0.9776    | 0.9994 | 0.9884   | 1748    |
| Toxoplasma  | 1.0000    | 0.9708 | 0.9852   | 1371    |
| Trichomonad | 0.9995    | 1.0000 | 0.9998   | 2045    |
| **Accuracy**     |            |            | **0.9921** | **5164** |
| **Macro Avg**    | **0.9924** | **0.9901** | **0.9911** | **5164** |
| **Weighted Avg** | **0.9922** | **0.9921** | **0.9920** | **5164** |

- The model achieved ~99% accuracy, indicating high and stable performance in classifying the three categories: RBCs, Toxoplasma, and Trichomonad.
- Toxoplasma achieved the highest precision, meaning all predictions labeled as Toxoplasma were correct. However, it had the lowest recall, with 40 Toxoplasma samples misclassified as RBCs.
- Trichomonad performed nearly perfectly across all metrics, with precision and recall both very high and closely matched.
- RBCs had an excellent recall, indicating nearly all RBC samples were correctly identified. However, precision was slightly lower due to 1 RBC sample misclassifies as Trichomonad and 40 Toxoplasma samples incorrectly predicted as RBCs.


### 5. **Model Deployment**
Converted the trained model into multiple formats for practical use:
- **TensorFlow.js (TFJS)**: for browser-based inference
- **SavedModel**: for server/cloud deployments
- **TensorFlow Lite (TFLite)**: optimized for mobile and embedded devices

### 6. **Inference**

![Inference](/images/projects/4/image-2.png)

> The model successfully predicted the new image (pics.png) as Toxoplasma with high confidence.


### Insights

Training with class weights and early stopping allows the model to handle class imbalance effectively while avoiding overfitting, ensuring optimal generalization by stopping training at diminishing returns points and returning the weights of the best performing model based on validation loss.



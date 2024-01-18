# Traffic Sign Recognition Project

## Overview
This repository contains the code and resources for a research project focused on improving road sign recognition for autonomous vehicles. The project's goal is to enhance recognition accuracy, especially in challenging conditions like poor lighting and adverse weather.

### Key Features
- Utilizes the [German Traffic Sign Benchmarks (GTSRB) dataset](https://benchmark.ini.rub.de/index.html).
- Implements various image enhancement techniques, including [ESRGAN (RRDB)](https://github.com/xinntao/ESRGAN) and simple enhancement methods.
- Develops and evaluates Convolutional Neural Network (CNN) and ResNet models for traffic sign classification.
- Explores model ensembling techniques to create a fusion model.

## Approaches

### Image Enhancement Techniques

To tackle unclear and low-quality images, this project employs advanced image enhancement techniques:

- **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network):** This approach enhances image clarity through upscaling and quality improvement using ESRGAN.
- **Simple Enhancement Techniques:** Leveraging bilateral denoising, histogram equalization, and Gaussian smoothing to enhance image quality while preserving critical details.

### Machine Learning Models

I explore the following machine learning models to achieve robust recognition:

- **Convolutional Neural Networks (CNNs):** A baseline CNN model is established for initial classification.
- **ResNet (Residual Neural Networks):** More complex ResNet models are introduced to extract intricate image features. We investigate both transfer learning and fine-tuning of pretrained ResNet models.

### Data Augmentation

Addressing dataset imbalances and potential sampling biases through data augmentation techniques to ensure robust model training.
![Augmented_data](https://github.com/Jieoi/traffic_sign_recognition/blob/main/img/augmented_dist.jpg)<br>
The distribution of labels before and after augmentation is shown

### Model Evaluation

Meticulous evaluation of various model structures, enhancements, and training iterations, with a primary focus on accuracy as the key performance metric. Comparative analysis guides model selection.

### Deployment on Google Colab

Optimized hardware resource utilization with Google Colab's T5 GPUs for efficient model training.

### Fusion Models

Ensemble techniques are employed, combining the strengths of different models to enhance recognition accuracy.


## Repository Structure
- `0_baseline_performance.ipynb`: Baseline model development and proof of concept on a local machine.
  - Hardware Requirements: Local machine with standard laptop CPU (AMD R7).
- `1_train_data_preparation_simple.ipynb`: Data preparation notebook for training data using simple methods on Google Colab.
  - Hardware Requirements: Google Colab with standard CPU.
- `2_train_data_preparation_RRDB.ipynb`: Data preparation notebook for training data using ESRGAN (RRDB) on Google Colab.
  - Hardware Requirements: Google Colab with standard CPU and extra RAM.
- `3_test_data_preparation_simple.ipynb`: Data preparation notebook for test data using simple methods on Google Colab.
  - Hardware Requirements: Google Colab with standard CPU.
- `4_test_data_preparation_RRDB.ipynb`: Data preparation notebook for test data using ESRGAN (RRDB) on Google Colab.
  - Hardware Requirements: Google Colab with standard CPU and extra RAM.
- `5_model_training_simple_enhanced_data_CNN.ipynb`: Iteratively develops and tests CNN models with simple enhancement on Google Colab.
  - Hardware Requirements: Google Colab with T5 GPU.
- `6_model_training_simple_enhanced_data_resnet.ipynb`: Iteratively develops and tests ResNet models with simple enhancement on Google Colab.
  - Hardware Requirements: Google Colab with T5 GPU and extra RAM.
- `7_model_training_RRDB_enhanced_data_CNN.ipynb`: Iteratively develops and tests CNN models with ESRGAN (RRDB) enhancement on Google Colab.
  - Hardware Requirements: Google Colab with T5 GPU and extra RAM.
- `8_model_training_RRDB_enhanced_data_resnet.ipynb`: Iteratively develops and tests ResNet models with ESRGAN (RRDB) enhancement on Google Colab.
  - Hardware Requirements: Google Colab with T5 GPU and extra RAM.
- `9_Fusion_model.ipynb`: Develops a fusion model using model ensemble techniques on Google Colab.
  - Hardware Requirements: Google Colab with T5 GPU.
- `getData.py`: Python script for extracting data from the source.
  - Hardware Requirements: standard laptop CPU
- `getFile.py`: Python script for getting files.
  - Hardware Requirements: standard laptop CPU.
- `testGetData.py`: Unit tests for data extraction.
  - Hardware Requirements: standard laptop CPU
- `testGetFile.py`: Unit tests for getting files.
  - Hardware Requirements: standard laptop CPU
- `data_augmentation/enhancing_image_RRDB.py`: Python script for implementing the ESRGAN (RRDB) model.
  - Hardware Requirements: Local Machine, called in the enhancement notebooks.
- `data_augmentation/processing_image.py`: Python script for implementing simple image enhancement and augmentation.
  - Hardware Requirements: Local Machine, called in the enhancement notebooks.


## Getting Started
Follow the notebook files mentioned above in sequential order for step-by-step implementation of the project. Make sure to set up the required dependencies and Google Colab environment as described in the notebooks.

## Dataset
The project utilizes the German Traffic Sign Benchmarks (GTSRB) dataset, which comprises 43 distinct traffic sign classes.

## Enhancements and Contributions
Contributions and enhancements to the project are welcome. If you have ideas for improvements or additional features, feel free to create pull requests or open issues.

## Result

### Image Enhancement Examples
Here, I provide visual examples to demonstrate the effect of image enhancement techniques:

#### ESRGAN Enhancement
![ESRGAN Enhancement](https://github.com/Jieoi/traffic_sign_recognition/blob/main/img/esrgan_example.jpg)<br>
 An example of an image enhanced using ESRGAN. Image quality increased after ESRGAN is applied

#### Simple Enhancement
![Simple Enhancement](https://github.com/Jieoi/traffic_sign_recognition/blob/main/img/simple_enhancement_example.jpg)<br>
An example of an image enhanced using simple enhancement methods. Using bicubic interpolation, bilateral denoising, adaptive histogram equalization, and Gaussian smoothing

### Model performance result
![Model_Accuray](https://github.com/Jieoi/traffic_sign_recognition/blob/main/img/model_accuracy.jpg)<br>
A plot for accuracies of all models. ESRGAN produced better result than Simple enhancement

### Model performance result
![Best Models](https://github.com/Jieoi/traffic_sign_recognition/blob/main/img/model_best.jpg)<br>
A plot for comaprison between best models. ResNet model with ESRGAN enhanced data performed the best (98.84% accuracy)

## Acknowledgments
- The GTSRB dataset is used in this project. It is obtained [here](https://benchmark.ini.rub.de/index.html).
- The ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) enhancement technique is used to improve image quality. The ESRGAN code and repository can be found [here](https://github.com/xinntao/ESRGAN).


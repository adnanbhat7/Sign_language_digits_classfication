# Hand Sign Classification with MobileNet and SVM

This project is a hand sign classifier that leverages MobileNet for feature extraction and an SVM (Support Vector Machine) for classification. The combined approach allows for efficient feature extraction with high accuracy, achieving a remarkable 98% accuracy on the test set.

## Motivation

The choice to use a combination of MobileNet and SVM was motivated by the need for an efficient and accurate model:
- **CNN for Feature Extraction**: MobileNet, a lightweight CNN, is known for its efficiency in extracting robust features from images while keeping computational costs low. It makes the model suitable for real-time applications where both speed and accuracy are crucial.
- **SVM for Classification**: SVMs are effective for classification tasks with a limited number of classes. By training on features extracted from MobileNet, the SVM classifier can focus solely on distinguishing classes, further optimizing accuracy.

This combined approach balances the strengths of deep learning (feature extraction) and traditional machine learning (classification), making it ideal for hand sign classification.

## Dataset

The [Hand Signs Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) was used for this project. The dataset consists of various hand gestures representing different letters of the American Sign Language (ASL) alphabet.

## Project Structure

- **Feature Extraction**: The images are processed through MobileNet layers to extract meaningful features.
- **Classification**: The extracted features are used to train an SVM classifier to recognize hand gestures.
- **Model Accuracy**: The model achieved 98% accuracy on the test set.

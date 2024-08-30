
# **Heart Disease Prediction Using Neural Networks**

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation](#installation)
4. [Project Workflow](#project-workflow)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architecture](#model-architecture)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)
12. [References](#references)

## **Project Overview**
This project aims to predict the likelihood of coronary artery disease using neural networks based on clinical data, such as blood pressure, cholesterol levels, heart rate, and other relevant attributes. The goal is to classify patients into varying degrees of heart disease risk. The project utilizes a dataset from the UCI Machine Learning Repository, demonstrating the application of machine learning and AI in healthcare.

## **Dataset Description**
- **Source**: UCI Machine Learning Repository
- **Dataset**: 303 patient records
- **Features**:
  - Age, Gender
  - Resting Blood Pressure
  - Serum Cholesterol Level
  - Maximum Heart Rate Achieved
  - Resting Electrocardiographic Results
  - Exercise-Induced Angina
  - ST Depression Induced by Exercise Relative to Rest
  - Number of Major Vessels Colored by Fluoroscopy
  - Thalassemia Type
- **Target**: Presence or absence of heart disease

## **Installation**
To run this project, you'll need Python installed on your machine, along with the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

## **Project Workflow**
1. **Data Loading**: Import the dataset and examine the data structure.
2. **Data Preprocessing**: Clean the data, handle missing values, normalize the features, and split into training and testing sets.
3. **Model Building**: Define the neural network architecture using Keras.
4. **Training**: Train the model using the training data with appropriate loss function and optimizer.
5. **Evaluation**: Assess the model using test data and various performance metrics.
6. **Deployment**: Save the trained model for deployment in real-world applications.

## **Data Preprocessing**
- **Stratified Train/Test Split**: To maintain class distribution in both training and test sets.
- **Normalization**: Feature-wise normalization is applied, centering each feature around zero and scaling to unit variance.
- **Handling Missing Data**: Impute missing values if necessary.
- **Feature Engineering**: Create additional features if needed.

## **Model Architecture**
- **Input Layer**: Accepts normalized feature inputs.
- **Hidden Layers**:
  - Multiple dense layers with ReLU activation.
  - Dropout layers to prevent overfitting.
  - L1 and L2 regularization techniques to simplify the model.
- **Output Layer**: A dense layer with a sigmoid activation function for binary classification.
  
**Loss Function**: Binary Crossentropy  
**Optimizer**: Adam (with learning rate tuning)  
**Regularization**: L1 and L2 to control weight magnitude  

## **Evaluation Metrics**
- **Accuracy**: Measures the overall correctness of the model predictions.
- **Precision**: Assesses the accuracy of positive predictions.
- **Recall (Sensitivity)**: Measures the model’s ability to identify true positives.
- **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.
- **Confusion Matrix**: Provides detailed insight into true and false predictions.
- **ROC-AUC**: Evaluates the model’s performance across different classification thresholds.

## **Results**
- The model showed promising results, achieving high accuracy and low overfitting due to dropout and regularization.
- Evaluation metrics indicate strong performance in correctly identifying heart disease presence and absence.

## **Usage**
1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Navigate to the project directory:
    ```bash
    cd heart-disease-prediction
    ```
3. Run the Jupyter Notebook or Python script:
    ```bash
    jupyter notebook Heart_Disease_Prediction.ipynb
    ```
4. To use the trained model for predictions:
    - Load the model using Keras `load_model`.
    - Pass the patient data as input and get predictions.

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

## **References**
- François Chollet, "Deep Learning with Python"
- UCI Machine Learning Repository
- Keras Documentation

---

This README provides an overview of your project, outlines the installation steps, and guides users through the process of using the model for heart disease prediction. Let me know if you need any more details or adjustments!

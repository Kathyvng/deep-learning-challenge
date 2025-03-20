# **Alphabet Soup Neural Network Model**  

## **Project Overview**  
This project aims to develop and optimize a deep learning model to predict the success of funding applications submitted to Alphabet Soup, a non-profit organization. The goal was to create a neural network model using TensorFlow/Keras that could classify whether an application would be successful based on various input features. The target performance goal was an accuracy of **75%**.  

---

## **Technologies Used**  
- Python 
- TensorFlow  
- scikit-learn  
- Pandas  
- NumPy  
- Google Colab

---

## **Files Included**  
| File | Description |  
|-------|-------------|  
| `model_training.ipynb` | Jupyter Notebook containing the model creation and training steps |  
| `AlphabetSoupCharity_Optimization_1.h5` | 1st Trained model saved in HDF5 format |  
| `AlphabetSoupCharity_Optimization_2.h5` | 2nd Trained model saved in HDF5 format |  
| `AlphabetSoupCharity_Optimization_3.h5` | 3rd Trained model saved in HDF5 format |  
| `Report.pdf` | Report detailing the analysis and model performance |  
| `readme.md` | This file |  

---

## **Data Preprocessing**  
### ✔️ **Target Variable:**  
- `IS_SUCCESSFUL` – Indicates whether the funding application was successful (1) or not (0).  

### ✔️ **Feature Variables:**  
- All other columns in the dataset except the non-beneficial identifiers.  

### ✔️ **Removed Variables:**  
- `EIN` and `NAME` – These are identifiers and not predictive features.  

### ✔️ **Data Cleaning Steps:**  
- Consolidated low-frequency categories in `APPLICATION_TYPE` and `CLASSIFICATION` into `"Other"`.  
- Converted categorical data to numeric using one-hot encoding.  
- Scaled the data using `StandardScaler` for consistent input distribution.  

---

## **Model Configuration**  
### **1. First Attempt:**  AlphabetSoupCharity_Optimization_1.h5
- 2 hidden layers (64 → 32 neurons)  
- Activation: ReLU  
- Epochs: 25  
- Accuracy: **72.1%**  

### **2. Second Attempt:**  AlphabetSoupCharity_Optimization_2.h5
- 3 hidden layers (64 → 32 → 16 neurons)  
- Activation: ReLU  
- Epochs: 50  
- Accuracy: **72.6%**  

### **3. Third Attempt:**  AlphabetSoupCharity_Optimization_3.h5
- 3 hidden layers (64 → 32 → 24 neurons)  
- Batch Normalization and Dropout added  
- Activation: ReLU  
- Epochs: 75  
- Accuracy: **72.9%**  

---

## **Model Evaluation**  
To evaluate the model, load it and test it against the test set:  
```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model("AlphabetSoupCharity_Optimization_3.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
```

---

## **Results Summary**  
Best accuracy achieved: **72.9%**  
Target accuracy (75%) not reached, but consistent improvements were seen with each attempt.  
Further tuning and model adjustments (e.g., adding more layers, adjusting learning rate) could improve performance.  

---

## **Recommendations for Improvement**  
**Alternative Models:**  
- Random Forest or Gradient Boosting – Tree-based models may handle structured data more effectively.  

**Hyperparameter Tuning:**  
- Use grid search or random search to adjust learning rate, batch size, and hidden layer size.  

**Feature Engineering:**  
- Create new features or select the most relevant ones to improve signal-to-noise ratio.  




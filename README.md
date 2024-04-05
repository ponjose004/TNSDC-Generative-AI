# TNSDC-Generative-AI

# Breast Cancer Classification

**Overview**

This project is designed to classify breast cancer tumors into two categories: benign (non-cancerous) and malignant (cancerous). It accomplishes this task by employing a neural network model trained on a dataset containing various features associated with breast tumors.

**Table of Contents**

- [Dependencies](#Dependencies)
- [Data Collection & Processing](#data-collection--processing)
- [Separating Features and Target](#separating-features-and-target)
- [Splitting the Data](#splitting-the-data)
- [Standardize the Data](#standardize-the-data)
- [Building the Neural Network](#building-the-neural-network)
- [Visualizing Accuracy and Loss](#visualizing-accuracy-and-loss)
- [Accuracy of the Model on Test Data](#accuracy-of-the-model-on-test-data)
- [Making Predictions](#making-predictions)
- [Building a Predictive System](#building-a-predictive-system)


**Dependencies**

To create and execute this project, we rely on several Python libraries and frameworks:

- **NumPy:** A library for numerical computations.
- **Pandas:** Used for data manipulation and analysis.
- **Matplotlib:** Utilized for data visualization and plotting.
- **scikit-learn:** Provides the breast cancer dataset for training and testing.
- **TensorFlow and Keras:** These libraries are used to build and train the neural network model.

**Data Collection & Processing**

- In this section, we load the breast cancer dataset from the scikit-learn library.
- We transform the dataset into a Pandas DataFrame for easier data manipulation.
- A 'target' column is added to the DataFrame to represent the labels of benign (1) and malignant (0).
- Basic data exploration tasks are performed to check for missing values and obtain statistical summaries.

**Separating Features and Target**

- Here, we separate the features (X) from the target (Y). The features are the characteristics of the tumors, and the target represents the tumor's status as benign or malignant.

**Splitting the Data**

- The dataset is split into training and testing sets using the `train_test_split` function. This is a common practice in machine learning to assess model performance.

**Standardize the Data**

- To ensure that all features have the same scale, we standardize the data using the `StandardScaler` from scikit-learn. Standardization means that each feature has a mean of 0 and a standard deviation of 1.

**Building the Neural Network**

- This section focuses on creating a neural network model to perform the classification.
- The model consists of three layers:
  - Input layer with 30 units, corresponding to the 30 features in our dataset.
  - Hidden layer with 20 units and ReLU activation, which introduces non-linearity into the model.
  - Output layer with 2 units and sigmoid activation, allowing the model to predict the probability of each class (benign or malignant).

**Visualizing Accuracy and Loss**

- We use Matplotlib to visualize the model's performance during training. Specifically, we plot the training and validation accuracy and loss across epochs.

**Accuracy of the Model on Test Data**

- After training the model, we evaluate its accuracy on the test data to assess how well it generalizes to unseen samples.

**Making Predictions**

- The model is employed to make predictions on the standardized test data. These predictions provide probabilities for each class (benign and malignant).

**Building a Predictive System**

- To demonstrate how the model can be used for real-world predictions, we provide a sample input data point.
- The input data point is standardized using the same scaler used for the training data.
- The model predicts the class label for the input data, which is then converted into a human-readable class label (Malignant or Benign).

---

You can use this detailed README.md to provide clear and informative documentation for your GitHub repository. Ensure you include actual code snippets, results, and any additional information that helps users understand the project and how to use it.

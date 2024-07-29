# deep-learning-challenge

This project uses machine learning to predict the success of charity donations based on historical data. The dataset includes various features such as application type, affiliation, classification, use case, organization, status, income amount, special considerations, and ask amount. The goal is to build a deep neural network model to classify whether a donation request will be successful.

## `AlphabetSoupCharity` script:

### Preparation

Import Dependencies

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
```

### Preprocessing

1. Path the Link to the Dataset to `pd.read_cs()`. [Link to the dataset](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv)
2. Drop Non-Beneficial Columns with `.drop()`. Columns `EIN` and `NAME` are identification columns and does not contribute to predictability.
3. Replace Application Types and Classifications with Low Counts to simplify the model and reduce noise.
4. Convert Categorical Data to Numeric with `pd.get_dummies()` for predictability.
5. Split Data into Features and Target with `train_test_split()`. To understand the fetures and target variables please refer to `Report.md` file in thos folder.
6. Scale the Data with `StandartScaler`.

### Model and Evaluation

1. Define the Mode:
- **Input Layer**: The number of neurons in the input layer corresponds to the number of input features in the dataset. In this case, it is defined as `number_input_features`, which is the length of `X_train[0]`.
- **First Hidden Layer**: 12 neurons
- **Second Hidden Layer**: 7 neurons
- **Output Layer**: 1 neuron

- **Activation Functions**:
- **First Hidden Layer**: ReLU
- **Second Hidden Layer**: ReLU
- **Output Layer**: Sigmoid

2. Compile the Model with
```python
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
```

3. Train the Model with `.fit()`
4. Evaluate the model with `.evaluate()`
5. Save the model to h5 file with `.save()`

### Summary of `AlphabetSoupCharity_Optimization` script:

The model achieved an accuracy of approximately 72.65%, indicating a decent performance in predicting the success of charity donation requests. Howeever, the target accuracy of 75% was not achieved. Further improvements could be made by experimenting with different architectures, hyperparameters, or additional feature manipulations.

## `AlphabetSoupCharity` script:

### Preprocessing

This script uses the same preprocessing steps mentioned above. The goal of this script was to improve the accuracy score of the original model by experimenting with different architectures, hyperparameters, and Keras Tunner.

Additional Dependencies
```python
import keras_tuner as kt
```

### Attempts summary:

1. Attempt 1: Adding an Extra Hidden Layer: 

Model Structure:
- **Input Features**: The number of input features is determined by the training data.
- **Hidden Layers**:
  - First Hidden Layer: 12 nodes, ReLU activation
  - Second Hidden Layer: 7 nodes, ReLU activation
  - Third Hidden Layer: 3 nodes, ReLU activation
- **Output Layer**: 1 node, Sigmoid activation


Results:
- **Loss**: 0.5534
- **Accuracy**: 0.7270

2. Attempt 2: Changing Activation Functions

Model Structure
- **Input Features**: The number of input features is determined by the training data.
- **Hidden Layers**:
  - First Hidden Layer: 12 nodes, Tanh activation
  - Second Hidden Layer: 7 nodes, Tanh activation
  - Third Hidden Layer: 3 nodes, Tanh activation
- **Output Layer**: 1 node, Sigmoid activation

Results:
- **Loss**: 0.5543
- **Accuracy**: 0.7257

3. Attempt 3: Using Keras Tuner

Model Structure and Hyperparameter Tuning
- **Input Features**: The number of input features is determined by the training data.
- **Hidden Layers**:
  - Various configurations with different numbers of layers and nodes determined by the tuner.
  - Activation functions: ReLU, Tanh, Sigmoid
- **Output Layer**: 1 node, Sigmoid activation

Best Hyperparameters
- **Activation**: Tanh
- **First Layer Units**: 5
- **Number of Layers**: 2
- **Units in Layers**: [1, 7, 1, 9, 7, 5]

Results
- **Loss**: 0.5747
- **Accuracy**: 0.7304

### Model Saving
The model was saved as `AlphabetSoupCharity_Optimization.h5`.

## Summary 

The deep learning model aimed to predict the success of funding applications achieved a maximum accuracy of 0.73, slightly below the target of 0.75. Despite multiple optimization attempts, including adding layers, changing activation functions, and using Keras Tuner for automated tuning, the desired performance was not attained.

A potential recommendation to solve this classification problem more effectively is to  Random Forest technique. These model can provide higher accuracy and better generalization, capturing more complex patterns in the data. 

Another possible improvement for the neural network model could be decreasing the number of input features in the dataset. By reducing dimensions, the model might focus on the most relevant features, potentially enhancing accuracy and reducing overfitting.

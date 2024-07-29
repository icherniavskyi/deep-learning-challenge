# Report: Predicting Successful Funding for Alphabet Soup Nonprofit Foundation

## Overview of the Analysis

The goal of this analysis is to predicts the success of applicants receiving funding from Alphabet Soup. By implementing machine learning techniques, particularly neural networks, we aim to identify the key factors that contribute to the successful utilization of funds, and optimize the allocation process for future applicants.

## Results

### Data Preprocessing

#### Target and Feature Variables
- **Target Variable**: `IS_SUCCESSFUL`, describes successful status.
- **Feature Variables**: All columns except `EIN` and `NAME`.
- **Removed Variables**: `EIN` and `NAME` were removed from the dataset as they do not contribute to the prediction, as they are identification columns.

### Compiling, Training, and Evaluating the Model

For the original model I have selected such configuration: 

1. **Number of Neurons and Layers**:
   - **Input Layer**: The number of neurons in the input layer corresponds to the number of input features in the dataset. In this case, it is defined as `number_input_features`, which is the length of `X_train[0]`.
   - **First Hidden Layer**: 12 neurons
   - **Second Hidden Layer**: 7 neurons
   - **Output Layer**: 1 neuron

2. **Activation Functions**:
   - **First Hidden Layer**: ReLU
   - **Second Hidden Layer**: ReLU
   - **Output Layer**: Sigmoid


#### Rationale

The first hidden layer uses 12 neurons to provide a broad capacity for capturing complex patterns in the data. The second hidden layer, with 7 neurons, allows the model to refine its feature extraction. The output layer has 1 neuron, which is suitable for binary classification tasks, with the sigmoid activation function ensuring the output is between 0 and 1, as our task to predict if the funding successful or not. ReLU is used in the hidden layers boosts training efficiency. The output layer uses a sigmoid function to convert the output to a probability, which is suitable for binary classification.

#### Target Model Performance

Unfortunately, I was unable to achieve the target model performance. The original model had an accuracy of 0.72, falling short of the target accuracy of 0.75.

#### Optimization:

To optimize the model I made three attempts:

1.  Attempt 1
- **Adjustment**: Added a third hidden layer with three neurons.
- **Result**: The accuracy improved slightly (0.726) but did not surpass 75%. 

1.  Attempt 2
- **Adjustment**: Changed the activation functions to `tanh`.
- **Result**: The accuracy remained below 75% (0.725).

1.  Attempt 3
- **Adjustment**: Keras Tunner Automation.
- **Result**: Accuracy improved to 0.73 however still below 75%.

## Summary 

The deep learning model aimed to predict the success of funding applications achieved a maximum accuracy of 0.73, slightly below the target of 0.75. Despite multiple optimization attempts, including adding layers, changing activation functions, and using Keras Tuner for automated tuning, the desired performance was not attained.

A potential recommendation to solve this classification problem more effectively is to  Random Forest technique. These model can provide higher accuracy and better generalization, capturing more complex patterns in the data. 

Another possible improvement for the neural network model could be decreasing the number of input features in the dataset. By reducing dimensions, the model might focus on the most relevant features, potentially enhancing accuracy and reducing overfitting.




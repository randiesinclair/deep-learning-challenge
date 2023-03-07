# Deep Learning Challenge

In this project, we'll use adeep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. . I will be using machine learning and neural networks to analyze a CSV containing over 34,000 organizations that have received funding from Alphabet Soup, using metadata columns to predict the success of future applicants.

# Technical Skills

- Machine learning
- Deep learning
- Neural networks
- Tensorflow
- Pandas
- Python
- Statistical analysis
- Big data technologies

# Project Parameters
### Preprocess the Data
- Create a dataframe containing the charity_data.csv data , and identify the target and feature variables in the dataset
- Drop the EIN and NAME columns
- Determine the number of unique values in each column
- For columns with more than 10 unique values, determine the number of data points for each unique value
- Create a new value called Other that contains rare categorical variables
- Create a feature array, X, and a target array, y by using the preprocessed data
- Split the preprocessed data into training and testing datasets
- Scale the data by using a StandardScaler that has been fitted to the training data 
### Compile, Train and Evaluate the Model
- Create a neural network model with a defined number of input features and nodes for each layer
- Create hidden layers and an output layer with appropriate activation functions
- Check the structure of the model
- Compile and train the model
- Evaluate the model using the test data to determine the loss and accuracy
- Export your results to an HDF5 file named AlphabetSoupCharity.h5
### Optimize the Model
- Repeat the preprocessing steps in a new Jupyter notebook
- Create a new neural network model, implementing at least 3 model optimization methods
- Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5
### Write a Report on the Neural Network Model
- Write an analysis that includes a title and multiple sections, labeled with headers and subheaders
- Format images in the report so that they display correction
- Explain the purpose of the analysis
- Answer all 6 questions in the results section
- Summarize the overall results of your model
- Describe how you could use a different model to solve the same problem, and explain why you would use that model

-------

## FINAL REPORT
-------
# Alphabet Soup Funding Analysis
## Purpose
(images)

(explain analysis)

(answer 6 questions in the results section)
### Data Preprocessing
- What variable(s) are the target(s) for your model?
    - The target for each model was the IS_SUCCESSFUL column. The ultimate goal is to predict which applicants will be successful, so we will utilize all other features to ultimately make a predict
- What variable(s) are the features for your model?
    - The available features for my model are: EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT	
- What variable(s) should be removed from the input data because they are neither targets nor features?
    - In the first model I ran I removed the EIN column and the NAME columns as instructed. In the first attempt at optimization, I added the NAME column back in case there was any bias to any specific organizations, as there where many that where listed multiple times.
#### Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?
- In the first model, 
- Were you able to achieve the target model performance?
 	- I was unsuccessful in achieving a target model performance of 75% or higher in all attempts, including my initial model and 3 different optimizations.
- What steps did you take in your attempts to increase model performance?
	- In my first optimization I added a third hidden layer, and utilized LeakyRelu algorithm instead of Relu which I used in the first attempt.

(summarize overall results of the model)

(describe how I could use a different model to solve the same problem, and explain why I would use that model)



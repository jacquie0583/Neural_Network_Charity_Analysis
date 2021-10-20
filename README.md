# Neural Network Charity Analysis:  Deep Learning ML
# Project Overview
For this project, we are using Neural Networks Machine Learning algorithms, also known as artificial neural networks, or ANN. For coding, we used Python TensorFlow library in order to create a binary classifier that is capable of predicting whether applicants will be successful if funded by the nonprofit foundation called Alphabet Soup. From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization.This ML model will help ensure that the foundation’s money is being used effectively. With neural networks ML algorithms we are creating a robust deep learning neural network capable of interpreting large complex datasets. Important steps in neural networks ML algorithms are a) data cleaning and data preprocessing as well as decision b) what data is beneficial for the model accuracy.

# Resources

•	Dataset charity_data.csv

•	Software: Google Colab

•	Languages: Python

•	Libraries: Scikit-learn, TensorFlow, Pandas

•	Environment: Python 3.7

Figure 1


<p align="center">
  <img width="600" height="300" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%201.png">
</p>


# The Process/Results
## Step 1:  Data Reprocessing
First,  we built the Pandas DataFrame we will be working with. We used Pandas and Scikit-Learn StandardScaler() function to preprocess the dataset. Next, we will compile, train, and evaluate the neural network models effectiveness.

Figure 2 - The Dataframe of Information we'll be working with


<p align="center">
  <img width="1000" height="600" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%202.jpg">
</p>


The variables that are considered the targets in this model are  our “IS_SUCCESSFUL” column. Target variables are also known as dependent variable and we are using this variable to train our ML model.
The variables  considered to be the features for this model are the input values, also known as independent variables. Those variables include all columns, except target variable and the ones we dropped “EIN" and "NAME” in the first trial (Figure 2) and only “EIN” (we kept the "NAME") in optimization trial.
The variables that should be removed and are neither targets nor features are variables that are meaningless for the model. The variables that don’t add to the accuracy to the model. One of the examples would be variables with all unique values. Another thing to keep in mind is to take care of the Noisy data and outliers. We can approach to this by dropping outliers or bucketing.

## Step2:  Compiling, Training, and Evaluating the Model

•	We utilized 2 layers, because 3 layers didn’t contribute much to the improvement of the ML module. This is because the additional layer was redundant—the complexity of the dataset was encapsulated within the two hidden layers. Adding layers does not always guarantee better model performance, and depending on the complexity of the input data, adding more hidden layers will only increase the chance of overfitting the training data.

•	We utilized relu activation function, since it has best accuracy for this model.

•	We used adam optimizer, which uses a gradient descent approach to ensure that the algorithm will not get stuck on weaker classifying variables and features and to enhance the performance of classification neural network. As for the loss function, I used binary crossentropy, which is specifically designed to evaluate a binary classification model.

•	The Model was trained on 100 epochs. We tried different epoch settings and the models did made significant continuous improvements as you can see in the results below.

Figure 3 - The design of the TensorFlow Model to start our work with


<p align="center">
  <img width="800" height="700" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%203.jpg">
</p>


Figure 4 - Original lower performance accuracy = 45%

<p align="center">
  <img width="800" height="200" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%204.jpg">
</p>


Figure 5 - Changing layers to hidden:  20,10 from 8,5

<p align="center">
  <img width="800" height="200" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%204_25.png">
</p>


Figure 6 - Changing the Activation: Tanh:

<p align="center">
  <img width="800" height="200" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%204.75.png">
</p>


Figure 7 - Adding a fourth layer

<p align="center">
  <img width="800" height="200" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%204.8.png">
</p>

Figure 8 - Adding 2nd outer layer 

<p align="center">
  <img width="800" height="200" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%204.9.png">
</p>


After few configurations of number of hidden nodes we were able to achieve the target model performance. The model accuracy improved from 64% then 76.10% then to 76.20%, and eventually were were pleased to get to 77.20% with low loss at .46%.
Figure 9 - Better with Optimization - accuracy at 64.57%

<p align="center">
  <img width="800" height="200" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%205.jpg">
</p>


Figure 10 - Even Better with modifications to Optimization - accuracy at 76.10%

<p align="center">
  <img width="800" height="200" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%206.jpg">
</p>

Figure 11 - and Better Performance at accuracy = 76.20%

<p align="center">
  <img width="600" height="300" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/Resources/image%207.jpg">
</p>


Figure 12 - and even Better Performance at accuracy = 77.20% - our Best so far.

<p align="center">
  <img width="600" height="300" src="https://github.com/jacquie0583/Neural_Network_Charity_Analysis/blob/main/image%2012.png">
</p>

In order to increase model performance, we took the following steps:

•	Checked input data and brought back NAME column, that was initially skipped. We set a condition on that value for any that are less than 50 in “Other” group. This approach reduced the number of unique categorical values by binning the values. Noisy variables reduced by binning.

•	Binned the ASK_AMT values.

•	Added neurons to hidden layers and added hidden layers.

•	At first, I added the third layer with 40 neurons; however, I’ve changed back to 2 layers, because the results did not improve much if any. Increase neurons for each layer (200 for 1st, 90 for 2nd).

•	Increased Epochs to 500.

•	Models weights are saved very 5 epochs.

•	and FINALLY, added The Random Forest Algorithm for the best performance of all in Figure 12.

# Summary
Summary of the results:

The model loss and accuracy score tell us how well the model does with the dataset and parameters that we build the model. In the end, the most optimal model we ran was the last one - Figure 12 above. We were pleased that we were able to get the loss score down to 0.46, meaning the probability model to fail is 46% and accuracy score up to 0.772, meaning that the probability model to be accurate is 77.20%.

# Recommendation for further analysis:
After a lot of experimenting and fine-tuning to the model we were able to continuously improve the accuracy score and eventually get up to 76% accuracy and 74% loss as you can see in Figure 11. This is not necessarily the best model for this dataset. The loss score for that model was still so high at about 74%.
Random Forests Algorithm might be the best fit for this dataset because the performance is better.
In Figure 12 above we can see the best results using this model. We were pleased that we were able to get the loss score down to .46, meaning the probability model to fail is 46% and accuracy score up to 0.772, meaning that the probability model to be accurate is 77.20%.
Adding new input values seemed a good choice when improving the model accuracy. In this case I would consider adding more input values (if there are some available in the original dataset, for example). Another thing we could do, is to consider gathering more data. Although gathering more data is not always the easy decision but it is sometimes necessary.







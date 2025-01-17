# Marketing-Analytics-Project
The question that this project is going to explore is predicting the property crime rate within the Unites States using the percentage of the population with a bachelor’s degree, per capita personal income, unemployment rate, police presence per 100,000 population and the existence of the death penalty. This project specifically focuses on the data for property crimes and all of the variables mentioned above for each state from 2008-2016. 


The reason behind the choice to explore this subject is the severe impact that property crime has on the economy. For instance, according to the United States Department of Justice, “property crimes in 2012 alone resulted in losses estimated at $15.5 billon.” 


In order to predict the property crime rate based on all of the variables mentioned above, I used the following models: Linear Regression, Lasso, Random Forest Regressor and Naïve Bayes Classification. 


The standard deviation or bias of the prediction was 588.9889236614156, 588.9958121159633, 478.0988035995921, 480.0988035995921 for the linear Regression, Lasso, Random Forest Regressor and Naïve Bayes Classification respectively. Since Random Forest Regressor has the smallest bias out of all of the other models, it is the model that should be used to predict the property crime rate. The fact that the linear regression model had a large standard deviation suggests that there is non-linear relationship between the variables mentioned above and using a flexible model such as the Random Forest Regressor to predict the property crime rate is reasonable. 

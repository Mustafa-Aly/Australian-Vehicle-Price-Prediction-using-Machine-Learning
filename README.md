This is a Regression Problem to Predict the Price of the Australian Vehicle
The dataset from kaggle "https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices/data"

This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market.

-Note : please read the requirement File to see the version of the libraries to not be any error if you run the code later 

First step : Exploration the Data , Handling and Cleaning ( Cleaning_and_Handeling_Dataset.ipynb ) :
  This some of the Handling and cleaning the Dataset:
  - Merge Brand and Model 
  - Car/Suv be in 8 category or drop it cause we have BodyType
  - transmission replace '-' with null
  - Engine replace "-" with null , electric cars have 0 cyl and 0 L
  - Fuel Type replcae "-" with null , be in 6 category
  - Fuel consumption replace - with null , extract the first number ( 8.1 L / 100KM ) -> (8.1)
  - Kilometres replace ['-' , '-/-'] with null , KNN imputer after handle the nulls
  - Colour extract the first string if it was a string (color) if not replace with null 
  - Location extract the state
  - cylinder replace - with null
  - Doors extract the number of doors ( 5 Doors) -> 5
  - Seats extract the number of Seats ( 8 Seats) -> 8
  - Price the wrong data replace it with Null then use KNN imputer to handle the Nulls
  - replace the Nulls in category columns with Mode
  - Save the new dataset to use it in analysis and modeling

Second Step : Analysis (Data_Visualization_and_Analysis.ipynb):
In this Step I tried to Ask some questions to make it easier to see what can affect on the Price 
So First I do some Univariate Analysis to each column
  - In Category Columns see the value count of each unique value and calculate its percentage of the total
  - In Numeric Columns see the distribution of the data and see if it (right skewness or left skewness)
Second I do Bivariate Analysis to some columns and See the impact on the Price
  - See the corrlation of the Numeric columns andd see what have Postive relation and which have negative
  - See the impact on Price depend of some Columns

Third Step : Modeling (Modeling_and__Tuning.ipynb)
In this step I do my best to get the best possible accuracy using different techniques 
- I use column transform ( Log transform to Kilometres Column ) , (One Hot Encoder to category columns have less than 7 unqiue values) , (Binary Encoding to category columns have more than 7 unqiue values)  
- Because It's a Regression Problem so I use [ R2_score , Mean_Absolute_error (MAE) ] as metrics 
- I see the diffrent between not do any thing to the Target column and when we done (Log Transform) to it [it do better when i use log]
- I tried a lot of regression models and compare the accuracy between models
- The Three highest performance model I use Tuning on it and get accuracy over 91.8%
- I tried diffrents Scaler betweee (Standerd Scaler , MinMax Scaler ,Robust Scaler ) but the best with Tuning model was (Standerd Scaler)
- Using some technique Like (Principal Component Analysis PCA , Feature selection ] but didn't give higher accuracy so I remove it
- Save the model with column transform and Scaler to the Deployment

Fourth Step : Deployment can see in this Link https://australian-vehicle-price-prediction-using-machine-learning-xed.streamlit.app/
I used StreamLit to deploy the model and it was so helpful 

Here some analysis of the Project 
![Dashboard 2](https://github.com/Mustafa-Aly/Australian-Vehicle-Price-Prediction-using-Machine-Learning/assets/129996921/e3975ea6-d983-4fc7-951b-8c3a77a759ac)
![Dashboard 1](https://github.com/Mustafa-Aly/Australian-Vehicle-Price-Prediction-using-Machine-Learning/assets/129996921/4a7598b5-d791-4f08-a2c4-5fbab0d1c6b4)


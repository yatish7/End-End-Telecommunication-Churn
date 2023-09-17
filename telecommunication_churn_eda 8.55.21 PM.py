

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
# %matplotlib inline

""" **Importing the CSV file**"""

tel_data = pd.read_csv('TelecomChurnDataset.csv')

"""**First 5 rows in dataset**"""

tel_data.head()

tel_data.shape # To know no of rows and columns in the dataset

tel_data.columns.values #To know what all columns are there in the dataset

tel_data.dtypes #To know the datatypes of the attributes

tel_data.describe() # Gives statistical measurments of numerical attributes

"""# ***Analysis from the Statistical Measurments***

SeniorCitizen is a binary atribute, but in case of its datatype its integer thats the reason we get stats for it too(which is not necessary)

75% people have tenure less than 55 months

The average charges done by people are $64.7, whereas 75% of people pay more than $35.5

# ***Visualise the No of Non-Churners to Churners***
"""

tel_data['Churn'].value_counts() # 5174 are no of Non-Churners wheras 1869 are no of Churners

100*tel_data['Churn'].value_counts()/len(tel_data['Churn']) # 73% of the people are non_churners wheras 27% are Churners

tel_data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);

"""**Find Null or Missing values**"""

tel_data.info(verbose=True) # 7043 entries or 7043 rows

""" **Find the Percentage of Missing Values**"""

missing = pd.DataFrame((tel_data.isnull().sum())*100/tel_data.shape[0]).reset_index() #isnull method creates a dataframe of size tel_data with True in the cells if the the data is False or null then the sum method will calculate the no of all the null values in the column and tel_data.shape(0) gives totla number of rows
#with reset_index() new column will be choosen after one column is done
plt.figure(figsize=(16,5))
ax = sns.pointplot(x='index',y=0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()

"""# ***DATA CLEANING***"""

tel_data_copy=tel_data.copy() # Copying the whole dataset to check null values for object datatypes

"""**As per teh Data set,Total Charges have null values**"""

tel_data_copy.TotalCharges =pd.to_numeric(tel_data_copy.TotalCharges, errors='coerce') # Object -> Numeric
tel_data_copy.isnull().sum() #isnull for finding how many are null, and sum for the sum of the null values in a column

"""**There are 11 null values in the TotalCharges column which were not identified the first time**"""

tel_data_copy.loc[tel_data_copy ['TotalCharges'].isnull() == True]
 # This will show all the records of the TotalCharges where ther is NaN value, generally the isnull() method will return value of type of boolean thats why we use True

"""# **Dealing with Missing Values**

Since only 11 records out of 7043 are null which is very minimal u can ignore them by dropping them if the percentage is way to less

"""

#Removing missing values
tel_data_copy.dropna(how = 'any', inplace = True) # This will drop the records with null values

#telco_data.fillna(0) # This will fill null values with 0 (Just Knowledge)

print(tel_data_copy['tenure'].max()) # Max record in Tenure column

"""**As you can see that the tenure has max number of 72 months its very difficult to analyse data in case of churners and non_churners for 72 monnths, thats y we make a different column and convert it into categorical varriable with classes so that analysis will become easy**"""

# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

tel_data_copy['tenure_group'] = pd.cut(tel_data_copy.tenure, range(1, 80, 12), right=False, labels=labels)

tel_data_copy['tenure_group'].value_counts() # This will count the records in each class

#drop column customerID and tenure this are not needed
tel_data_copy.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
# Assuming your DataFrame is named 'tel_data'
tel_data_copy.drop(['DeviceProtection', 'StreamingTV', 'StreamingMovies'], axis=1, inplace=True)

tel_data_copy.head()

"""**In order to drop columns always have domain knowledge and then see whether the column is needed or not and drop it**


In the above case, the customer_id is something which is randomly generated for a customer, so for the churn analysis the ID or name is not at all required, tenure is removed because we have tenure group

# ***Uni-Variate Analysis***
"""

for i, predictor in enumerate(tel_data_copy.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=tel_data_copy, x=predictor, hue='Churn')

tel_data_copy['Churn'] = np.where(tel_data_copy.Churn == 'Yes',1,0) # Churn- Categorical->Binary

tel_data_copy.head()

"""**Convert all the categorical variables into dummy variables**"""

tel_data_dummies = pd.get_dummies(tel_data_copy) # u can use one-hot encoding method too, get_dummies will do the same thing, dummy trap is not done but it can be done
tel_data_dummies.head()

"""**Have a very good domain knowledge and check the correlation between or do bi-variate analysis, tri variate analysis for more insights on related varriables**"""

tel_data_dummies.to_csv('tel_churn.csv')
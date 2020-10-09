#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


# ## Importing dataset
# 
# 1. Since data is in form of excel file we have to use pandas read_excel to load the data
# 2. After loading it is important to check the complete information of data as it can indication many of the hidden infomation such as null values in a column or a row
# 3. Check whether any null values are there or not. if it is present then following can be done,
#     1. Imputing data using Imputation method in sklearn
#     2. Filling NaN values with mean, median and mode using fillna() method
# 4. Describe data --> which can give statistical analysis

# In[2]:


df_train = pd.read_excel(r"C:\Users\Sameer\Desktop\datasets\flight_price_dataset/flight_train.xlsx")


# In[3]:


pd.set_option("display.max_columns", None)


# In[4]:


df_train.head()


# In[5]:


df_train.info()


# In[6]:


df_train["Duration"].value_counts()


# In[7]:


df_train.dropna(inplace = True)


# In[8]:


df_train.isnull().sum()


# # EDA

# From description we can see that Date_of_Journey is a object data type,\
# Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction
# 
# For this we require pandas **to_datetime** to convert object data type to datetime dtype.
# 
# <span style="color: red;">**.dt.day method will extract only day of that date**</span>\
# <span style="color: red;">**.dt.month method will extract only month of that date**</span>

# In[9]:


df_train["Journey_Day"] = pd.to_datetime(df_train.Date_of_Journey, format = "%d/%m/%Y").dt.day


# In[10]:


df_train["Jounrey_Month"] = pd.to_datetime(df_train.Date_of_Journey, format = "%d/%m/%Y").dt.month


# In[11]:


df_train.head()


# In[12]:


# Since we have converted date of jouney into integers or sperate columns(No need to convert year section as the data us for only one year i.e 2019), we can drop it now as it is of no use now

df_train.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[13]:


# Updated dataset after converting Date of Journey columns into integers and droping Date of Journey column 
df_train.head() 


# From description we can see that Deep_Time is a object data type,\
# Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction
# 
# For this we require pandas **to_datetime** to convert object data type to datetime dtype.
# 
# <span style="color: red;">**.dt.day method will extract only day of that date**</span>\
# <span style="color: red;">**.dt.month method will extract only month of that date**</span>

# In[14]:


# Exracting Hours

df_train["Dep_Hr"] = pd.to_datetime(df_train.Dep_Time).dt.hour

# Extracting Mintutes

df_train["Dep_Minute"] = pd.to_datetime(df_train.Dep_Time).dt.minute


# In[15]:


# Since we have converted Departe time into hours and minutes we can drop "Dep_Time" column

df_train.drop(["Dep_Time"], axis = 1, inplace = True)


# In[16]:


# Updated dataset after converting departure time into hours and minutes and droping "Dep_Time" column
df_train.head()


# From description we can see that Arrival_Time is a object data type,\
# Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction
# 
# For this we require pandas **to_datetime** to convert object data type to datetime dtype.
# 
# <span style="color: red;">**.dt.day method will extract only day of that date**</span>\
# <span style="color: red;">**.dt.month method will extract only month of that date**</span>

# In[17]:


# Extracting hours from Arrival time

df_train["Arrival_Hr"] = pd.to_datetime(df_train.Arrival_Time).dt.hour

# Extracting minutes from Arrival Time

df_train["Arrival_Minutes"] = pd.to_datetime(df_train.Arrival_Time).dt.minute

# Now we can drop Arrival_time as it is of no use now

df_train.drop(["Arrival_Time"], axis = 1, inplace = True )


# In[18]:


# Updated Dataset after dropping Arrival_Time column

df_train.head()


# In[19]:


# Time taken by the flight to reach destination is also called Duration
# It is the difference between Departure Time and Arrival Time

# Assigning and converting Duraton Column into list

duration = list(df_train["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) !=2:     # Checks if duration contains only hours or minutes
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"  # Adds 0 mintue
        else:
            duration[i] = "0h " + duration[i]          # Adss 0 hour

duration_hours = []
duration_mins= []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))             # Extract hours from the feature
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))  # Extract minutes from the feature


# In[20]:


df_train["Duration_Hrs"] = duration_hours
df_train["Duratin_Min"] = duration_mins


# In[21]:


df_train.head()


# In[22]:


df_train.drop(["Duration"], axis = 1, inplace = True)


# In[23]:


# Updatetd Dataset after droppint "Duration" column

df_train.head()


# ## Handling Categorical Data
# 
# One can find many ways to handle categorical data. Some of them categorical data are,
# 1. <span style="color: blue;">**Nominal data**</span> --> data are not in any order --> <span style="color: green;">**OneHotEncoder**</span> is used in this case
# 2. <span style="color: blue;">**Ordinal data**</span> --> data are in order --> <span style="color: green;">**LabelEncoder**</span> is used in this case

# In[24]:


df_train["Airline"].value_counts()


# In[25]:


# Airline vs Price

sns.catplot(x = "Airline", y = "Price", data = df_train.sort_values("Price", ascending = False), kind = "boxen", height = 6, aspect = 2)
plt.show()


# In[26]:


# From the above graph it is clear that jet airways has the maximun price
# And also apart from the jet airways almost all the other airlines has the same median


# In[27]:


# As Airline is a Nominal Category we will perform one hot encoding

Airline  = df_train[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first = True)
Airline.head()


# In[28]:


df_train["Source"].value_counts()


# In[29]:


# Source vs Price

sns.catplot(x = "Source", y = "Price", data = df_train.sort_values("Price", ascending = False), kind = "boxen", height = 6, aspect = 2)
plt.show()


# In[30]:


# Here there is differnce in medians
# As Source is a nominal data we will perform one hot encoding
Source = df_train[["Source"]]
Source = pd.get_dummies(Source, drop_first = True)
Source.head()


# In[31]:


df_train["Destination"].value_counts()


# In[32]:


# Destinstion vs Price

sns.catplot(x = "Destination", y = "Price", data = df_train.sort_values("Price", ascending = False), kind = "boxen", height = 6, aspect = 2)
plt.show()


# In[33]:


# In the above graph the median for New Delho, Cochin and Banglore is almost same 
#and median for Hyderabad, Kolkata, Delhi is almost same.


# In[34]:


# As Destination is also a Nominal data so we will perform one hot encoding

Destination = df_train[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first = True)
Destination.head()


# In[35]:


df_train.Route


# In[36]:


df_train["Total_Stops"].value_counts()


# In[37]:


df_train["Additional_Info"].value_counts()


# In[38]:


# from the above analysis it is clear that Additinal_info columns as majority of no_info data so it can be dropped
# Also Route and Total_Stops are related to each other so one of them can be droppped. 

df_train.drop(["Additional_Info", "Route"], inplace = True, axis = 1)


# In[39]:


# updated dataset after dropping Additional_Info and Route column
df_train.head()


# In[40]:


# Now we analyse the Total Stops Column

df_train.Total_Stops.value_counts()


# In[41]:


# As Total stops is a ordinal data we will perfom label encoding

df_train.replace({"non-stop" :0, "1 stop" : 1, "2 stops" : 2, "3 stops" : 3, "4 stops" : 4 }, inplace = True)


# In[42]:


df_train.head()


# In[43]:


# Now we concatenate all the converted columns with the updated dataset i,e df_train, Airline, Source, Destination

df_train = pd.concat([df_train, Airline, Source, Destination], axis = 1 )


# In[44]:


# updated dataset after the concatinating operation

df_train.head()


# In[45]:


df_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[46]:


df_train.head()


# In[47]:


df_train.shape


# In[48]:


df_test = pd.read_excel(r"C:\Users\Sameer\Desktop\datasets\flight_price_dataset\flight_test.xlsx")
df_test.head()


# In[49]:


# Preprocssing the test data

print("Test data Info")
print("*"*70)
print(df_test.info)

print()
print()

print("Null values")
print("*"*70)
df_test.dropna(inplace = True)
print(df_test.isnull().sum())


# In[50]:


# EDA

# Date of Jouney

df_test["Journey_Day"] = pd.to_datetime(df_test["Date_of_Journey"], format = "%d/%m/%Y").dt.day
df_test["Journey_Month"] = pd.to_datetime(df_test["Date_of_Journey"], format = "%d/%m/%Y").dt.month
df_test.drop(["Date_of_Journey"], axis = 1,inplace = True)


# In[51]:


# Departue Time

df_test["Dep_Hr"] = pd.to_datetime(df_test["Dep_Time"]).dt.hour
df_test["Dep_Minute"] = pd.to_datetime(df_test["Dep_Time"]).dt.minute
df_test.drop(["Dep_Time"], axis = 1, inplace = True)


# In[52]:


# Arrival Time

df_test["Arrival_Hr"] = pd.to_datetime(df_test["Arrival_Time"]).dt.hour
df_test["Arrival_Mintue"] = pd.to_datetime(df_test["Arrival_Time"]).dt.minute
df_test.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[53]:


# Duration
duration = list(df_test["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[54]:


# Adding Duration column to test set
df_test["Duration_Hrs"] = duration_hours
df_test["Duration_Min"] = duration_mins
df_test.drop(["Duration"], axis = 1, inplace = True)


# In[55]:


# Caregorical Data

print("Airline")
print("*"*70)
print(df_test.Airline.value_counts())
# Airline = df_test[["Airline"]]
Airline = pd.get_dummies(df_test["Airline"], drop_first = True)


# In[56]:


print("Source")
print("*"*70)
print(df_test.Source.value_counts())
# Source = df_test["Source"]
Source = pd.get_dummies(df_test["Source"], drop_first = True)


# In[57]:


print("Destination")
print("*"*70)
print(df_test.Destination.value_counts())
# Destination = df_test["Destination"]
Destination = pd.get_dummies(df_test["Destination"], drop_first = True)


# In[58]:



# from the above analysis it is clear that Additinal_info columns as majority of no_info data so it can be dropped
# Also Route and Total_Stops are related to each other so one of them can be droppped.

df_test.drop(["Additional_Info", "Route"], axis = 1, inplace = True)


# In[59]:


# As Total stops is a ordinal data we will perfom label encoding

df_test.replace({"non-stop" : 0, "1 stop" : 1, "2 stops" : 2, "3 stops" : 3, "4 stops" : 4 }, inplace = True)


# In[60]:


# Now we concatenate all the converted columns with the updated dataset i,e df_train, Airline, Source, Destination

df_test = pd.concat([df_test, Airline, Source, Destination], axis = 1 )


# In[61]:


df_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[62]:


df_test.shape


# In[63]:


df_test.head()


# In[64]:


df_test.columns


# ## Feature Selection
# 
# Finding out the best feature which will contribute and have good relation with target variable.
# Following are some of the feature selection methods,
# 
# 
# 1. <span style="color: purple;">**heatmap**</span>
# 2. <span style="color: purple;">**feature_importance_**</span>
# 3. <span style="color: purple;">**SelectKBest**</span>

# In[65]:


df_train.shape


# In[66]:


df_train.columns


# In[67]:


X = df_train.loc[:,['Total_Stops', 'Journey_Day', 'Jounrey_Month', 'Dep_Hr',
       'Dep_Minute', 'Arrival_Hr', 'Arrival_Minutes', 'Duration_Hrs',
       'Duratin_Min', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]


# In[68]:


X.head()


# In[69]:


y = df_train.iloc[:,1]


# In[70]:


y.head()


# In[71]:


# Now we find the correlation between Independent Variables and Depenent Variable

plt.figure(figsize = (22,22))
sns.heatmap(df_train.corr(), annot = True, cmap = "RdYlGn")
plt.show()


# In[72]:


# Now we will see important feature using ExtraTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)


# In[73]:


# Here we plot graph of feature importance for better visualization

plt.figure(figsize= (12,8))
feat_importances = pd.Series(selection.feature_importances_, index = X.columns)
feat_importances.nlargest(20).plot(kind = "barh")
plt.show()


# ## Fitting model using Random Forest
# 
# 1. Split dataset into train and test set in order to prediction w.r.t X_test
# 2. If needed do scaling of data
#     * Scaling is not done in Random forest
# 3. Import model
# 4. Fit the data
# 5. Predict w.r.t X_test
# 6. In regression check **RSME** Score
# 7. Plot graph

# In[74]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)


# In[75]:


from sklearn.ensemble import RandomForestRegressor

ran_fg = RandomForestRegressor()
ran_fg.fit(X_train,y_train)


# In[76]:


y_pred = ran_fg.predict(X_test)


# In[77]:


ran_fg.score(X_train,y_train)


# In[78]:


ran_fg.score(X_test,y_test)


# In[79]:


sns.distplot(y_test-y_pred)
plt.show()


# In[80]:


plt.scatter(y_test,y_pred, alpha = 0.4)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[81]:


from sklearn import metrics


# In[82]:


print("MAE: ", metrics.mean_absolute_error(y_test,y_pred))
print("MSE: ", metrics.mean_squared_error(y_test,y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[109]:


metrics.r2_score(y_test,y_pred)


# ## Hyperparameter Tuning
# 
# 
# * Choose following method for hyperparameter tuning
#     1. **RandomizedSearchCV** --> Fast
#     2. **GridSearchCV**
# * Assign hyperparameters in form of dictionery
# * Fit the model
# * Check best paramters and best score

# In[83]:


from sklearn.model_selection import RandomizedSearchCV


# In[85]:


# Randomized Search CV

# Number of trees in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of featueres to be considered at every split
max_features = ["auto", "sqrt"]

# Maximum numbe of levels in Tree
max_depth = [int(x) for x in np.linspace(start = 5, stop = 30, num = 6)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[86]:


# Creating the Random Grid

random_grid = {"n_estimators": n_estimators,
              "max_features": max_features,
              "max_depth": max_depth,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf
              }


# In[87]:


# Random search for CV using 5 fold cross validation
# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = ran_fg, param_distributions = random_grid, scoring = "neg_mean_squared_error", n_iter = 15, cv = 5, verbose = 2, random_state= 42, n_jobs = 1)


# In[88]:


rf_random.fit(X_train, y_train)


# In[89]:


rf_random.best_params_


# In[96]:


prediction = rf_random.predict(X_test)


# In[97]:


plt.figure(figsize = (9,9))
sns.distplot(y_test - prediction)
plt.show()


# In[98]:


plt.figure(figsize = (9,9))
plt.scatter(x = y_test, y = prediction)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[100]:


print("MAE: ", metrics.mean_absolute_error(y_test, prediction))
print("MSE: ", metrics.mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# # Saving the Model so that we can reuse is again
# 

# In[101]:


import pickle
# open a file where we want to store the data 
file = open("flight_rfr.pkl", "wb")

# dump all the information to the file
pickle.dump(ran_fg, file)


# In[104]:


model = open("flight_rfr.pkl", "rb")
forest = pickle.load(model)


# In[106]:


y_prediction = forest.predict(X_test)


# In[107]:


metrics.r2_score(y_test, y_prediction)


# In[ ]:





# In[ ]:





# In[ ]:





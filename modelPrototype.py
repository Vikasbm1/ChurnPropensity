#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import math
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, ConfusionMatrixDisplay, precision_score, recall_score


# In[2]:


from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

#
bucket_name = 'churn-model-prediction'
source_blob_name = 'Telecom-Customer-Churn.csv'
destination_file_name = '/home/jupyter/ChurnPropensity/dataset1/Telecom-Customer-Churn.csv'

download_blob(bucket_name, source_blob_name, destination_file_name)


# In[3]:


#df = pd.read_csv('Telecom-Customer-Churn.csv',index_col = 'customerID')
df = pd.read_csv('dataset1/Telecom-Customer-Churn.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns.values


# In[7]:


df_columns = df.columns.tolist()
for column in df_columns:
    unique_values = df[column].unique()
    print(f"{column} unique values: {unique_values}")


# In[8]:


df.dtypes


# In[9]:


df.describe()

#below we can see senior citizen has categorical data
#The average customer stayed in the company is 32 months and 75% of customer has a tenure of 55 month
#Average monthly charges are USD 64.76 and 25% of customers pay more than USD 89.85


# In[10]:


df.info()


# In[11]:


#errors='coerce' parameter -invalid parsing will be set to NaN (Not a Number)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# In[12]:


# to check the nan value which coerce has replaced if any value found that cannot be converted to numeric

df.isnull().sum() 


# In[13]:


df.Churn.value_counts()
#here we can see the data is much imbalanced so we would be needing resampling of the data
#to evaluate the model


# In[14]:


x=list(df.Churn.value_counts())
print('No' , x[0]/(x[0]+x[1])*100 ,'%')
print('Yes' , x[1]/(x[0]+x[1])*100 ,'%')
df['Churn'].value_counts()


# In[15]:


df.dropna(inplace = True)
df2=df.iloc[:,1:]  #----------it will drop the first column that is customer ID which is irrelevent
#print(df2) it will print whole data set except first column


# In[16]:


#Convertin the predictor variable in a binary numeric variable
#df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
#df2['Churn'].replace(to_replace='No',  value=0, inplace=True)


# Convert the predictor variable into a binary numeric variable
df2['Churn'] = df2['Churn'].replace({'Yes': 1, 'No': 0})
df2['Churn'] = df2['Churn'].infer_objects(copy=False)



# In[17]:


df_dummies = pd.get_dummies(df2)
df_dummies.head()


# In[18]:


plt.figure(figsize=(10,6))
df_dummies.corr()["Churn"].sort_values(ascending=False).plot(kind="bar")
#plt.savefig("dataVisualisation/correlation.png", dpi=300)
plt.show()


# In[19]:


#data exploration


# In[20]:


# for demographics data 


# In[21]:


print(len(df_dummies))


# In[22]:


print(df['gender'].value_counts()*100.0/ len(df))


# In[23]:


colors = ['#4D3415','#E4512B']
ax= (df['gender'].value_counts()*100 /len(df)).plot(kind='bar',rot=0 ,color=colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())  #--------it will format the yaxis into the percentage format
#ax.xaxis.set_major_formatter(mtick.PercentFormatter())

ax.set_xlabel('Gender')
ax.set_ylabel('% Customer')
ax.set_title('Gender Distribution')
#plt.savefig("dataVisualisation/genderDistribution.png", dpi=300)


# In[24]:


df2[['gender','Churn']].groupby(['gender']).mean()


# In[25]:


#above reflects that about 26% each male and female have churned


# In[26]:


print(df['SeniorCitizen'].value_counts()*100.0/ len(df))


# In[27]:


ax = (df['SeniorCitizen'].value_counts()*100.0 / len(df))\
.plot.pie(autopct='%.1f%%', labels = ['No' , 'Yes'] ,figsize =(5,5) , fontsize = 12) 

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('Senior Citizens %' ,fontsize = 12)

ax.set_title('% of senior citizen' , fontsize = 12)
#plt.savefig("dataVisualisation/SeniorCitizenDistribution.png", dpi=300)


# In[28]:


df2[['Partner' , 'Churn']].groupby(['Partner']).mean()


# In[29]:


# above indicate that 32.98% of customers without a partner have churned.
#and 19.72% of customers with a partner have churned.


# In[30]:


print(df.columns)


# In[31]:


#print(df.columns.value_counts().shape[0])


# In[32]:


columns = df.columns  #will findout the features which would be having the 2 values exact
binary_cols = []

for col in columns:
    if df[col].value_counts().shape[0] ==2:
        binary_cols.append(col)

binary_cols  #categorical features with two classes


# In[33]:


df2[['Dependents' , 'Churn']].groupby(['Dependents']).mean()


# In[34]:


df2[['PhoneService' , 'Churn']].groupby(['PhoneService']).mean()


# In[35]:


df2[['PaperlessBilling' , 'Churn']].groupby(['PaperlessBilling']).mean()


# In[36]:


print(df['InternetService'].value_counts())


# In[37]:


#now we will analyse the features which is having more than 2 features:
#internet service
colors = ['#4D3425','#E4512B' ,'#3E5124']
#ax=(df['InternetService'].value_counts()*100.0 /len(df)).plot(kind='bar',stacked=True,rot=0,color=colors)
#ax.yaxis.set_major_formatter(mtick.PercentFormatter())

ax=(df['InternetService'].value_counts()).plot(kind='bar',stacked = True,rot=0,color=colors)


ax.set_ylabel(' Count' )
ax.set_title('Internet Service')

ax.set_xlabel('InternetService')
#plt.savefig("dataVisualisation/InternetServiceDistribution.png", dpi=300)


# In[38]:


df2[['InternetService' , 'Churn']].groupby('InternetService').mean()


# In[39]:


colors = ['#4D3425','#E4512B' ,'#3E5124']
ax=(df['StreamingTV'].value_counts()).plot(kind='bar',color=colors,rot=0)
ax.set_ylabel('Counts')
ax.set_title('StreamingTV')
#plt.savefig("dataVisualisation/StreamingTVDistribution.png", dpi=300)


# In[40]:


df2[['StreamingTV' , 'Churn']].groupby('StreamingTV').mean()


# In[41]:


ax = (df['TechSupport'].value_counts()*100.0 / len(df))\
.plot.pie(autopct='%.1f%%', labels = ['No' , 'Yes' , 'Internet Service'] ,figsize =(5,5) , fontsize = 12) 
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('Count %')
ax.set_title('TechSupport')
#plt.savefig("dataVisualisation/TechSupportDistribution.png", dpi=300)


# In[42]:


df2[['TechSupport' ,'Churn']].groupby('TechSupport').mean()


# In[43]:


ax = (df['OnlineSecurity'].value_counts()*100.0/len(df))\
.plot.pie(autopct='%.1f%%', labels=['Yes','No','OnlineSecurity'],figsize = (5,5),fontsize=12)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('count%')
ax.set_title('OnlineSecurity')
ax.set_xlabel('Online_Security')
#plt.savefig("dataVisualisation/Online_SecurityDistribution.png", dpi=300)


# In[44]:


df2[['OnlineSecurity' , 'Churn']].groupby('OnlineSecurity').mean()


# In[45]:


ax=sns.displot(df['tenure'],binwidth=2,color = 'red',bins=30,kde = True)
plt.xlabel('Tenure (months)')
plt.ylabel('# of Customers')
plt.title('# of Customers by their tenure')
#plt.savefig("dataVisualisation/TenureDistribution.png", dpi=300)


# In[46]:


ax=df['Contract'].value_counts().plot(kind='bar',rot=0,width=0.3)
ax.set_ylabel('# of Customers')
ax.set_title('# of Customers by Contract Type')
#plt.savefig("dataVisualisation/ContractDistribution.png", dpi=300)


# In[47]:


ax1 = sns.displot(df[df['Contract']=='Month-to-month']['tenure'] , binwidth=2,color = 'lightblue',bins=30)
plt.xlabel('Tenure (months)')
plt.ylabel('# of Customers')
plt.title('Month to Month Contract')
#plt.savefig("dataVisualisation/Month_to_MonthDistribution.png", dpi=300)


# In[48]:


#the customer who is taking the month to month contract are having the short tenures and less loyal


# In[49]:


ax1=sns.displot(df[df['Contract']=='One year']['tenure'],binwidth=2,color='red',bins=30)

plt.xlabel('Tenure (months)')
plt.ylabel('# of Customers')
plt.title('One Year Contract')
#plt.savefig("dataVisualisation/oneYearDistribution.png", dpi=300)


# In[50]:


ax2=sns.displot(df[df['Contract']=='Two year']['tenure'],binwidth=2,color='green',bins=30)
plt.xlabel('Tenure (months)')
plt.ylabel('# of Customers')
plt.title('Two Year Contract')
#plt.savefig("dataVisualisation/two_YearDistribution.png", dpi=300)


# In[51]:


df.columns.values


# In[52]:


columns = df.columns  #will findout the features which would be having the 2 values exact
binary_cols1 = []

for col in columns:
    if df[col].value_counts().shape[0] ==3:
        binary_cols1.append(col)

binary_cols1  #categorical features with two classes


# In[53]:


services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,20))
for i, item in enumerate(services):
    if i < 3:
        ax = df[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0)
        
    elif i >=3 and i < 6:
        ax = df[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0)
        
    elif i < 9:
        ax = df[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0)
    ax.set_title(item)
  #  plt.savefig("dataVisualisation/PhoneServiceDistribution.png", dpi=300)
   # plt.savefig("dataVisualisation/MultipleLinesDistribution.png", dpi=300)
    #plt.savefig("dataVisualisation/OnlineBackupDistribution.png", dpi=300)
    #plt.savefig("dataVisualisation/DeviceProtectionDistribution.png", dpi=300)
    #plt.savefig("dataVisualisation/StreamingMoviesDistribution.png", dpi=300)


# In[54]:


df[['MonthlyCharges','TotalCharges']].plot.scatter(x='MonthlyCharges',y='TotalCharges')
#plt.savefig("dataVisualisation/month_VS_totalCharges.png", dpi=300)


# In[55]:


#now we will ckeck the skewness of our predicted data
# and will check the behaviour of rest of the field with churn


# In[56]:


x=list(df.Churn.value_counts())
print('No' , x[0]/(x[0]+x[1])*100 ,'%')
print('Yes' , x[1]/(x[0]+x[1])*100 ,'%')
df['Churn'].value_counts()


# In[57]:


colors = ['#4D3425','#E4512B']

ax=(df['Churn'].value_counts()*100.0/len(df)).plot(kind='bar' , color=colors,rot=0,stacked=True,fig=(4,2))

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers',size = 14)
ax.set_xlabel('Churn',size = 14)
ax.set_title('Churn Rate', size = 14)
#plt.savefig("dataVisualisation/ChurnRate.png", dpi=300)


# In[58]:


# now we will do eda of all fields with churn


# In[59]:


sns.boxplot(x=df.Churn, y=df.tenure)
#plt.savefig("dataVisualisation/Tenure_vs_churn.png", dpi=300)


# In[60]:


colors = ['#4D3425','#E4512B']

contract_churn=df.groupby(['Contract','Churn']).size().unstack()
print(contract_churn)


# In[61]:


colors = ['#4D3425','#E4512B']

contract_churn=df.groupby(['Contract','Churn']).size().unstack()

print(contract_churn.T)


# In[62]:


colors = ['#4D3425','#E4512B']
contract_churn = df.groupby(['Contract','Churn']).size().unstack(level=0)

print(contract_churn*100.0/ contract_churn.sum())


# In[63]:


colors = ['#4D3425', '#E4512B', '#1F77B4', '#FF7F0E']

contract_churn = df.groupby(['Contract','Churn']).size().unstack()

ax=(contract_churn.T*100.0/ contract_churn.T.sum()).T.plot(kind='bar',width=0.3,stacked=True,rot=0,figsize=(10,6),color=colors)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

ax.legend(loc='best',prop={'size':10},title = 'Churn')

ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Contract Type',size = 14)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#plt.savefig("dataVisualisation/churn_vs_contractType.png", dpi=300)


# In[64]:


df.MonthlyCharges.value_counts()


# In[65]:


ax=sns.kdeplot(df.MonthlyCharges[(df['Churn'] == 'No')],color="Red", shade = True)

ax=sns.kdeplot(df.MonthlyCharges[(df['Churn'] == 'Yes')],color='blue',shade=True)

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')
#plt.savefig("dataVisualisation/Monthly_ChargesDistribution.png", dpi=300)


# In[66]:


#we can see that highest % of customer churned when the monthly charges are high
#below method also we can use it to visualise the same using histogram


# In[67]:


# Define the data
no_churn = df.MonthlyCharges[df['Churn'] == 'No']
yes_churn = df.MonthlyCharges[df['Churn'] == 'Yes']

# Create the plot
plt.figure(figsize=(10, 6))

# Plot histograms
plt.hist(no_churn, bins=30, color='red', alpha=0.5, label='Not Churn')
plt.hist(yes_churn, bins=30, color='blue', alpha=0.5, label='Churn')

# Add labels and title
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.title('Distribution of Monthly Charges by Churn')
plt.legend(loc='upper right')

# Show the plot
plt.show()


# In[68]:


df.TotalCharges.value_counts().head()


# In[69]:


# Define the data
no_churn = df.TotalCharges[df['Churn'] == 'No']
yes_churn = df.TotalCharges[df['Churn'] == 'Yes']

# Create the plot
plt.figure(figsize=(10, 6))

# Plot histograms
plt.hist(no_churn, bins=30, color='red', alpha=0.5, label='Not Churn')
plt.hist(yes_churn, bins=30, color='blue', alpha=0.5, label='Churn')

# Add labels and title
plt.xlabel('TotalCharges')
plt.ylabel('Frequency')
plt.title('Distribution of TotalCharges by Churn')
plt.legend(loc='upper right')

# Show the plot
plt.show()

#result --churn rate is high when total charges are low
#plt.savefig("dataVisualisation/Totaol_charges_Distribution.png", dpi=300)


# In[70]:


df2[['SeniorCitizen','Churn']].groupby('SeniorCitizen').mean()


# In[71]:


colors = ['#4D3425','#E4512B']

sen_churn=df.groupby(['SeniorCitizen','Churn']).size().unstack()
ax=(sen_churn.T*100.0/sen_churn.T.sum()).T.plot(kind='bar',stacked=True,rot=0,width = 0.2,figsize=(8,6),color=colors)

ax.legend(['No','Yes'],title='Churn',loc='center')

ax.set_ylabel('% Customers')
ax.set_title('Churn by Seniority Level',size = 14)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
#plt.savefig("dataVisualisation/churn_vs_seniority.png", dpi=300)


# In[72]:


df_dummies = pd.get_dummies(df2)
df_dummies.head()


# In[73]:


#now  EDA is done  now we will go with some predictive models and compare their performances.
# We will use the data frame where we had created dummy variables

#1. Logistic Regression


# In[74]:


Y=df_dummies['Churn'].values
print(Y)


# In[75]:


X=df_dummies.drop(columns=['Churn'])


# In[76]:


"""The fit(data) method is used to compute the mean and std dev for a given feature so that it can be used further for scaling.
The transform(data) method is used to perform scaling using mean and std dev calculated using the .fit() method.
The fit_transform() method does both fit and transform."""


# In[77]:


Y=df_dummies['Churn'].values
X=df_dummies.drop(columns=['Churn'])
from sklearn.preprocessing import MinMaxScaler

features=X.columns.values
scaler=MinMaxScaler(feature_range=(0,1))
model=scaler.fit(X) #will find mean and sd
scaled_model=model.transform(X)  #will transform the data into same range i.e., scaling of data will be done
X.columns=features


# In[78]:


print(features)


# In[79]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.3,random_state=101)


# In[80]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', max_iter=200)
result = model.fit(X_train, y_train)


# In[233]:


import joblib

# Save the model to a file
joblib.dump(model, 'model.joblib')


# In[234]:


model = joblib.load('model.joblib')


# In[81]:


from sklearn import metrics

prediction_test=model.predict(X_test) #
#y_prediction_test_proba = model.predict_proba(X_test)
print('test :',metrics.accuracy_score(y_test,prediction_test))
prediction_test1=model.predict(X_train)
print('train :',metrics.accuracy_score(y_train,prediction_test1))


# In[ ]:





# In[82]:


def confusion_matrix_plot(X_train, y_train, X_test, y_test, y_pred, classifier, classifier_name):
    cm = confusion_matrix(y_pred,y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot()
    plt.title(f"Confusion Matrix - {classifier_name}")
    plt.show()
    
    print(f"Accuracy Score Test = {accuracy_score(y_pred,y_test)}")
    print(f"Accuracy Score Train = {classifier.score(X_train,y_train)}")
    return print("\n")


# In[83]:


confusion_matrix_plot(X_train,y_train,X_test,y_test, prediction_test,model,"Logistic Regression")

#plt.savefig("dataVisualisation/logistic_confusion_matrix.png", dpi=300)


# In[84]:


model.coef_[0]


# In[85]:


Columns=X.columns.values
print(Columns)


# In[86]:


weights = pd.Series(model.coef_[0], index=X.columns.values)

# Sort and plot the top 10 coefficients as a line plot
weights.sort_values(ascending=False)[:10].plot(kind='line', marker='o')
plt.title('Top 10 Feature Weights')
plt.xlabel('Feature')
plt.ylabel('Weight')
plt.xticks(rotation=45)
plt.show()



# In[87]:


weights=pd.Series(model.coef_[0],index=X.columns.values)

print(weights.sort_values(ascending=False)[:10].plot(kind='bar'))
#plt.savefig("dataVisualisation/weights_first_10.png", dpi=300)


# In[88]:


weights=pd.Series(model.coef_[0],index=X.columns.values)
print(weights.sort_values(ascending=False)[-10:].plot(kind='bar'))
#plt.savefig("dataVisualisation/wt_last_10.png", dpi=300)
#Negative relation means that likeliness of churn decreases with that variable


# In[89]:


first_X_test_record = X_test.iloc[1]
first_y_test_record = y_test[1]
print(first_X_test_record)
print(first_y_test_record)



# In[90]:


# Extracting one record from X_test and reshaping it
single_X_test_record = X_test.iloc[1].values.reshape(1, -1) # Reshape to 2D array

# Making the prediction
prediction_test = model.predict(single_X_test_record)

print(prediction_test)


# In[138]:


import joblib

# Save the model to a file
joblib.dump(model, 'model.joblib')


# In[139]:


# Load the model from the file
model = joblib.load('model.joblib')


# In[134]:


import sklearn
print(sklearn.__version__)


# In[145]:


get_ipython().system('pip install google-cloud-storage')


# In[245]:


import os
from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def upload_directory(bucket_name, source_directory, destination_directory):
    """Uploads all files in a directory to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            blob_path = os.path.join(destination_directory, file)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(file_path)
            print(f"File {file_path} uploaded to {blob_path}.")

# Example usage
bucket_name = 'churn-model-prediction'
source_directory = '/home/jupyter/ChurnPropensity/dataVisualisation'
destination_directory = 'VizualisationForLooker'

upload_directory(bucket_name, source_directory, destination_directory)


# In[ ]:





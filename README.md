# Churn Users Prediction using Supervised & Unsupervised Machine Learning
# I. Introduction
## 1. About Supervised Machine Learning:
- Supervised machine learning uses labeled data to train models, where input-output pairs are provided, enabling the model to predict outcomes for new, unseen data.
- Several application of unsupervised machine learning:
    - Fraud detection in financial systems.
    - Predictive maintenance in industries.
    - Medical diagnosis based on patient data.
## 2. About Unsupervised Machine Learning:
- Unsupervised machine learning involves training models on data without labeled outcomes, aiming to identify patterns, structures, or groupings within the dataset.
- Several application of unsupervised machine learning:
    - Customer segmentation for targeted marketing.
    - Anomaly detection in fraud detection or cybersecurity.
    - Data compression and visualization.
## 3. Project Purpose:
- Detect churned users and their likely patterns and behaviors using supervised machine learning.
- Segment the churn users into distinct groups through an unsupervised machine learning model.
- Provide insights about the differencies between the groups.
# II. Identify Churned Users by Supervised Machine Learning Model
## 1. Data & Libraries Preparation:
  ![image](https://github.com/user-attachments/assets/900981ca-5018-4fd9-b5f2-25d93a81c010)
## 2. Exploratory Data Analysis - EDA:
### 2.1. Explore Data:
  ![image](https://github.com/user-attachments/assets/e16c05bd-c4a5-4010-947d-8b8fabef242e)
  ![image](https://github.com/user-attachments/assets/5e4f9764-a8fc-4175-8706-36d0d5f8c5e7)
  ![image](https://github.com/user-attachments/assets/94940db7-e895-42bd-93ad-e28e49fbb02f)
### 2.2. Check & Handle Missing and Duplicate Values:
#### 2.2.1. Handle Missing Values:
  ![image](https://github.com/user-attachments/assets/e0beab59-378d-467c-a4a3-8b2cdc379b06)
  ![image](https://github.com/user-attachments/assets/39484b54-2f2a-4132-91b8-12357b9d606c)
  - Conlusion:
    - Tenure: 264 values
    - WarehouseToHome: 251 missing values
    - HourSpendOnApp: 255 missing values
    - OrderAmoutHikeformLastYear: 265 missing values
    - CouponUsed: 256 missing values
    - OrderCount: 258 missing values
    - DaySinceLastOrder: 307 missing values
  - Solution: Filling missing values with respectively median values.
#### 2.2.2 Handle Duplicate Values:
  ![image](https://github.com/user-attachments/assets/217344a1-32a8-44be-9886-c9e657da28e6)
  ![image](https://github.com/user-attachments/assets/7f70ba27-cd0c-4e68-98be-6e9399e552f1)
  - Conclusion:
    - Primal Key: CustomerID
    - Duplicates in primal key: None
    - Duplicates in sample: None
  - Solution: No action needed.
### 2.3. Univariate Analysis:
#### 2.3.1. Correct Data Type:
  ![image](https://github.com/user-attachments/assets/b2faf1dc-7f88-419f-9003-320cfb8e9b8f)
#### 2.3.2. Numerical Data Analysis:
a. Select correct numerical data columns:
  ![image](https://github.com/user-attachments/assets/cb3f510b-4cc4-4c9d-ad00-f5c596dfea06)
b. Checking value distribution using clustered column:
  ![image](https://github.com/user-attachments/assets/a54faa67-65d0-4b31-8bfe-ded88ed58661)
  - Example of chart:
  ![image](https://github.com/user-attachments/assets/de8edff8-d59e-4c75-835a-d0a9a09dbb50)

c. Checking value distribution using boxplot:
  ![image](https://github.com/user-attachments/assets/af468909-5c2b-4c64-9eb5-6d8d33b5c1f1)
  - Example of chart:
  ![image](https://github.com/user-attachments/assets/f4078814-83db-4f93-8ef0-d9067410e1af)

d. Result:
  ![image](https://github.com/user-attachments/assets/2cd43d07-6354-4fbf-a62b-4a1099ed7f4b)
e. Solution:
  ![image](https://github.com/user-attachments/assets/85f06ebd-eeeb-4761-9e7c-a6c7e64f9d3f)
  - Solution: All outliers are replaced by their respective median values.
#### 2.3.3. Categorical Data Analysis:
a. Checking value distribution using clustered column:
  ![image](https://github.com/user-attachments/assets/1237ab33-7ada-46ec-acdb-f2457a3e70c5)
  - Example of chart:
  ![image](https://github.com/user-attachments/assets/6caeb8a2-c655-485f-8121-8fb5d1e87773)

b. Correcting data value:
  ![image](https://github.com/user-attachments/assets/38b11f5c-04c9-4e6f-9573-678281ef7dc4)
### 2.4. Bivariate & Multivariate Analysis:
#### 2.4.1. Correlation Coefficient:
  ![image](https://github.com/user-attachments/assets/64e7013d-3a1f-4e71-8e64-1610302817ac)
  - Variables with positive correlation with churn users:
    - Notable relationship: Complain.
  - Variables with negative correlation with churn users:
    - Medium Relationship: Tenure.
#### 2.4.2. Numerical Variables Relationship:
  ![image](https://github.com/user-attachments/assets/3fe04dbb-5349-45af-be75-e9ad823baddf)
  ![image](https://github.com/user-attachments/assets/241139af-2079-4879-97ac-2e1fa81cffdb)
#### 2.4.3. Categorical Variables Relationship:
  ![image](https://github.com/user-attachments/assets/4a0d4116-2303-41c5-abcb-0e0d842d67d2)
  ![image](https://github.com/user-attachments/assets/0ee2ef0b-35c2-4fbe-827c-dd559cdba7ae)
  ![image](https://github.com/user-attachments/assets/b5caabfd-0f90-4075-a18d-89f0c6cea96a)
  ![image](https://github.com/user-attachments/assets/05285b36-fd30-43b4-b4ae-1cd28b28e3eb)
  ![image](https://github.com/user-attachments/assets/82860ce9-16cd-4c4b-934b-3f36f1f650cf)
  ![image](https://github.com/user-attachments/assets/7bf97420-e83f-4def-bcde-a41804701ee2)
  ![image](https://github.com/user-attachments/assets/b29e4a39-7d71-4c29-82b5-a53858a671a7)
## 3. Feature Engineering:
  ![image](https://github.com/user-attachments/assets/80818c2b-9cb2-4e74-83a8-291c05a5a446)
  - Select potetial features for the model.
## 4. Feature Transforming:
  ![image](https://github.com/user-attachments/assets/ccc1780d-4aaa-4558-9b31-f394fb6add94)
  - Encode selected features for later progression.
## 5. Model Training:
### 5.1. Split Dataset:
  ![image](https://github.com/user-attachments/assets/5ef3eae7-defb-4160-b313-3ce4dc65cd1d)
### 5.2. Normalization:
  ![image](https://github.com/user-attachments/assets/23727880-31c7-48fc-a4a2-cb4c24fd4aea)
### 5.3. Apply Model:
  ![image](https://github.com/user-attachments/assets/df58c747-fdf5-40e5-b000-c36c512a6fe0)
  - Apply Random Forest to train the model.
## 6. Model Evaluation:
  ![image](https://github.com/user-attachments/assets/4f150615-e0ee-43be-a220-adf11fa2aa75)
  ![image](https://github.com/user-attachments/assets/04a887af-d1d9-401f-9e6a-a4d2ffb6e4f3)
  - Evaluate the model using Precision Score, F-1 Score & Model Fitting.
  - The model is considerably accuracy as it achives 75&-95% on both train and test dataset in 3 tests.
## 7. Model Improvement:
  ![image](https://github.com/user-attachments/assets/a57cc591-0c3a-403e-bfef-48e26e158e1d)
  - The model achives highest accuracy with the following super-parameters: {'class_weight': 'balanced', 'max_depth': None, 'n_estimators': 200}
# III. Segment Churned Users Into Groups Unsupervised Machine Learning Model:
## 1. Data & Libraries Preparation:
  ![image](https://github.com/user-attachments/assets/593b673b-260d-473a-b459-ec5081b16feb)
  - Only churned users are selected as the unsupervised learning is aimed at this customer segment.
## 2. Exploratory Data Analysis - EDA:
  - The EDA step is not required as it has been done on the previous Supervised Model part.
## 3. Feature Engineering & Feature Transforming:
### 3.1. Select Potential Features:
  ![image](https://github.com/user-attachments/assets/0f2b92f3-90cd-4f77-b40a-5fe3d0a47433)
### 3.2. Encoding:
  ![image](https://github.com/user-attachments/assets/ad00485b-5630-4723-ae0b-e39f6b2adc7c)
### 3.3. Normalization:
  ![image](https://github.com/user-attachments/assets/deaaeaea-1d08-4d6c-a1e2-9e080fe276d1)
  - As the results of normalization cannot be used in Dimension Reduction (PCA) due to low explained variance rate, the this step is only shown as an example.
### 3.4. Dimension Reduction:
  ![image](https://github.com/user-attachments/assets/0a479fae-54ad-4e32-bb14-d0cebfdb0571)
  - With high explained variance rate (97.9%), the result of PCA can be used later for unsupervised machine learning model.
## 4. Model Training - Apply K-Means Model:
### 4.1. Choosing K - Elbow Method:
  ![image](https://github.com/user-attachments/assets/6ec23da1-4025-4320-a0f2-9ccef4cfb025)
### 4.2. Applying K-Means:
  ![image](https://github.com/user-attachments/assets/d848a151-99e8-406f-9eda-887704acff13)
## 5. Model Evaluation:
  ![image](https://github.com/user-attachments/assets/588907f3-5476-495f-b3f7-af440d03fb3f)

### 5.1. Silhouette Score:
  ![image](https://github.com/user-attachments/assets/a57bf600-b396-4042-a019-30e7ffa4d4ee)
### 5.2. Distribution Of Clusters:
  ![image](https://github.com/user-attachments/assets/ae066dbf-c815-4e32-9ebc-430afa2133a4)
## 6. Analyzation:
- To get information about the clusters, a Supervised model is applied.
### 6.1. Select Potential Features:
  ![image](https://github.com/user-attachments/assets/e2cd70e4-d757-4e2b-beb6-7bff4fa63f0b)
### 6.2. Encoding:
  ![image](https://github.com/user-attachments/assets/b057606d-a61b-4d65-8523-bf01d48f4a93)
### 6.3. Split Dataset into Train & Test Set:
  ![image](https://github.com/user-attachments/assets/64da4619-eafd-4d1b-901c-8e90c15c8c32)
### 6.4. Normalization:
  ![image](https://github.com/user-attachments/assets/41a01173-4b61-46fd-9bc6-1b185bacde3b)
### 6.5. Model Training using Random Forest:
  ![image](https://github.com/user-attachments/assets/89956b77-a3cc-45fb-bfd1-c9d22c589819)
### 6.6. Model Evaluation:
  ![image](https://github.com/user-attachments/assets/0156a958-ae08-4679-9391-04f8c94fafee)
### 6.7. Feature Importance:
  ![image](https://github.com/user-attachments/assets/5959a947-23e0-49a6-90db-3aaf8d2e4264)
  - Top features with greatest effect on deciding churned users, in order:
    - CashbackAmount
    - OrderCount
    - PreferedOrderCat
### 6.8. Cluster Analyzation:
#### 6.8.1. Distribution of Cashback Amount Across Clusters:
  ![image](https://github.com/user-attachments/assets/d3f01b3d-3707-491b-b694-7d8bf4dae60f)
  ![image](https://github.com/user-attachments/assets/22bf3a7b-93c2-49cf-961d-02df03416112)
#### 6.8.2. Distribution of Order Count Across Clusters:
  ![image](https://github.com/user-attachments/assets/f2cd650b-f536-48cf-8366-a013f8161e9c)
  ![image](https://github.com/user-attachments/assets/649d73ff-cb46-4a00-bb4c-a883d64676f6)
#### 6.8.3. Distribution of Preferred Order Category Across Clusters:
  ![image](https://github.com/user-attachments/assets/3fa125dc-fe09-4ad2-aefe-b8600bbdd90d)
  ![image](https://github.com/user-attachments/assets/e3d2de23-1f90-4079-9844-a794005cc726)
# IV. Model Status
## 1. Supervised Machine Learning Model:
  - Random Forest: High Precision, Fitting & Effective.
  - Accuracy of Random Forest Fine-Tuned: 91.72%.
## 2. Unsupervised Machine Learning Model:
  - K-means Model. K = 4
  - Silhouetter Score: 0.46
# V. Insights
## 1. Common features of churned users:
  - The users who have less that 15 months of tenure
  - The users who complained in the last month
  - The users who login by computer
  - The users who stay in city tier 2 or 3
  - The users who pay by Cash on Delivery or E-wallet
  - The users who had preferred order category of Fashion or Mobile Phone
  - The users with big satisfaction score
  - The users who are currently single
## 2. Characteristics of churned users' groups:
  - Group 1 (Cluster 0) - people with:
    - Low average cashback amount in last month (145-155).
    - Lower number of orders placed in last month (from 1-3 orders).
    - Prefer mobile phone and laptop & accessory in last month.

  - Group 2 (Cluster 1) - people with:
    - High average cashback amount in last month (200 - 230).
    - Higher number of orders placed last month (from 2-5 orders).
    - Prefer laptop & accessory and fashion in the last month.
   
  - Group 3 (Cluster 2) - people with:
    - Low average cashback amount in last month (120 - 130).
    - Lower number of orders placed last month (from 1-2 orders).
    - Prefer mobile phone in the last month.

  - Group 4 (Cluster 3) - people with:
    - Medium average cashback amount in last month (165 - 190).
    - Higher number of orders placed last month (from 2-4 orders).
    - Prefer laptop & accessory, mobile phone and fashion in the last month.
# VI. Recommendation:
  - Several factors of users that cannot be changed or improved: MaritalStatus, Tenure. The company is not able to do anything regarding these factors.
  - On the other hand, there are other factors can be improved by the company to reduce the number of churn users:
    - Complain: The business need to improve the quality of product and customer service, solve the problems as they rise so the customers will be satisfied and may not complain.
    - LoginDevice: The business need to test and fix any bugs or technical problems that affect the experience of customers login using computer
    - CityTier: The business need to invest more on marketing in city tier 1, reduce investment on city tier 2 and 3.
    - PaymentMethod: The business need to find and resolve any inconvenience in using Cash on Delivery and E-wallet to make purchase, in order to improve the purchasing experience of users in these payment methods.
    - Preferred Category: The business need to improve the quality of product in Fashion and Mobile Phone segments.
    - Satisfaction Score: On this factor, it is needed to check if the score is in an ascending or descending order. In case the score is sorted in ascending order, this is a strange phenomenon that should be monitoring further.






















































  











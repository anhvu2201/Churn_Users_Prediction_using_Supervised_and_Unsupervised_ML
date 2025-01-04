# Churn Users Prediction using Unsupervised Machine Learning
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
  ![image](https://github.com/user-attachments/assets/f3d053f9-c934-43da-b985-94c0eaf7d040)
## 2. Exploratory Data Analysis - EDA:
### 2.1. Explore Data:
  ![image](https://github.com/user-attachments/assets/85ae4c69-b8ef-44d3-8731-0cbcfdc334c4)
  ![image](https://github.com/user-attachments/assets/ab1a87c6-fc80-426c-ae6e-17b7af4f49f9)
### 2.2. Check & Handle Missing and Duplicate Values:
#### 2.2.1. Handle Missing Values:
  ![image](https://github.com/user-attachments/assets/3b2f077b-d104-4b9a-9de7-851fa68c0f1f)
  ![image](https://github.com/user-attachments/assets/9709ecd3-2248-4535-a561-49857bc29670)
  - Solution: Filling missing values with respectively median values.
#### 2.2.2 Handle Duplicate Values:
  ![image](https://github.com/user-attachments/assets/db786ad8-aa4a-4290-adb9-4972cfdf59c3)
  ![image](https://github.com/user-attachments/assets/c7414137-b17c-495b-8841-a00646f3a89c)
  - Solution: No action required
### 2.3. Univariate Analysis:
#### 2.3.1. Correct Data Type:
  ![image](https://github.com/user-attachments/assets/8b573d83-2920-4ef7-acfd-0075481089e2)
#### 2.3.2. Numerical Data Analysis:
a. Select correct numerical data columns:
  ![image](https://github.com/user-attachments/assets/af519ccb-ec9e-4d29-b306-49d48757afd2)
b. Checking value distribution using clustered column:
  ![image](https://github.com/user-attachments/assets/b76cb123-797f-46f9-9096-86fd3d651480)
  - Example of chart:
  ![image](https://github.com/user-attachments/assets/de8edff8-d59e-4c75-835a-d0a9a09dbb50)

c. Checking value distribution using boxplot:
  ![image](https://github.com/user-attachments/assets/aba2ca58-c094-448a-8662-157a069926e4)
  - Example of chart:
  ![image](https://github.com/user-attachments/assets/f4078814-83db-4f93-8ef0-d9067410e1af)

d. Result:
  ![image](https://github.com/user-attachments/assets/631e4572-3b17-4df5-9662-0cb59c366f6f)
e. Solution:
  ![image](https://github.com/user-attachments/assets/f740e443-7fb9-4be8-9da5-3abed4f84352)
#### 2.3.3. Categorical Data Analysis:
a. Checking value distribution using clustered column:
  ![image](https://github.com/user-attachments/assets/0082059e-f52a-439d-b9d9-5a6656b6f13f)
  - Example of chart:
  ![image](https://github.com/user-attachments/assets/6caeb8a2-c655-485f-8121-8fb5d1e87773)

b. Correcting data value:
  ![image](https://github.com/user-attachments/assets/1a592e42-0759-4ada-a6d6-26fdfe18a69b)
### 2.4. Bivariate & Multivariate Analysis:
#### 2.4.1. Correlation Coefficient:
  ![image](https://github.com/user-attachments/assets/1691d6dd-2071-4d3e-8a05-db0e0d597a18)
  ![image](https://github.com/user-attachments/assets/4fae6607-6c51-4874-ae4d-cc071ce516ee)
#### 2.4.2. Numerical Variables Relationship:
  ![image](https://github.com/user-attachments/assets/da27e975-9d1c-4499-b36a-0d6dd6a649cb)
  ![image](https://github.com/user-attachments/assets/aadeb87d-c098-408c-b932-a8a6c55549d6)
#### 2.4.3. Categorical Variables Relationship:
  ![image](https://github.com/user-attachments/assets/ed17bb6b-8d39-467e-868c-a79f6fcc95ce)
  ![image](https://github.com/user-attachments/assets/e9d273bc-f656-4875-9817-d89f9ebd89f9)
  ![image](https://github.com/user-attachments/assets/79069399-e082-4e30-977d-c190a809a709)
  ![image](https://github.com/user-attachments/assets/21f8916f-d064-4cbb-bd35-40a15429a346)
  ![image](https://github.com/user-attachments/assets/a78326f9-a9f5-4aa8-a369-dba11a9853ff)
  ![image](https://github.com/user-attachments/assets/ee677fe0-44ad-4727-893a-d0455afc3a75)
  ![image](https://github.com/user-attachments/assets/52bd3dda-5f11-471d-afef-1e2f0fa7f080)
## 3. Feature Engineering:
  ![image](https://github.com/user-attachments/assets/c64a16c3-3575-48e4-a9ef-1ed167ef6a2f)
  - Select potetial features for the model.
## 4. Feature Transforming:
  ![image](https://github.com/user-attachments/assets/a054dc9e-01ce-4a5e-9956-7cc5a2f53ce4)
  - Encode selected features for later progression.
## 5. Model Training:
### 5.1. Split Dataset:
  ![image](https://github.com/user-attachments/assets/e4076857-460e-4cb8-bf4b-e867db3aa9c1)
### 5.2. Normalization:
  ![image](https://github.com/user-attachments/assets/d6a6c946-1e03-4e7f-90b1-0bdc93c5988e)
### 5.3. Apply Model:
  ![image](https://github.com/user-attachments/assets/ff4bc2ee-9605-425d-b0ac-b5bb2dbc80fa)
  - Apply Random Forest to train the model.
## 6. Model Evaluation:
  ![image](https://github.com/user-attachments/assets/b8506463-f07a-4435-b321-c02a6db060db)
  - Evaluate the model using Precision Score, F-1 Score & Model Fitting.
  - The model is considerably accuracy as it achives 75&-95% on both train and test dataset in 3 tests.
## 7. Model Improvement:
  ![image](https://github.com/user-attachments/assets/2b806328-ba50-4fa3-95c8-5cf19ebe816c)
  - The model achives highest accuracy with the following super-parameters: {'class_weight': 'balanced', 'max_depth': None, 'n_estimators': 200}
# III. Segment Churned Users Into Groups Unsupervised Machine Learning Model:
## 1. Data & Libraries Preparation:
  ![image](https://github.com/user-attachments/assets/04e6ac70-c643-414e-8f44-053eee3a407b)
## 2. Exploratory Data Analysis - EDA:
  - The EDA step is not required as it has been done already on the Supervised Model part.
## 3. Feature Engineering & Feature Transforming:
### 3.1. Select Potential Features:
  ![image](https://github.com/user-attachments/assets/13f17737-76d9-4694-b08f-6cdcb5c3ca01)
### 3.2. Encoding:
  ![image](https://github.com/user-attachments/assets/d9bba210-9bd8-4574-aa83-e9b368b88ada)
### 3.3. Normalization:
  ![image](https://github.com/user-attachments/assets/7230be14-2e73-47e3-81c7-1b488156a23d)
  - As the results of normalization cannot be used in Dimension Reduction (PCA) due to low explained variance rate, the this step is only shown as an example.
### 3.4. Dimension Reduction:
  ![image](https://github.com/user-attachments/assets/6f507ccb-a6c4-457c-9a2e-ce1a73ed9d4a)
  - With high explained variance rate (97.9%), the result of PCA can be used late for unsupervised machine learning model.
## 4. Model Training - Apply K-Means Model:
### 4.1. Choosing K - Elbow Method:
  ![image](https://github.com/user-attachments/assets/c2aabd8f-7041-4d16-b80c-100c878b2e75)
### 4.2. Applying K-Means:
  ![image](https://github.com/user-attachments/assets/666441bc-f822-4e5b-b08b-7a62f24ef5b9)
## 5. Model Evaluation:
  ![image](https://github.com/user-attachments/assets/3667f8c2-f126-4634-bc5a-576796526c42)

### 5.1. Silhouette Score:
  ![image](https://github.com/user-attachments/assets/b7caf230-31ba-42fc-b5bd-b31525eb5531)
### 5.2. Distribution Of Clusters:
  ![image](https://github.com/user-attachments/assets/ec159452-59f3-4c22-884c-6338a0b237c5)
## 6. Analyzation:
- To get information about the clusters, a Supervised model is applied.
### 6.1. Select Potential Features:
  ![image](https://github.com/user-attachments/assets/7e3c4b58-83c2-406d-8841-6bf8512076b1)
### 6.2. Encoding:
  ![image](https://github.com/user-attachments/assets/2faf72a1-e583-4b8f-bde1-2de450a21ab9)
### 6.3. Split Dataset into Train & Test Set:
  ![image](https://github.com/user-attachments/assets/88765ef7-ece8-447d-8461-14f59d82df7d)
### 6.4. Normalization:
  ![image](https://github.com/user-attachments/assets/2cc666fd-9532-498c-85c9-0f5580afc97f)
### 6.5. Model Training using Random Forest:
  ![image](https://github.com/user-attachments/assets/d6d9fc12-ca61-42f7-9d3f-4fc8e0d62493)
### 6.6. Model Evaluation:
  ![image](https://github.com/user-attachments/assets/2af8b14a-90ef-41fa-891d-014888559711)
### 6.7. Feature Importance:
  ![image](https://github.com/user-attachments/assets/385e000b-bf53-4495-91c7-41a1a46cdb66)
  - Top features with greatest effect on deciding churned users, in order:
    - CashbackAmount
    - OrderCount
    - PreferedOrderCat
### 6.8. Cluster Analyzation:
#### 6.8.1. Distribution of Cashback Amount Across Clusters:
  ![image](https://github.com/user-attachments/assets/76d21de2-a419-4ab6-bc9b-fa170401cfd9)
  ![image](https://github.com/user-attachments/assets/65b1cd4c-1815-421b-820a-21321b2d120b)
#### 6.8.2. Distribution of Order Count Across Clusters:
  ![image](https://github.com/user-attachments/assets/282864cc-0f87-45c3-830d-459097125ef8)
  ![image](https://github.com/user-attachments/assets/f1488ed0-b5cc-4985-af75-778c80e7c245)
#### 6.8.3. Distribution of Preferred Order Category Across Clusters:
  ![image](https://github.com/user-attachments/assets/b7f635ed-b6ab-4d80-b203-c4a626a5f35a)
  ![image](https://github.com/user-attachments/assets/116e27e2-39e7-46a3-bca6-3f8f382eb8c3)
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






















































  











import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.title("Data scientists salaries")

upload_file = st.file_uploader('Upload a file containing earthquake data')


classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

salary_cap = st.sidebar.number_input( 'Salary in USD ' , 40000 , 500000)

# Check to see if a file has been uploaded
if upload_file is not None:
    # If it has then do the following:

    
    # Read the file to a dataframe using pandas
    salaries = pd.read_csv(upload_file)
    
    # Create a section for the dataframe statistics
    st.header('Statistics of Dataframe')
    st.write(salaries.describe())

    
    
    salaries = salaries[salaries.salary<salary_cap]
    # Create a section for the dataframe header
    st.header('Header of Dataframe')
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error


    X = salaries['salary'].values.reshape(-1,1)
    y = salaries['salary_in_usd'].values

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 11)

    reg = LinearRegression()

    reg.fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    print('Results of the test_set:')
    print("Feature (X): {}, \nPredictions (y_pred): {}, \nActual Values (y_true): {}".format(np.round(X_test[:2].flatten(), 2), np.round(y_test_pred[:2], 2), np.round(y_test[:2]), 2))
    print()

    # Compute RMSE
    r_squared = reg.score(X_test, y_test)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Print the metrics
    st.header("Model evaluation results:")
    st.write('R_Squared:', r_squared)
    st.write("RMSE_train: {}".format(np.round(rmse_train,2)))
    st.write("RMSE_test: {}".format(np.round(rmse_test,2)))
    st.write("MAE_test: {}".format(np.round(mae_test, 2)))

    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    # Inladen van de dataset
    # Inladen van de dataset
    df = salaries[salaries['job_title_Data Scientist'] == 1]
    
    st.write("Linear regression tussen  Employee Residence US en Salary in USD") 
    # Selecteren van features en target variabele
    features = ['employee_residence_US']
    target = 'salary_in_usd'
    # Scheiden van train en test set
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    # Opstellen en trainen van het model met feature engineering en hyperparameter tuning
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)

    # Voorspellen van de delivery_time op de test set
    y_pred = model.predict(X_test)

    # Evalueren van het model
    r2 = r2_score(y_test, y_pred)
    st.header('The R2 score')
    st.write("R2 score: ", r2)
    
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.tree import plot_tree

    from sklearn.utils import resample

    from sklearn.utils import resample
    highest = salaries.loc[(salaries['work_year_2022']==1)]
    lowest = salaries.loc[(salaries['work_year_2022']==0)]

    SA_Resample = pd.concat([highest,lowest])
    
    X = SA_Resample.loc[:, SA_Resample.columns != 'work_year_2022']
    y = SA_Resample['work_year_2022']

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=500, stratify=y)

    # Instantiate dt_entropy, set 'entropy' as the information criterion
    dt = DecisionTreeClassifier(criterion='entropy',
                                max_depth=11,
                                min_samples_split=5,
                                min_samples_leaf=5,
                                max_features='sqrt',
                                random_state=1)


    # Fit dt_entropy to the training set
    dt.fit(X_train, y_train)

    # Predict test set labels
    y_pred = dt.predict(X_test)

    # Evaluate acc_test
    acc_test = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred) #geeft aan hoe goed het model is in het voorspellen van de positieve klassen
    recall = recall_score(y_test, y_pred) #geeft aan hoeveel procent van de positieve voorbeelden als positief zijn geklasseerd
    print('Test set accuracy: {:.2f}'.format(acc_test)) 
    print('Test set precision: {:.2f}'.format(precision))
    print('Test set recall: {:.2f}'.format(recall))

    fig = plt.figure(figsize=(10,30))
    plot_tree(dt, filled=True)
    plt.title("Decision tree voor work_year")
    plt.show()
    fig.savefig("decision_tree.png")
    st.pyplot(fig)

    from sklearn.tree import plot_tree
    fig = plt.figure(figsize=(10,30))
    plot_tree(dt, filled=True)
    plt.title("Decision tree voor work_year")
    plt.show()
    fig.savefig("decision_tree.png")
    
    ConfusionMatrixDisplay.from_estimator(
        dt, X_test, y_test, cmap=plt.cm.Blues)
    
    fig = plt.show()
    
    st.pyplot(fig)

    

    
    
    
    
   

#sidebar
salary_var = st.sidebar.number_input( ' salary ' , 4000 , 30400000 )
salary_in_usd_var = st.sidebar.number_input( ' salary_in_usd ' , 2859 , 600000 )
work_year_2020_var = st.sidebar.number_input( ' work_year_2020 ' , 0 , 1 )
work_year_2021_var = st.sidebar.number_input( ' work_year_2021 ' , 0 , 1 )
work_year_2022_var = st.sidebar.number_input( ' work_year_2022 ' , 0 , 1 )
experience_level_EN_var = st.sidebar.number_input( ' experience_level_EN ' , 0 , 1 )
experience_level_EX_var = st.sidebar.number_input( ' experience_level_EX ' , 0 , 1 )
experience_level_MI_var = st.sidebar.number_input( ' experience_level_MI ' , 0 , 1 )
experience_level_SE_var = st.sidebar.number_input( ' experience_level_SE ' , 0 , 1 )
employment_type_CT_var = st.sidebar.number_input( ' employment_type_CT ' , 0 , 1 )
employment_type_FL_var = st.sidebar.number_input( ' employment_type_FL ' , 0 , 1 )
employment_type_FT_var = st.sidebar.number_input( ' employment_type_FT ' , 0 , 1 )
employment_type_PT_var = st.sidebar.number_input( ' employment_type_PT ' , 0 , 1 )
job_title_3D Computer Vision Researcher_var = st.sidebar.number_input( ' job_title_3D Computer Vision Researcher ' , 0 , 1 )
job_title_AI Scientist_var = st.sidebar.number_input( ' job_title_AI Scientist ' , 0 , 1 )
job_title_Analytics Engineer_var = st.sidebar.number_input( ' job_title_Analytics Engineer ' , 0 , 1 )
job_title_Applied Data Scientist_var = st.sidebar.number_input( ' job_title_Applied Data Scientist ' , 0 , 1 )
job_title_Applied Machine Learning Scientist_var = st.sidebar.number_input( ' job_title_Applied Machine Learning Scientist ' , 0 , 1 )
job_title_BI Data Analyst_var = st.sidebar.number_input( ' job_title_BI Data Analyst ' , 0 , 1 )
job_title_Big Data Architect_var = st.sidebar.number_input( ' job_title_Big Data Architect ' , 0 , 1 )
job_title_Big Data Engineer_var = st.sidebar.number_input( ' job_title_Big Data Engineer ' , 0 , 1 )
job_title_Business Data Analyst_var = st.sidebar.number_input( ' job_title_Business Data Analyst ' , 0 , 1 )
job_title_Cloud Data Engineer_var = st.sidebar.number_input( ' job_title_Cloud Data Engineer ' , 0 , 1 )
job_title_Computer Vision Engineer_var = st.sidebar.number_input( ' job_title_Computer Vision Engineer ' , 0 , 1 )
job_title_Computer Vision Software Engineer_var = st.sidebar.number_input( ' job_title_Computer Vision Software Engineer ' , 0 , 1 )
job_title_Data Analyst_var = st.sidebar.number_input( ' job_title_Data Analyst ' , 0 , 1 )
job_title_Data Analytics Engineer_var = st.sidebar.number_input( ' job_title_Data Analytics Engineer ' , 0 , 1 )
job_title_Data Analytics Lead_var = st.sidebar.number_input( ' job_title_Data Analytics Lead ' , 0 , 1 )
job_title_Data Analytics Manager_var = st.sidebar.number_input( ' job_title_Data Analytics Manager ' , 0 , 1 )
job_title_Data Architect_var = st.sidebar.number_input( ' job_title_Data Architect ' , 0 , 1 )
job_title_Data Engineer_var = st.sidebar.number_input( ' job_title_Data Engineer ' , 0 , 1 )
job_title_Data Engineering Manager_var = st.sidebar.number_input( ' job_title_Data Engineering Manager ' , 0 , 1 )
job_title_Data Science Consultant_var = st.sidebar.number_input( ' job_title_Data Science Consultant ' , 0 , 1 )
job_title_Data Science Engineer_var = st.sidebar.number_input( ' job_title_Data Science Engineer ' , 0 , 1 )
job_title_Data Science Manager_var = st.sidebar.number_input( ' job_title_Data Science Manager ' , 0 , 1 )
job_title_Data Scientist_var = st.sidebar.number_input( ' job_title_Data Scientist ' , 0 , 1 )
job_title_Data Specialist_var = st.sidebar.number_input( ' job_title_Data Specialist ' , 0 , 1 )
job_title_Director of Data Engineering_var = st.sidebar.number_input( ' job_title_Director of Data Engineering ' , 0 , 1 )
job_title_Director of Data Science_var = st.sidebar.number_input( ' job_title_Director of Data Science ' , 0 , 1 )
job_title_ETL Developer_var = st.sidebar.number_input( ' job_title_ETL Developer ' , 0 , 1 )
job_title_Finance Data Analyst_var = st.sidebar.number_input( ' job_title_Finance Data Analyst ' , 0 , 1 )
job_title_Financial Data Analyst_var = st.sidebar.number_input( ' job_title_Financial Data Analyst ' , 0 , 1 )
job_title_Head of Data_var = st.sidebar.number_input( ' job_title_Head of Data ' , 0 , 1 )
job_title_Head of Data Science_var = st.sidebar.number_input( ' job_title_Head of Data Science ' , 0 , 1 )
job_title_Head of Machine Learning_var = st.sidebar.number_input( ' job_title_Head of Machine Learning ' , 0 , 1 )
job_title_Lead Data Analyst_var = st.sidebar.number_input( ' job_title_Lead Data Analyst ' , 0 , 1 )
job_title_Lead Data Engineer_var = st.sidebar.number_input( ' job_title_Lead Data Engineer ' , 0 , 1 )
job_title_Lead Data Scientist_var = st.sidebar.number_input( ' job_title_Lead Data Scientist ' , 0 , 1 )
job_title_Lead Machine Learning Engineer_var = st.sidebar.number_input( ' job_title_Lead Machine Learning Engineer ' , 0 , 1 )
job_title_ML Engineer_var = st.sidebar.number_input( ' job_title_ML Engineer ' , 0 , 1 )
job_title_Machine Learning Developer_var = st.sidebar.number_input( ' job_title_Machine Learning Developer ' , 0 , 1 )
job_title_Machine Learning Engineer_var = st.sidebar.number_input( ' job_title_Machine Learning Engineer ' , 0 , 1 )
job_title_Machine Learning Infrastructure Engineer_var = st.sidebar.number_input( ' job_title_Machine Learning Infrastructure Engineer ' , 0 , 1 )
job_title_Machine Learning Manager_var = st.sidebar.number_input( ' job_title_Machine Learning Manager ' , 0 , 1 )
job_title_Machine Learning Scientist_var = st.sidebar.number_input( ' job_title_Machine Learning Scientist ' , 0 , 1 )
job_title_Marketing Data Analyst_var = st.sidebar.number_input( ' job_title_Marketing Data Analyst ' , 0 , 1 )
job_title_NLP Engineer_var = st.sidebar.number_input( ' job_title_NLP Engineer ' , 0 , 1 )
job_title_Principal Data Analyst_var = st.sidebar.number_input( ' job_title_Principal Data Analyst ' , 0 , 1 )
job_title_Principal Data Engineer_var = st.sidebar.number_input( ' job_title_Principal Data Engineer ' , 0 , 1 )
job_title_Principal Data Scientist_var = st.sidebar.number_input( ' job_title_Principal Data Scientist ' , 0 , 1 )
job_title_Product Data Analyst_var = st.sidebar.number_input( ' job_title_Product Data Analyst ' , 0 , 1 )
job_title_Research Scientist_var = st.sidebar.number_input( ' job_title_Research Scientist ' , 0 , 1 )
job_title_Staff Data Scientist_var = st.sidebar.number_input( ' job_title_Staff Data Scientist ' , 0 , 1 )
salary_currency_AUD_var = st.sidebar.number_input( ' salary_currency_AUD ' , 0 , 1 )
salary_currency_BRL_var = st.sidebar.number_input( ' salary_currency_BRL ' , 0 , 1 )
salary_currency_CAD_var = st.sidebar.number_input( ' salary_currency_CAD ' , 0 , 1 )
salary_currency_CHF_var = st.sidebar.number_input( ' salary_currency_CHF ' , 0 , 1 )
salary_currency_CLP_var = st.sidebar.number_input( ' salary_currency_CLP ' , 0 , 1 )
salary_currency_CNY_var = st.sidebar.number_input( ' salary_currency_CNY ' , 0 , 1 )
salary_currency_DKK_var = st.sidebar.number_input( ' salary_currency_DKK ' , 0 , 1 )
salary_currency_EUR_var = st.sidebar.number_input( ' salary_currency_EUR ' , 0 , 1 )
salary_currency_GBP_var = st.sidebar.number_input( ' salary_currency_GBP ' , 0 , 1 )
salary_currency_HUF_var = st.sidebar.number_input( ' salary_currency_HUF ' , 0 , 1 )
salary_currency_INR_var = st.sidebar.number_input( ' salary_currency_INR ' , 0 , 1 )
salary_currency_JPY_var = st.sidebar.number_input( ' salary_currency_JPY ' , 0 , 1 )
salary_currency_MXN_var = st.sidebar.number_input( ' salary_currency_MXN ' , 0 , 1 )
salary_currency_PLN_var = st.sidebar.number_input( ' salary_currency_PLN ' , 0 , 1 )
salary_currency_SGD_var = st.sidebar.number_input( ' salary_currency_SGD ' , 0 , 1 )
salary_currency_TRY_var = st.sidebar.number_input( ' salary_currency_TRY ' , 0 , 1 )
salary_currency_USD_var = st.sidebar.number_input( ' salary_currency_USD ' , 0 , 1 )
employee_residence_AE_var = st.sidebar.number_input( ' employee_residence_AE ' , 0 , 1 )
employee_residence_AR_var = st.sidebar.number_input( ' employee_residence_AR ' , 0 , 1 )
employee_residence_AT_var = st.sidebar.number_input( ' employee_residence_AT ' , 0 , 1 )
employee_residence_AU_var = st.sidebar.number_input( ' employee_residence_AU ' , 0 , 1 )
employee_residence_BE_var = st.sidebar.number_input( ' employee_residence_BE ' , 0 , 1 )
employee_residence_BG_var = st.sidebar.number_input( ' employee_residence_BG ' , 0 , 1 )
employee_residence_BO_var = st.sidebar.number_input( ' employee_residence_BO ' , 0 , 1 )
employee_residence_BR_var = st.sidebar.number_input( ' employee_residence_BR ' , 0 , 1 )
employee_residence_CA_var = st.sidebar.number_input( ' employee_residence_CA ' , 0 , 1 )
employee_residence_CH_var = st.sidebar.number_input( ' employee_residence_CH ' , 0 , 1 )
employee_residence_CL_var = st.sidebar.number_input( ' employee_residence_CL ' , 0 , 1 )
employee_residence_CN_var = st.sidebar.number_input( ' employee_residence_CN ' , 0 , 1 )
employee_residence_CO_var = st.sidebar.number_input( ' employee_residence_CO ' , 0 , 1 )
employee_residence_CZ_var = st.sidebar.number_input( ' employee_residence_CZ ' , 0 , 1 )
employee_residence_DE_var = st.sidebar.number_input( ' employee_residence_DE ' , 0 , 1 )
employee_residence_DK_var = st.sidebar.number_input( ' employee_residence_DK ' , 0 , 1 )
employee_residence_DZ_var = st.sidebar.number_input( ' employee_residence_DZ ' , 0 , 1 )
employee_residence_EE_var = st.sidebar.number_input( ' employee_residence_EE ' , 0 , 1 )
employee_residence_ES_var = st.sidebar.number_input( ' employee_residence_ES ' , 0 , 1 )
employee_residence_FR_var = st.sidebar.number_input( ' employee_residence_FR ' , 0 , 1 )
employee_residence_GB_var = st.sidebar.number_input( ' employee_residence_GB ' , 0 , 1 )
employee_residence_GR_var = st.sidebar.number_input( ' employee_residence_GR ' , 0 , 1 )
employee_residence_HK_var = st.sidebar.number_input( ' employee_residence_HK ' , 0 , 1 )
employee_residence_HN_var = st.sidebar.number_input( ' employee_residence_HN ' , 0 , 1 )
employee_residence_HR_var = st.sidebar.number_input( ' employee_residence_HR ' , 0 , 1 )
employee_residence_HU_var = st.sidebar.number_input( ' employee_residence_HU ' , 0 , 1 )
employee_residence_IE_var = st.sidebar.number_input( ' employee_residence_IE ' , 0 , 1 )
employee_residence_IN_var = st.sidebar.number_input( ' employee_residence_IN ' , 0 , 1 )
employee_residence_IQ_var = st.sidebar.number_input( ' employee_residence_IQ ' , 0 , 1 )
employee_residence_IR_var = st.sidebar.number_input( ' employee_residence_IR ' , 0 , 1 )
employee_residence_IT_var = st.sidebar.number_input( ' employee_residence_IT ' , 0 , 1 )
employee_residence_JE_var = st.sidebar.number_input( ' employee_residence_JE ' , 0 , 1 )
employee_residence_JP_var = st.sidebar.number_input( ' employee_residence_JP ' , 0 , 1 )
employee_residence_KE_var = st.sidebar.number_input( ' employee_residence_KE ' , 0 , 1 )
employee_residence_LU_var = st.sidebar.number_input( ' employee_residence_LU ' , 0 , 1 )
employee_residence_MD_var = st.sidebar.number_input( ' employee_residence_MD ' , 0 , 1 )
employee_residence_MT_var = st.sidebar.number_input( ' employee_residence_MT ' , 0 , 1 )
employee_residence_MX_var = st.sidebar.number_input( ' employee_residence_MX ' , 0 , 1 )
employee_residence_MY_var = st.sidebar.number_input( ' employee_residence_MY ' , 0 , 1 )
employee_residence_NG_var = st.sidebar.number_input( ' employee_residence_NG ' , 0 , 1 )
employee_residence_NL_var = st.sidebar.number_input( ' employee_residence_NL ' , 0 , 1 )
employee_residence_NZ_var = st.sidebar.number_input( ' employee_residence_NZ ' , 0 , 1 )
employee_residence_PH_var = st.sidebar.number_input( ' employee_residence_PH ' , 0 , 1 )
employee_residence_PK_var = st.sidebar.number_input( ' employee_residence_PK ' , 0 , 1 )
employee_residence_PL_var = st.sidebar.number_input( ' employee_residence_PL ' , 0 , 1 )
employee_residence_PR_var = st.sidebar.number_input( ' employee_residence_PR ' , 0 , 1 )
employee_residence_PT_var = st.sidebar.number_input( ' employee_residence_PT ' , 0 , 1 )
employee_residence_RO_var = st.sidebar.number_input( ' employee_residence_RO ' , 0 , 1 )
employee_residence_RS_var = st.sidebar.number_input( ' employee_residence_RS ' , 0 , 1 )
employee_residence_RU_var = st.sidebar.number_input( ' employee_residence_RU ' , 0 , 1 )
employee_residence_SG_var = st.sidebar.number_input( ' employee_residence_SG ' , 0 , 1 )
employee_residence_SI_var = st.sidebar.number_input( ' employee_residence_SI ' , 0 , 1 )
employee_residence_TN_var = st.sidebar.number_input( ' employee_residence_TN ' , 0 , 1 )
employee_residence_TR_var = st.sidebar.number_input( ' employee_residence_TR ' , 0 , 1 )
employee_residence_UA_var = st.sidebar.number_input( ' employee_residence_UA ' , 0 , 1 )
employee_residence_VN_var = st.sidebar.number_input( ' employee_residence_VN ' , 0 , 1 )
remote_ratio_0_var = st.sidebar.number_input( ' remote_ratio_0 ' , 0 , 1 )
remote_ratio_50_var = st.sidebar.number_input( ' remote_ratio_50 ' , 0 , 1 )
remote_ratio_100_var = st.sidebar.number_input( ' remote_ratio_100 ' , 0 , 1 )
company_location_AE_var = st.sidebar.number_input( ' company_location_AE ' , 0 , 1 )
company_location_AS_var = st.sidebar.number_input( ' company_location_AS ' , 0 , 1 )
company_location_AT_var = st.sidebar.number_input( ' company_location_AT ' , 0 , 1 )
company_location_AU_var = st.sidebar.number_input( ' company_location_AU ' , 0 , 1 )
company_location_BE_var = st.sidebar.number_input( ' company_location_BE ' , 0 , 1 )
company_location_BR_var = st.sidebar.number_input( ' company_location_BR ' , 0 , 1 )
company_location_CA_var = st.sidebar.number_input( ' company_location_CA ' , 0 , 1 )
company_location_CH_var = st.sidebar.number_input( ' company_location_CH ' , 0 , 1 )
company_location_CL_var = st.sidebar.number_input( ' company_location_CL ' , 0 , 1 )
company_location_CN_var = st.sidebar.number_input( ' company_location_CN ' , 0 , 1 )
company_location_CO_var = st.sidebar.number_input( ' company_location_CO ' , 0 , 1 )
company_location_CZ_var = st.sidebar.number_input( ' company_location_CZ ' , 0 , 1 )
company_location_DE_var = st.sidebar.number_input( ' company_location_DE ' , 0 , 1 )
company_location_DK_var = st.sidebar.number_input( ' company_location_DK ' , 0 , 1 )
company_location_DZ_var = st.sidebar.number_input( ' company_location_DZ ' , 0 , 1 )
company_location_EE_var = st.sidebar.number_input( ' company_location_EE ' , 0 , 1 )
company_location_ES_var = st.sidebar.number_input( ' company_location_ES ' , 0 , 1 )
company_location_FR_var = st.sidebar.number_input( ' company_location_FR ' , 0 , 1 )
company_location_GB_var = st.sidebar.number_input( ' company_location_GB ' , 0 , 1 )
company_location_GR_var = st.sidebar.number_input( ' company_location_GR ' , 0 , 1 )
company_location_HN_var = st.sidebar.number_input( ' company_location_HN ' , 0 , 1 )
company_location_HR_var = st.sidebar.number_input( ' company_location_HR ' , 0 , 1 )
company_location_HU_var = st.sidebar.number_input( ' company_location_HU ' , 0 , 1 )
company_location_IE_var = st.sidebar.number_input( ' company_location_IE ' , 0 , 1 )
company_location_IL_var = st.sidebar.number_input( ' company_location_IL ' , 0 , 1 )
company_location_IN_var = st.sidebar.number_input( ' company_location_IN ' , 0 , 1 )
company_location_IQ_var = st.sidebar.number_input( ' company_location_IQ ' , 0 , 1 )
company_location_IR_var = st.sidebar.number_input( ' company_location_IR ' , 0 , 1 )
company_location_IT_var = st.sidebar.number_input( ' company_location_IT ' , 0 , 1 )
company_location_JP_var = st.sidebar.number_input( ' company_location_JP ' , 0 , 1 )
company_location_KE_var = st.sidebar.number_input( ' company_location_KE ' , 0 , 1 )
company_location_LU_var = st.sidebar.number_input( ' company_location_LU ' , 0 , 1 )
company_location_MD_var = st.sidebar.number_input( ' company_location_MD ' , 0 , 1 )
company_location_MT_var = st.sidebar.number_input( ' company_location_MT ' , 0 , 1 )
company_location_MX_var = st.sidebar.number_input( ' company_location_MX ' , 0 , 1 )
company_location_MY_var = st.sidebar.number_input( ' company_location_MY ' , 0 , 1 )
company_location_NG_var = st.sidebar.number_input( ' company_location_NG ' , 0 , 1 )
company_location_NL_var = st.sidebar.number_input( ' company_location_NL ' , 0 , 1 )
company_location_NZ_var = st.sidebar.number_input( ' company_location_NZ ' , 0 , 1 )
company_location_PK_var = st.sidebar.number_input( ' company_location_PK ' , 0 , 1 )
company_location_PL_var = st.sidebar.number_input( ' company_location_PL ' , 0 , 1 )
company_location_PT_var = st.sidebar.number_input( ' company_location_PT ' , 0 , 1 )
company_location_RO_var = st.sidebar.number_input( ' company_location_RO ' , 0 , 1 )
company_location_RU_var = st.sidebar.number_input( ' company_location_RU ' , 0 , 1 )
company_location_SG_var = st.sidebar.number_input( ' company_location_SG ' , 0 , 1 )
company_location_SI_var = st.sidebar.number_input( ' company_location_SI ' , 0 , 1 )
company_location_TR_var = st.sidebar.number_input( ' company_location_TR ' , 0 , 1 )
company_location_UA_var = st.sidebar.number_input( ' company_location_UA ' , 0 , 1 )
company_location_US_var = st.sidebar.number_input( ' company_location_US ' , 0 , 1 )
company_location_VN_var = st.sidebar.number_input( ' company_location_VN ' , 0 , 1 )
company_size_L_var = st.sidebar.number_input( ' company_size_L ' , 0 , 1 )
company_size_M_var = st.sidebar.number_input( ' company_size_M ' , 0 , 1 )
company_size_S_var = st.sidebar.number_input( ' company_size_S ' , 0 , 1 )


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

    

    
    
    
    
   

job_title_3d_computer_vision_researcher_var = st.sidebar.number_input( ' job_title_3d_computer_vision_researcher ' , 0 , 1 )
job_title_ai_scientist_var = st.sidebar.number_input( ' job_title_ai_scientist ' , 0 , 1 )
job_title_analytics_engineer_var = st.sidebar.number_input( ' job_title_analytics_engineer ' , 0 , 1 )
job_title_applied_data_scientist_var = st.sidebar.number_input( ' job_title_applied_data_scientist ' , 0 , 1 )
job_title_applied_machine_learning_scientist_var = st.sidebar.number_input( ' job_title_applied_machine_learning_scientist ' , 0 , 1 )
job_title_bi_data_analyst_var = st.sidebar.number_input( ' job_title_bi_data_analyst ' , 0 , 1 )
job_title_big_data_architect_var = st.sidebar.number_input( ' job_title_big_data_architect ' , 0 , 1 )
job_title_big_data_engineer_var = st.sidebar.number_input( ' job_title_big_data_engineer ' , 0 , 1 )
job_title_business_data_analyst_var = st.sidebar.number_input( ' job_title_business_data_analyst ' , 0 , 1 )
job_title_cloud_data_engineer_var = st.sidebar.number_input( ' job_title_cloud_data_engineer ' , 0 , 1 )
job_title_computer_vision_engineer_var = st.sidebar.number_input( ' job_title_computer_vision_engineer ' , 0 , 1 )
job_title_computer_vision_software_engineer_var = st.sidebar.number_input( ' job_title_computer_vision_software_engineer ' , 0 , 1 )
job_title_data_analyst_var = st.sidebar.number_input( ' job_title_data_analyst ' , 0 , 1 )
job_title_data_analytics_engineer_var = st.sidebar.number_input( ' job_title_data_analytics_engineer ' , 0 , 1 )
job_title_data_analytics_lead_var = st.sidebar.number_input( ' job_title_data_analytics_lead ' , 0 , 1 )
job_title_data_analytics_manager_var = st.sidebar.number_input( ' job_title_data_analytics_manager ' , 0 , 1 )
job_title_data_architect_var = st.sidebar.number_input( ' job_title_data_architect ' , 0 , 1 )
job_title_data_engineer_var = st.sidebar.number_input( ' job_title_data_engineer ' , 0 , 1 )
job_title_data_engineering_manager_var = st.sidebar.number_input( ' job_title_data_engineering_manager ' , 0 , 1 )
job_title_data_science_consultant_var = st.sidebar.number_input( ' job_title_data_science_consultant ' , 0 , 1 )
job_title_data_science_engineer_var = st.sidebar.number_input( ' job_title_data_science_engineer ' , 0 , 1 )
job_title_data_science_manager_var = st.sidebar.number_input( ' job_title_data_science_manager ' , 0 , 1 )
job_title_data_scientist_var = st.sidebar.number_input( ' job_title_data_scientist ' , 0 , 1 )
job_title_data_specialist_var = st.sidebar.number_input( ' job_title_data_specialist ' , 0 , 1 )
job_title_director_of_data_engineering_var = st.sidebar.number_input( ' job_title_director_of_data_engineering ' , 0 , 1 )
job_title_director_of_data_science_var = st.sidebar.number_input( ' job_title_director_of_data_science ' , 0 , 1 )
job_title_etl_developer_var = st.sidebar.number_input( ' job_title_etl_developer ' , 0 , 1 )
job_title_finance_data_analyst_var = st.sidebar.number_input( ' job_title_finance_data_analyst ' , 0 , 1 )
job_title_financial_data_analyst_var = st.sidebar.number_input( ' job_title_financial_data_analyst ' , 0 , 1 )
job_title_head_of_data_var = st.sidebar.number_input( ' job_title_head_of_data ' , 0 , 1 )
job_title_head_of_data_science_var = st.sidebar.number_input( ' job_title_head_of_data_science ' , 0 , 1 )
job_title_head_of_machine_learning_var = st.sidebar.number_input( ' job_title_head_of_machine_learning ' , 0 , 1 )
job_title_lead_data_analyst_var = st.sidebar.number_input( ' job_title_lead_data_analyst ' , 0 , 1 )
job_title_lead_data_engineer_var = st.sidebar.number_input( ' job_title_lead_data_engineer ' , 0 , 1 )
job_title_lead_data_scientist_var = st.sidebar.number_input( ' job_title_lead_data_scientist ' , 0 , 1 )
job_title_lead_machine_learning_engineer_var = st.sidebar.number_input( ' job_title_lead_machine_learning_engineer ' , 0 , 1 )
job_title_ml_engineer_var = st.sidebar.number_input( ' job_title_ml_engineer ' , 0 , 1 )
job_title_machine_learning_developer_var = st.sidebar.number_input( ' job_title_machine_learning_developer ' , 0 , 1 )
job_title_machine_learning_engineer_var = st.sidebar.number_input( ' job_title_machine_learning_engineer ' , 0 , 1 )
job_title_machine_learning_infrastructure_engineer_var = st.sidebar.number_input( ' job_title_machine_learning_infrastructure_engineer ' , 0 , 1 )
job_title_machine_learning_manager_var = st.sidebar.number_input( ' job_title_machine_learning_manager ' , 0 , 1 )
job_title_machine_learning_scientist_var = st.sidebar.number_input( ' job_title_machine_learning_scientist ' , 0 , 1 )
job_title_marketing_data_analyst_var = st.sidebar.number_input( ' job_title_marketing_data_analyst ' , 0 , 1 )
job_title_nlp_engineer_var = st.sidebar.number_input( ' job_title_nlp_engineer ' , 0 , 1 )
job_title_principal_data_analyst_var = st.sidebar.number_input( ' job_title_principal_data_analyst ' , 0 , 1 )
job_title_principal_data_engineer_var = st.sidebar.number_input( ' job_title_principal_data_engineer ' , 0 , 1 )
job_title_principal_data_scientist_var = st.sidebar.number_input( ' job_title_principal_data_scientist ' , 0 , 1 )
job_title_product_data_analyst_var = st.sidebar.number_input( ' job_title_product_data_analyst ' , 0 , 1 )
job_title_research_scientist_var = st.sidebar.number_input( ' job_title_research_scientist ' , 0 , 1 )
job_title_staff_data_scientist_var = st.sidebar.number_input( ' job_title_staff_data_scientist ' , 0 , 1 )
salary_currency_aud_var = st.sidebar.number_input( ' salary_currency_aud ' , 0 , 1 )
salary_currency_brl_var = st.sidebar.number_input( ' salary_currency_brl ' , 0 , 1 )
salary_currency_cad_var = st.sidebar.number_input( ' salary_currency_cad ' , 0 , 1 )
salary_currency_chf_var = st.sidebar.number_input( ' salary_currency_chf ' , 0 , 1 )
salary_currency_clp_var = st.sidebar.number_input( ' salary_currency_clp ' , 0 , 1 )
salary_currency_cny_var = st.sidebar.number_input( ' salary_currency_cny ' , 0 , 1 )
salary_currency_dkk_var = st.sidebar.number_input( ' salary_currency_dkk ' , 0 , 1 )
salary_currency_eur_var = st.sidebar.number_input( ' salary_currency_eur ' , 0 , 1 )
salary_currency_gbp_var = st.sidebar.number_input( ' salary_currency_gbp ' , 0 , 1 )
salary_currency_huf_var = st.sidebar.number_input( ' salary_currency_huf ' , 0 , 1 )
salary_currency_inr_var = st.sidebar.number_input( ' salary_currency_inr ' , 0 , 1 )
salary_currency_jpy_var = st.sidebar.number_input( ' salary_currency_jpy ' , 0 , 1 )
salary_currency_mxn_var = st.sidebar.number_input( ' salary_currency_mxn ' , 0 , 1 )
salary_currency_pln_var = st.sidebar.number_input( ' salary_currency_pln ' , 0 , 1 )
salary_currency_sgd_var = st.sidebar.number_input( ' salary_currency_sgd ' , 0 , 1 )
salary_currency_try_var = st.sidebar.number_input( ' salary_currency_try ' , 0 , 1 )
salary_currency_usd_var = st.sidebar.number_input( ' salary_currency_usd ' , 0 , 1 )
employee_residence_ae_var = st.sidebar.number_input( ' employee_residence_ae ' , 0 , 1 )
employee_residence_ar_var = st.sidebar.number_input( ' employee_residence_ar ' , 0 , 1 )
employee_residence_at_var = st.sidebar.number_input( ' employee_residence_at ' , 0 , 1 )
employee_residence_au_var = st.sidebar.number_input( ' employee_residence_au ' , 0 , 1 )
employee_residence_be_var = st.sidebar.number_input( ' employee_residence_be ' , 0 , 1 )
employee_residence_bg_var = st.sidebar.number_input( ' employee_residence_bg ' , 0 , 1 )
employee_residence_bo_var = st.sidebar.number_input( ' employee_residence_bo ' , 0 , 1 )
employee_residence_br_var = st.sidebar.number_input( ' employee_residence_br ' , 0 , 1 )
employee_residence_ca_var = st.sidebar.number_input( ' employee_residence_ca ' , 0 , 1 )
employee_residence_ch_var = st.sidebar.number_input( ' employee_residence_ch ' , 0 , 1 )
employee_residence_cl_var = st.sidebar.number_input( ' employee_residence_cl ' , 0 , 1 )
employee_residence_cn_var = st.sidebar.number_input( ' employee_residence_cn ' , 0 , 1 )
employee_residence_co_var = st.sidebar.number_input( ' employee_residence_co ' , 0 , 1 )
employee_residence_cz_var = st.sidebar.number_input( ' employee_residence_cz ' , 0 , 1 )
employee_residence_de_var = st.sidebar.number_input( ' employee_residence_de ' , 0 , 1 )
employee_residence_dk_var = st.sidebar.number_input( ' employee_residence_dk ' , 0 , 1 )
employee_residence_dz_var = st.sidebar.number_input( ' employee_residence_dz ' , 0 , 1 )
employee_residence_ee_var = st.sidebar.number_input( ' employee_residence_ee ' , 0 , 1 )
employee_residence_es_var = st.sidebar.number_input( ' employee_residence_es ' , 0 , 1 )
employee_residence_fr_var = st.sidebar.number_input( ' employee_residence_fr ' , 0 , 1 )
employee_residence_gb_var = st.sidebar.number_input( ' employee_residence_gb ' , 0 , 1 )
employee_residence_gr_var = st.sidebar.number_input( ' employee_residence_gr ' , 0 , 1 )
employee_residence_hk_var = st.sidebar.number_input( ' employee_residence_hk ' , 0 , 1 )
employee_residence_hn_var = st.sidebar.number_input( ' employee_residence_hn ' , 0 , 1 )
employee_residence_hr_var = st.sidebar.number_input( ' employee_residence_hr ' , 0 , 1 )
employee_residence_hu_var = st.sidebar.number_input( ' employee_residence_hu ' , 0 , 1 )
employee_residence_ie_var = st.sidebar.number_input( ' employee_residence_ie ' , 0 , 1 )
employee_residence_in_var = st.sidebar.number_input( ' employee_residence_in ' , 0 , 1 )
employee_residence_iq_var = st.sidebar.number_input( ' employee_residence_iq ' , 0 , 1 )
employee_residence_ir_var = st.sidebar.number_input( ' employee_residence_ir ' , 0 , 1 )
employee_residence_it_var = st.sidebar.number_input( ' employee_residence_it ' , 0 , 1 )
employee_residence_je_var = st.sidebar.number_input( ' employee_residence_je ' , 0 , 1 )
employee_residence_jp_var = st.sidebar.number_input( ' employee_residence_jp ' , 0 , 1 )
employee_residence_ke_var = st.sidebar.number_input( ' employee_residence_ke ' , 0 , 1 )
employee_residence_lu_var = st.sidebar.number_input( ' employee_residence_lu ' , 0 , 1 )
employee_residence_md_var = st.sidebar.number_input( ' employee_residence_md ' , 0 , 1 )
employee_residence_mt_var = st.sidebar.number_input( ' employee_residence_mt ' , 0 , 1 )
employee_residence_mx_var = st.sidebar.number_input( ' employee_residence_mx ' , 0 , 1 )
employee_residence_my_var = st.sidebar.number_input( ' employee_residence_my ' , 0 , 1 )
employee_residence_ng_var = st.sidebar.number_input( ' employee_residence_ng ' , 0 , 1 )
employee_residence_nl_var = st.sidebar.number_input( ' employee_residence_nl ' , 0 , 1 )
employee_residence_nz_var = st.sidebar.number_input( ' employee_residence_nz ' , 0 , 1 )
employee_residence_ph_var = st.sidebar.number_input( ' employee_residence_ph ' , 0 , 1 )
employee_residence_pk_var = st.sidebar.number_input( ' employee_residence_pk ' , 0 , 1 )
employee_residence_pl_var = st.sidebar.number_input( ' employee_residence_pl ' , 0 , 1 )
employee_residence_pr_var = st.sidebar.number_input( ' employee_residence_pr ' , 0 , 1 )
employee_residence_pt_var = st.sidebar.number_input( ' employee_residence_pt ' , 0 , 1 )
employee_residence_ro_var = st.sidebar.number_input( ' employee_residence_ro ' , 0 , 1 )
employee_residence_rs_var = st.sidebar.number_input( ' employee_residence_rs ' , 0 , 1 )
employee_residence_ru_var = st.sidebar.number_input( ' employee_residence_ru ' , 0 , 1 )
employee_residence_sg_var = st.sidebar.number_input( ' employee_residence_sg ' , 0 , 1 )
employee_residence_si_var = st.sidebar.number_input( ' employee_residence_si ' , 0 , 1 )
employee_residence_tn_var = st.sidebar.number_input( ' employee_residence_tn ' , 0 , 1 )
employee_residence_tr_var = st.sidebar.number_input( ' employee_residence_tr ' , 0 , 1 )
employee_residence_ua_var = st.sidebar.number_input( ' employee_residence_ua ' , 0 , 1 )
employee_residence_us_var = st.sidebar.number_input( ' employee_residence_us ' , 0 , 1 )
employee_residence_vn_var = st.sidebar.number_input( ' employee_residence_vn ' , 0 , 1 )
remote_ratio_0_var = st.sidebar.number_input( ' remote_ratio_0 ' , 0 , 1 )
remote_ratio_50_var = st.sidebar.number_input( ' remote_ratio_50 ' , 0 , 1 )
remote_ratio_100_var = st.sidebar.number_input( ' remote_ratio_100 ' , 0 , 1 )
company_location_ae_var = st.sidebar.number_input( ' company_location_ae ' , 0 , 1 )
company_location_as_var = st.sidebar.number_input( ' company_location_as ' , 0 , 1 )
company_location_at_var = st.sidebar.number_input( ' company_location_at ' , 0 , 1 )
company_location_au_var = st.sidebar.number_input( ' company_location_au ' , 0 , 1 )
company_location_be_var = st.sidebar.number_input( ' company_location_be ' , 0 , 1 )
company_location_br_var = st.sidebar.number_input( ' company_location_br ' , 0 , 1 )
company_location_ca_var = st.sidebar.number_input( ' company_location_ca ' , 0 , 1 )
company_location_ch_var = st.sidebar.number_input( ' company_location_ch ' , 0 , 1 )
company_location_cl_var = st.sidebar.number_input( ' company_location_cl ' , 0 , 1 )
company_location_cn_var = st.sidebar.number_input( ' company_location_cn ' , 0 , 1 )
company_location_co_var = st.sidebar.number_input( ' company_location_co ' , 0 , 1 )
company_location_cz_var = st.sidebar.number_input( ' company_location_cz ' , 0 , 1 )
company_location_de_var = st.sidebar.number_input( ' company_location_de ' , 0 , 1 )
company_location_dk_var = st.sidebar.number_input( ' company_location_dk ' , 0 , 1 )
company_location_dz_var = st.sidebar.number_input( ' company_location_dz ' , 0 , 1 )
company_location_ee_var = st.sidebar.number_input( ' company_location_ee ' , 0 , 1 )
company_location_es_var = st.sidebar.number_input( ' company_location_es ' , 0 , 1 )
company_location_fr_var = st.sidebar.number_input( ' company_location_fr ' , 0 , 1 )
company_location_gb_var = st.sidebar.number_input( ' company_location_gb ' , 0 , 1 )
company_location_gr_var = st.sidebar.number_input( ' company_location_gr ' , 0 , 1 )
company_location_hn_var = st.sidebar.number_input( ' company_location_hn ' , 0 , 1 )
company_location_hr_var = st.sidebar.number_input( ' company_location_hr ' , 0 , 1 )
company_location_hu_var = st.sidebar.number_input( ' company_location_hu ' , 0 , 1 )
company_location_ie_var = st.sidebar.number_input( ' company_location_ie ' , 0 , 1 )
company_location_il_var = st.sidebar.number_input( ' company_location_il ' , 0 , 1 )
company_location_in_var = st.sidebar.number_input( ' company_location_in ' , 0 , 1 )
company_location_iq_var = st.sidebar.number_input( ' company_location_iq ' , 0 , 1 )
company_location_ir_var = st.sidebar.number_input( ' company_location_ir ' , 0 , 1 )
company_location_it_var = st.sidebar.number_input( ' company_location_it ' , 0 , 1 )
company_location_jp_var = st.sidebar.number_input( ' company_location_jp ' , 0 , 1 )
company_location_ke_var = st.sidebar.number_input( ' company_location_ke ' , 0 , 1 )
company_location_lu_var = st.sidebar.number_input( ' company_location_lu ' , 0 , 1 )
company_location_md_var = st.sidebar.number_input( ' company_location_md ' , 0 , 1 )
company_location_mt_var = st.sidebar.number_input( ' company_location_mt ' , 0 , 1 )
company_location_mx_var = st.sidebar.number_input( ' company_location_mx ' , 0 , 1 )
company_location_my_var = st.sidebar.number_input( ' company_location_my ' , 0 , 1 )
company_location_ng_var = st.sidebar.number_input( ' company_location_ng ' , 0 , 1 )
company_location_nl_var = st.sidebar.number_input( ' company_location_nl ' , 0 , 1 )
company_location_nz_var = st.sidebar.number_input( ' company_location_nz ' , 0 , 1 )
company_location_pk_var = st.sidebar.number_input( ' company_location_pk ' , 0 , 1 )
company_location_pl_var = st.sidebar.number_input( ' company_location_pl ' , 0 , 1 )
company_location_pt_var = st.sidebar.number_input( ' company_location_pt ' , 0 , 1 )
company_location_ro_var = st.sidebar.number_input( ' company_location_ro ' , 0 , 1 )
company_location_ru_var = st.sidebar.number_input( ' company_location_ru ' , 0 , 1 )
company_location_sg_var = st.sidebar.number_input( ' company_location_sg ' , 0 , 1 )
company_location_si_var = st.sidebar.number_input( ' company_location_si ' , 0 , 1 )
company_location_tr_var = st.sidebar.number_input( ' company_location_tr ' , 0 , 1 )
company_location_ua_var = st.sidebar.number_input( ' company_location_ua ' , 0 , 1 )
company_location_us_var = st.sidebar.number_input( ' company_location_us ' , 0 , 1 )
company_location_vn_var = st.sidebar.number_input( ' company_location_vn ' , 0 , 1 )
company_size_l_var = st.sidebar.number_input( ' company_size_l ' , 0 , 1 )
company_size_m_var = st.sidebar.number_input( ' company_size_m ' , 0 , 1 )
company_size_s_var = st.sidebar.number_input( ' company_size_s ' , 0 , 1 )


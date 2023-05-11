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

    
    
    from joblib import dump, load

    #Load your trained models and csv file
    model = load(str(str('/Decisiontree.good.joblib'))
    df= salaries

    # Define your target values (y) and your features (df_un) often X
    y='Attrition_Yes';
    df_dt=df.loc[:, df.columns != y]; # exclude the target value
    df_dt= df_dt.loc[:, df_dt.columns != 'work_year_2021']; #exclude the total energy column

    #Proberen
    st.set_page_config(page_title='voorspellen')

    hide_default_format = """
           <style>
           #MainMenu {visibility: hidden; }
           footer {visibility: hidden;}
           </style>
           """
    st.markdown(hide_default_format, unsafe_allow_html=True)


# Create a dropdown box on your main paige
page = st.sidebar.selectbox('Kies de pagina die je nodig hebt',
         ['Home',
            'Classificatie model',
            'Regressie model'])
    
    
    
    column= ['work_year', 'job_title', 'remote_ratio',
           'salary_in_usd'];
    
    columnames_onehot = []
    with open("train_colnames.txt", "r") as f:
          for line in f:
                columnames_onehot.append(str(line.strip()))

    if st.sidebar.button('Predict'):
        dic={}
        for i in range(0,len(column)):
            dic[str(column[i])] = var[i]
        X_unseen = pd.DataFrame.from_dict([dic])
        X_unseen = pd.get_dummies(X_unseen).reindex(columns=columnames_onehot, fill_value=0)
        
        prediction=model.predict(np.array(X_unseen))[0]
        
        pred_prob = model.predict_proba(X_unseen)
        
        if prediction == 0:
            st.success(f"Er is een kans van {pred_prob[0][0] * 100}% dat deze medewerker bij de organisatie blijft ")
        else:
            st.error(
                f"Er is een kans van {pred_prob[0][1] * 100}% dat deze medewerker de organisatie verlaat")

    st.sidebar.text("")

# Regression page
elif page == 'Regressie model':
    st.title('Can we predict the total average energy used through a household based on the average annual electric usage?')
    average_annual_electric_use_mmbtu_var = st.number_input('Insert the average annual electric use in mmbtu ', 0.0, 594.99)

    if st.button('Predict'):
        prediction_lin= reg.predict(np.array(average_annual_electric_use_mmbtu_var).reshape(-1,1))
        st.write('The prediction of average total energy use in mmbtu is between the {} '.format(
            np.round(prediction_lin[0] - 26.76), 2), 'and the {}' .format(np.round(prediction_lin[0] + 26.76), 2))
    
    
    
    
    
    
    
    
    
    
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

    

    
    
    
    
   


remote_ratio_0_var = st.sidebar.number_input( ' remote_ratio_0 ' , 0 , 1 )
remote_ratio_50_var = st.sidebar.number_input( ' remote_ratio_50 ' , 0 , 1 )
remote_ratio_100_var = st.sidebar.number_input( ' remote_ratio_100 ' , 0 , 1 )
company_size_l_var = st.sidebar.number_input( ' company_size_l ' , 0 , 1 )
company_size_m_var = st.sidebar.number_input( ' company_size_m ' , 0 , 1 )
company_size_s_var = st.sidebar.number_input( ' company_size_s ' , 0 , 1 )


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

# Check to see if a file has been uploaded
if upload_file is not None:
    # If it has then do the following:

    # Read the file to a dataframe using pandas
    salaries = pd.read_csv(upload_file)
    salaries = salaries[salaries.salary<300000]
    # Create a section for the dataframe statistics
    st.header('Statistics of Dataframe')
    st.write(salaries.describe())

    # Create a section for the dataframe header
    st.header('Header of Dataframe')
    st.write(salaries.head())
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

    # Selecteren van features en target variabele
    features = ['salary_in_usd']
    target = 'salary'
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
    

    def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

    clf = get_classifier(classifier_name, params)
    
    _train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    #plt.show()
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.title("Data scientists salaries")

upload_file = st.file_uploader('Upload a file containing earthquake data')

# Check to see if a file has been uploaded
if upload_file is not None:
    # If it has then do the following:

    # Read the file to a dataframe using pandas
    salaries = pd.read_csv(upload_file)

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
    print("Model evaluation results:")
    print('R_Squared:', r_squared)
    print("RMSE_train: {}".format(np.round(rmse_train,2)))
    print("RMSE_test: {}".format(np.round(rmse_test,2)))
    print("MAE_test: {}".format(np.round(mae_test, 2)))


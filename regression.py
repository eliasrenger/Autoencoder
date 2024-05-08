# external imports
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# internal imports
from config import *

def custom_regression_function(epoch, a, b, c):
    # Define your function here, for example, a quadratic function
    return a * epoch**2 + b * epoch + c

# load data 
df = pd.read_csv(f'data/analyzed_{MODEL_NAME}_fixed_last.csv')
grouped_data = df.groupby('fixed point')

# regression fit for each fixed point
regression_results = {}
for cluster, data in grouped_data:
    popt, _ = curve_fit(custom_regression_function, data['epoch'], data['x_mean', 'y_mean'])
    regression_results[cluster] = popt

# Print regression results
for cluster, coefficients in regression_results.items():
    print(f"Cluster: {cluster}")
    print("Coefficients:", coefficients)
    print()

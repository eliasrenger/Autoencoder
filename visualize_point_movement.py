import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# internal imports
from config import *

df = pd.read_csv(f'data/analyzed_{MODEL_NAME}_fixed.csv')
grouped_df = df.groupby('fixed point')
marker_dict = {0: 'o', 1: 's', 2: '^', 3: 'X', 4: 'p', 5: 'D', 6: 'v', 7: '>', 8: '<'}

vis_train = True
arr_train = pd.DataFrame(np.load(DATA_PATH)['train_data'])
arr_train.columns = ['x', 'y']
print(arr_train)

# Create a figure and axis object
fig, ax = plt.subplots()
sns.lineplot(data=df, x='x_mean', y='y_mean', sort=False,
            hue='epoch', palette='flare', style='fixed point',
            markers=marker_dict, dashes=True, legend='full',
            ax=ax)
if vis_train:
    sns.scatterplot(data=arr_train, 
                x='x', 
                y='y', 
                color='black', 
                marker='x', 
                s=100, 
                label='training data')

# Set axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Movement of Fixed Points within Epochs')

# Add legend
ax.legend(title='')

# Show the plot
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# internal imports
from config import *

df = pd.read_csv(f'data/analyzed_{MODEL_NAME}_fixed_300epochs.csv')
marker_dict = {0: 'o', 1: 's', 2: '^', 3: 'X', 4: 'p', 5: 'D', 6: 'v', 7: '>', 8: '<'}
print(df[df['epoch'] == 212])

vis_train = True
arr_train = pd.DataFrame(np.load(DATA_PATH)['train_data'])
arr_train.columns = ['x', 'y']

# Create a figure and axis object
fig, ax = plt.subplots()
ax.set_axisbelow(True)
ax.grid(True)

if vis_train:
    sns.scatterplot(data=arr_train, 
                x='x', 
                y='y', 
                color='black', 
                marker='x', 
                s=60, 
                label='training data',
                ax = ax)
sns.scatterplot(data=df, x='x_mean', y='y_mean',
            hue='epoch', palette='flare', style='fixed point',
            s=30, markers=marker_dict, edgecolor='none',
            legend='auto', ax=ax)
last_epoch = pd.DataFrame(df.iloc[-1]).T
print(last_epoch)
#sns.scatterplot(data=last_epoch, x='x_mean', y='y_mean',
#                color='red', marker='x', s=50)

# Set axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Movement of Fixed Points within Epochs')

# Add legend
ax.legend(title='')
handles, labels = ax.get_legend_handles_labels()
labels = labels[:1] + labels[8:]
handles = handles[:1] + handles[8:]
print(handles, labels)
ax.legend(handles=handles, labels=labels, loc='upper left')

# Show the plot
plt.show()
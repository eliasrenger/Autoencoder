import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# internal imports
from config import *

df1 = pd.read_csv(f'data/analyzed_{MODEL_NAME}_fixed.csv')
df1['fixed point'] = 0.001
df2 = pd.read_csv(f'data/analyzed_sin_2d_200_points_sgd_0.01_fixed.csv')
df2['fixed point'] = 0.01
df = pd.concat([df1, df2], ignore_index=True)
df.rename(columns={'fixed point': 'learning rate'}, inplace=True)
grouped_df = df.groupby('learning rate')
marker_dict = {0.001: 'o', 0.01: 's'}

# Create a figure and axis object
fig, ax = plt.subplots()
sns.lineplot(data=df, x='x_mean', y='y_mean', sort=False,
            hue='epoch', palette='flare', style='learning rate',
            markers=marker_dict, dashes=True, legend='full',
            ax=ax)

# Set axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Movement of Fixed Points within Epochs')

# Add legend
ax.legend(title='')

# Show the plot
plt.show()
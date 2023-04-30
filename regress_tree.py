import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import tree

#import the data
df = pd.read_csv('encoded_df.csv')

y = df['total_sales_price']
x = df.drop(columns = 'total_sales_price')

#randomly split the data 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)

#create the decision tree
dt = DecisionTreeRegressor(max_depth = 5)
dt.fit(x_train, y_train)

#predict using the text data
y_pred_dt = dt.predict(x_test)

#create a plot for actual vs predicited values
plt.figure(figsize=(10,5))
plt.title('Regression Decision Tree')
plt.xlabel('True(Price)')
plt.ylabel('Predicted(Price)')

#create a line for perfect prediction
plt.plot(y_test, y_test, color = "black")

#create a line for real prediction 
m, b = np.polyfit(y_test, y_pred_dt, 1)
plt.plot(y_test, m * y_test + b, color = "red")

legend_drawn_flag = True
plt.legend(["Ideal", "Actual"], loc=0, frameon=legend_drawn_flag)
plt.scatter(y_test, y_pred_dt)

#plt.savefig('act_vs_pred', dpi=100)
plt.show()

#show the decision tree
plt.figure(figsize=(325,50))
tree.plot_tree(dt, fontsize = 30, feature_names=list(x.columns), filled = True)
    
#plt.savefig('reg_tree_plot', dpi=100)

#find the average mae and r2 score 
print("Mean Absolute Error:" + str(mean_absolute_error(y_test, y_pred_dt)))
print("R Squared Score:" + str(r2_score(y_test, y_pred_dt)))


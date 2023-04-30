import numpy as np
import pandas as pd
import random as ran

np.set_printoptions(precision=3)
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

df = pd.read_csv('diamonds.csv', index_col=[0])
df.pop('fancy_color_intensity')
df.pop('fancy_color_overtone')
df.pop('fancy_color_secondary_color')
df.pop('fancy_color_dominant_color')
df.pop('fluor_color')
df.pop('fluor_intensity')
df.pop('culet_condition')
df.pop('culet_size')
df.pop('eye_clean')
df.pop('lab')
print(df.columns)
print(df['total_sales_price'])

chosen_idx = np.random.choice(219703, replace=False, size=43000)

def Pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr


print(Pearson_correlation(df['carat_weight'], df['total_sales_price']))
print(Pearson_correlation(df['depth_percent'], df['total_sales_price']))
print(Pearson_correlation(df['table_percent'], df['total_sales_price']))
print(Pearson_correlation(df['meas_length'], df['total_sales_price']))
print(Pearson_correlation(df['meas_width'], df['total_sales_price']))
print(Pearson_correlation(df['meas_depth'], df['total_sales_price']))

print("")
print(Pearson_correlation(df['meas_width'], df['meas_length']))
print(Pearson_correlation(df['meas_depth'], df['meas_length']))

print(Pearson_correlation(df['meas_length'], df['meas_width']))

print(Pearson_correlation(df['meas_depth'], df['meas_width']))

print(Pearson_correlation(df['meas_length'], df['meas_depth']))
print(Pearson_correlation(df['meas_width'], df['meas_depth']))


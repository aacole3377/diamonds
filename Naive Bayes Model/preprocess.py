import pandas as pd
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
df.pop('depth_percent')
df.pop('table_percent')
df.pop("meas_length")
df.pop("meas_depth")
df=df[df['total_sales_price'] <= 500000]
print(df)
df.to_csv('preprocessed.csv')
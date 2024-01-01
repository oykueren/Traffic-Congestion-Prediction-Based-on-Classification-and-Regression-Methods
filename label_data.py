import pandas as pd
import numpy as np
from st_dbscan import ST_DBSCAN
from coordinates import convert_to_utm
import os

def process_csv(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)

    df_reduced = df[["Timestamp","Longitude", "Latitude"]]
    df_copy = df[["Taxi ID","Timestamp", "Speed (km/h)","Longitude", "Latitude", "Time"]]

    df_convert_time = convert_to_utm(df_reduced, src_epsg=4326, dst_epsg=32633,
                            col_lat='Latitude', col_lon='Longitude')

    from sklearn.preprocessing import StandardScaler

    df_without_time = df_convert_time.drop(["Timestamp"], axis=1)
    df_scaled = StandardScaler().fit_transform(df_without_time)

    scaled_data = pd.DataFrame(df_scaled, columns=["Longitude", "Latitude"], index=df_convert_time.index)
    a = df_copy["Timestamp"]
    b = scaled_data

    df_new = pd.concat([a, b], axis=1)
    df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'], errors="coerce")
    df_new['Timestamp'] = df_new['Timestamp'].astype('int64') / 10**9  # Convert nanoseconds to seconds

    st_dbscan_very_dense = ST_DBSCAN(eps1 = 0.01, eps2 = 300, min_samples = 6)
    st_dbscan_very_dense.fit(df_new) 

    st_dbscan_dense = ST_DBSCAN(eps1 = 0.02, eps2 = 300, min_samples = 6)
    st_dbscan_dense.fit(df_new) 

    st_dbscan_moderate = ST_DBSCAN(eps1 = 0.05, eps2 = 300, min_samples = 6) 
    st_dbscan_moderate.fit(df_new) 

    st_dbscan_low = ST_DBSCAN(eps1 = 0.1, eps2 = 300, min_samples = 6)
    st_dbscan_low.fit(df_new) 

    value_very_dense = pd.DataFrame(st_dbscan_very_dense.labels, index=df_reduced.index, columns=["Label1"])
    value_dense = pd.DataFrame(st_dbscan_dense.labels, index=df_reduced.index, columns=["Label2"])
    value_moderate = pd.DataFrame(st_dbscan_moderate.labels, index=df_reduced.index, columns=["Label3"])
    value_low = pd.DataFrame(st_dbscan_low.labels, index=df_reduced.index, columns=["Label4"])

    df = pd.concat([df_copy, value_very_dense, value_dense, value_moderate, value_low], axis=1)

    label_counts = df['Label1'].value_counts()
    mask = (df['Label1'] != -1) & (df['Label1'].map(label_counts) >= 50)
    df.loc[mask, 'Label1'] = 'Very Dense'

    label_counts = df['Label2'].value_counts()
    mask = (df['Label2'] != -1) & (df['Label2'].map(label_counts) >= 50)
    df.loc[mask, 'Label2'] = 'Dense'

    label_counts = df['Label3'].value_counts()
    mask = (df['Label3'] != -1) & (df['Label3'].map(label_counts) >= 50)
    df.loc[mask, 'Label3'] = 'Moderate'

    label_counts = df['Label4'].value_counts()
    mask = (df['Label4'] != -1) & (df['Label4'].map(label_counts) >= 50)
    df.loc[mask, 'Label4'] = 'Low Traffic'

    conditions = [
        df.apply(lambda row: 'Very Dense' in row.values, axis=1),
        df.apply(lambda row: 'Dense' in row.values, axis=1),
        df.apply(lambda row: 'Moderate' in row.values, axis=1),
        df.apply(lambda row: 'Low Traffic' in row.values, axis=1)
    ]

    # Define the label for each condition
    labels = ['Very Dense', 'Dense', 'Moderate', 'Low Traffic']

    # Use np.select to apply these conditions and labels to the DataFrame
    df['Traffic_Label'] = np.select(conditions, labels, default='No Traffic')

    df.drop(["Label1", "Label2", "Label3", "Label4"], axis=1, inplace=True)

    df.loc[df["Traffic_Label"] == "No Traffic", "Traffic_Label"] = 0
    df.loc[df["Traffic_Label"] == "Low Traffic", "Traffic_Label"] = 1
    df.loc[df["Traffic_Label"] == "Moderate", "Traffic_Label"] = 2
    df.loc[df["Traffic_Label"] == "Dense", "Traffic_Label"] = 3
    df.loc[df["Traffic_Label"] == "Very Dense", "Traffic_Label"] = 4
        
    # Save the processed DataFrame to a new CSV file
    return df.to_csv(output_file_path, index=False)

import os
import pandas as pd

# Your process_csv function here ...
os.chdir(r'C:\Users\ysrmhmt\Desktop\Lectures\Data Mining\Project')

data_preparation_dir = 'Data Preperation'
clean_data_dir = os.path.join(data_preparation_dir, 'CleanedData')
labelled_data_dir = os.path.join(data_preparation_dir, 'Labelled Data')

# Create Labelled Data directory if it doesn't exist
if not os.path.exists(labelled_data_dir):
    os.makedirs(labelled_data_dir)

# Process each CSV file in CleanedData folder
for file in os.listdir(clean_data_dir):
    if file.endswith('.csv'):
        input_file_path = os.path.join(clean_data_dir, file)
        output_file_path = os.path.join(labelled_data_dir, file)
        process_csv(input_file_path, output_file_path)
        print(f'Processed and saved labelled data for {file}')
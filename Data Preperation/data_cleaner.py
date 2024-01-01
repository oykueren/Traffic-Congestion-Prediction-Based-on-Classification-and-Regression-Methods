import pandas as pd
import numpy as np
import os

def process_csv(input_file_path, output_file_path):
    column_names = ["Taxi ID", "Timestamp", "Speed (km/h)", "Distance (m)", "Linestring"]
    df = pd.read_csv(input_file_path, header=None, names=column_names)

    #remove unnecessary string
    df["Linestring"] = df["Linestring"].str.replace("(", "")
    df["Linestring"] = df["Linestring"].str.replace(")", "")
    df["Linestring"] = df["Linestring"].str.replace("LINESTRING" , "")

    #split coordinates and create array
    df["Linestring"] = df["Linestring"].str.split(r',|\s+')

    #rename the column
    df = df.rename(columns={"Linestring": "Coordinates"})

    #Create new array to work with rows which has more than 4 coordinates. It will make easier and less computational
    indices_of_repetition = []
    for i in df.index:
        # df["Coordinates"][i] = [float(element) for element in df["Coordinates"][i]]
        if len(df["Coordinates"][i]) > 4:
            indices_of_repetition.append(i)
                
    new_df = df.loc[indices_of_repetition]

    #drop the rows from df
    df.drop(indices_of_repetition, inplace=True)

    #reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)


    #Divide the coordinates into 4 and create new rows
    new_rows = []

    # Iterate over rows in the new_df DataFrame
    for _, row in new_df.iterrows():
        len_row = len(row["Coordinates"])
        n = 0
        
        # Iterate over pairs of coordinates with a step of 2
        while n <= len_row - 4:
            x1, x2, x3, x4 = row["Coordinates"][n], row["Coordinates"][n + 1], row["Coordinates"][n + 2], row["Coordinates"][n + 3]

            
            # Create a new row with the extracted coordinates
            new_row = {
                "Taxi ID": row["Taxi ID"],
                "Timestamp": row["Timestamp"],
                "Speed (km/h)": row["Speed (km/h)"],
                "Distance (m)": row["Distance (m)"],
                "Coordinates": [x1, x2, x3, x4]
            }
            
            # Append the new row to the list
            new_rows.append(new_row)
            
            n += 2



    # Append the new rows to the df DataFrame
    cor4_df = pd.DataFrame(new_rows)
    # Concatenate df and new_df
    df = pd.concat([df, cor4_df], ignore_index=True)

    df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['Coordinates'].to_list(), index=df.index)
    df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].astype(float)

    df["Longitude"] = df["x1"]
    df["Latitude"] = df["y1"]
    df = df.drop(['x1', 'y1', 'x2', 'y2', 'Coordinates', "Distance (m)"], axis=1)
    
    #Coordinate based filtering
    df = df[(df["Longitude"] >= 28.82545) & (df["Longitude"] <= 29.12508) & 
                    (df["Latitude"] <= 40.29373) & (df["Latitude"] >= 40.16594)]
    
    #Speed based filtering.
    df = df[df['Speed (km/h)'] <= 150]
    

    # Convert Timestamps, invalid parsing will result in NaN
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Round down the timestamps to the nearest minute
    df['Timestamp'] = df['Timestamp'].dt.floor('T')

    # Function to get the middle row of each group
    def get_middle_row(group):
        middle_index = group.index[len(group) // 2]
        return group.loc[middle_index]

    # Group by Taxi ID and Timestamp, and apply the function to get the middle row
    df = df.groupby(["Taxi ID", "Timestamp"]).apply(get_middle_row)
    df = df.rename_axis(['Taxi_ID_Index', 'Timestamp_Index'])
    df = df.reset_index()
    df = df.drop(['Taxi_ID_Index', 'Timestamp_Index'], axis=1)

    # Extract minute and hour, and create a new 'Time' column
    df['Minute'] = df['Timestamp'].dt.minute
    df['Hour'] = df['Timestamp'].dt.hour
    df_time = df["Hour"] * 60 + df["Minute"]
    df_time = pd.DataFrame(df_time, columns=["Time"])

    # Drop the original minute and hour columns
    df.drop(["Minute", "Hour"], axis=1, inplace=True)

    # Concatenate the new 'Time' column
    df = pd.concat([df, df_time], axis=1)

    # Drop rows where Timestamp is NaN
    df.dropna(subset=['Timestamp'], inplace=True)

    # Save to CSV
    df.to_csv(output_file_path, index=False)


    df.to_csv(output_file_path, index=False)
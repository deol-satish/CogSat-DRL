import pandas as pd
import numpy as np
import os
import sys
import logging
from utils.settings import channelFreqs, default_dataset_path

logging.basicConfig(level=logging.DEBUG)


def get_df_processed(dataset_path=default_dataset_path):

    df = pd.read_csv(dataset_path, header=0)
    df['Time'] = pd.to_datetime(df['Time'])

    # Convert Time to UTC timestamp
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC').astype('int64')

    # Checking the frequency columns in the dataframe match preset frequencies

    for column in df.columns:
        if "Freq_Hz" in column:
            logging.debug(f"Column name: {column}, Type: {df[column].dtype}, Unique values: {df[column].nunique()}, Sample values: {df[column].unique()[:2]}")
            logging.debug(f"Unique values: {sorted(list(df[column].unique()))}")
            df_freq_set = sorted(list(df[column].unique()))


    logging.debug(set(map(float, channelFreqs)) == set(map(float, df_freq_set)))

    # Create the mapping dictionary: frequency -> mapped number (1 to 10)
    freq_to_channel = {freq: idx + 1 for idx, freq in enumerate(channelFreqs)}

    # Add special case for 0
    freq_to_channel[0] = 0


    # Predefined list of strings to check in the column names
    obs_reegex_alls = ["RSSI", "SNR", "Lat","Lon"]

    # Initialize an empty list to store matching column names
    obs_space_columns = ['Time']

    # Loop through the columns in the dataframe
    for column in df.columns:
        if "Freq_Hz" in column:
            df[column] = df[column].map(freq_to_channel)
        # logging.debug information about the column
        logging.debug(f"Column name: {column}, Type: {df[column].dtype}, Unique values: {df[column].nunique()}, Sample values: {df[column].unique()[:2]}")

        # Check if any string in the predefined list is present in the column name
        if any(substring.lower() in column.lower() for substring in obs_reegex_alls):
            obs_space_columns.append(column)

    # Remove specific columns from the list
    obs_space_columns.remove('GEO1_Lat')
    obs_space_columns.remove('GEO1_Lon')

    # Show the resulting list of columns
    logging.debug("Columns containing any of the specified strings:")
    logging.debug(obs_space_columns)


    # Create a new dataframe with the selected columns
    df_obs_space = df[obs_space_columns]

    # Show the new dataframe (optional)
    logging.debug("New DataFrame with selected columns:")
    logging.debug(df_obs_space.head())

    return df_obs_space

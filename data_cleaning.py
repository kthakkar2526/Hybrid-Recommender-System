import pandas as pd

FILE_PATH = "data/Music Info.csv"

def process_data(df):
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on the 'track_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with the string 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    return (
        df
        .drop_duplicates(subset="track_id")
        .drop(columns=["genre", "spotify_id"])
        .fillna({"tags": "no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower()
        )
        .reset_index(drop=True)
    )

def prepare_data_for_filtering(df):
    """
    Cleans the input DataFrame by dropping specific columns.

    This function takes a DataFrame and removes the columns "track_id", "name",
    and "spotify_preview_url". It is intended to prepare the data for content based
    filtering by removing unnecessary features.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing songs information.

    Returns:
    pandas.DataFrame: A DataFrame with the specified columns removed.
    """
    return (
        df
        .drop(columns=["track_id", "name", "spotify_preview_url"])
    )

def main(file_path):
    """
    Main function to load, clean, and save data.
    Parameters:
    file_path (str): The file path to the raw data CSV file.
    Returns:
    None
    """
    # load the data
    raw_data = pd.read_csv(file_path)
    
    # perform data cleaning
    cleaned_data = process_data(raw_data)
    
    # save cleaned data
    cleaned_data.to_csv("data/cleaned_data.csv", index=False)

if __name__ == "__main__":
    main(FILE_PATH)
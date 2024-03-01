import pandas as pd
import os

# Assume you have a lot of dataframes that only differ in the fact that they each have a different last column with a user input score
# Then, merge all dataframes into a single dataframe with the mean of the user input scores as the last column for each row "score_mean"

def merge_scores(dataframes):
    # Merge all dataframes into a single dataframe
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Calculate the mean of the user input scores
    merged_df['score_mean'] = merged_df.iloc[:, -len(dataframes):].mean(axis=1)

    return merged_df

def load_dataframes(rootfolder):
    dataframes = []
    for dirpath, dirnames, filenames in os.walk(rootfolder):
        for filename in [f for f in filenames if f.endswith("_scored.csv")]:
            df = pd.read_csv(os.path.join(dirpath, filename))
            dataframes.append(df)
    return dataframes

if __name__ == "__main__":
    rootfolder = "data/applications"
    dataframes = load_dataframes(rootfolder)
    merged_df = merge_scores(dataframes)
    outpath = rootfolder + "/applications_merged.csv"
    merged_df.to_csv(outpath, index=False)
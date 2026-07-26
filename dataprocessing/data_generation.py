from utilsforecast.plotting import plot_series
import pandas as pd
import glob

if __name__ == "__main__":

    # Get a list of all CSV files in a directory
    csv_files = glob.glob("../data/*.csv")

    # Create an empty dataframe to store the combined data
    df = pd.DataFrame()

    # Loop through each CSV file and append its contents to the combined dataframe
    for csv_file in csv_files:
        t_df = pd.read_csv(csv_file, index_col=0)
        df = pd.concat([df, t_df])

    # rename column
    df.rename(columns={'time': 'ds'}, inplace=True)

    # save full data
    df.to_csv("../data/prediction_data.csv")

    # visualization
    df['ds'] = pd.to_datetime(df['ds'])
    figure = plot_series(df, max_ids=12)
    figure.savefig("../data/data_brgm.png")

# Spotify Playlist Analysis

This project analyzes Spotify playlist data using R and Python. It covers various tasks including data fetching, preprocessing, visualization, and the application of machine learning models.

# Machine Learning Spotify
## Aplicaciones de Machine Learning con metadatos de Spotify

Algunos modelos aplicados a playlists de Spotify.

## Project Structure

- `CodigoSpotify_VF.R`: R script for data analysis and modeling.
- `Spotify_ML.R`: R script for fetching data, preprocessing, and machine learning.
- `CreateFeatures2.ipynb`: Jupyter Notebook for fetching audio features from Spotify.
- `data/`: Directory containing datasets.

## CodigoSpotify_VF.R

This R script is used for:
- Data manipulation and cleaning.
- Generating visualizations (like boxplots and density plots) of audio features.
- Building and evaluating classification models, specifically decision trees and random forests.

**Libraries Used:**
- `readr`
- `dplyr`
- `stringr`
- `missForest`
- `ggplot2`
- `rpart`
- `rpart.plot`

## Spotify_ML.R

This R script is used for:
- Fetching playlist data from the Spotify API using the `spotifyr` package.
- Preprocessing data, including normalization and handling categorical variables.
- Performing feature engineering, such as creating time-based features from track release and addition dates.
- Building and evaluating machine learning models:
    - Logistic Regression
    - Random Forest (including techniques for handling imbalanced data like oversampling, undersampling, ROSE, and SMOTE)
    - Gradient Boosting
- Visualizing data, such as correlation plots and ROC curves.

**Libraries Used:**
- `spotifyr`
- `tidyverse`
- `corrplot`
- `caret`
- `MASS`
- `pROC`
- `ROSE`
- `DMwR`
- `purrr`
- `viridis`
- `lubridate`

## CreateFeatures2.ipynb

This Jupyter Notebook is used for:
- Fetching track IDs from a specified Spotify playlist.
- Retrieving audio features for these tracks in batches using the Spotipy library.
- Concatenating the features into a Pandas DataFrame.
- Exporting the resulting DataFrame to a CSV file (e.g., `Gustan.csv`).

**Libraries Used:**
- `pandas`
- `spotipy`

## Data

The `data/` directory stores the datasets used and generated by the project. This includes files such as:
- `Spotify_ML_data.csv`
- `Semana.csv`
- `ComparadoCompleto_RF.csv` (referenced in `CodigoSpotify_VF.R`)
- `Gustan.csv` (generated by `CreateFeatures2.ipynb`)

For more detailed information about the datasets, please refer to the `data/Readme.md` file.

## How to Run

### Prerequisites
- R and RStudio (for .R scripts)
- Python and Jupyter Notebook (for .ipynb notebooks)
- Spotify API credentials (client ID and secret) are required for `Spotify_ML.R` and `CreateFeatures2.ipynb` to fetch data. These need to be set up within the scripts/notebooks themselves.

### Running the R Scripts (`CodigoSpotify_VF.R`, `Spotify_ML.R`)
1. Ensure all required R libraries listed in their respective sections are installed. You can install them using `install.packages("library_name")` in the R console.
2. Open the R script in RStudio.
3. Modify file paths for data loading/saving as needed (e.g., the `read.csv` path in `CodigoSpotify_VF.R` and `df.to_csv` in `CreateFeatures2.ipynb`).
4. For `Spotify_ML.R`, you'll need to set your `playlist_username` and `playlist_uris` variables, and your Spotify API credentials.
5. Run the script line by line or source the entire file.

### Running the Jupyter Notebook (`CreateFeatures2.ipynb`)
1. Ensure all required Python libraries listed in its section are installed. You can install them using `pip install library_name` in your terminal/command prompt.
2. Set your Spotify API credentials (cid and secret) within the notebook.
3. Update the playlist details (username and playlist ID) that you want to fetch.
4. Run the cells in the notebook sequentially.
5. Note the output CSV file path (e.g., `C:/Users/julio.hernandezl/Documents/Cesar/Kesar/Machine Learning/Curso Random Forest/Version2/Gustan.csv`) and modify if necessary.

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some YourFeatureName'`).
5. Push to the branch (`git push origin feature/YourFeatureName`).
6. Open a Pull Request.

## License

The license for this project is currently not specified. Please consider adding an open-source license if you intend to share this project widely. Common choices include the MIT License, Apache License 2.0, or GNU General Public License v3.0.

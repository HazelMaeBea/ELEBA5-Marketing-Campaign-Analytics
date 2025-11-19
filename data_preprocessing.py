"""
Data Preprocessing Module for iFood CRM Campaign Analysis
Handles data loading, feature engineering, and cleaning operations.
"""

import pandas as pd
import numpy as np
from datetime import date
import os
import kagglehub


class DataPreprocessor:
    """
    Preprocesses raw iFood customer data for analysis and modeling.
    """

    def __init__(self, filepath='ml_project1_data.csv', use_kaggle=False, kaggle_dataset='rodsaldanha/arketing-campaign'):
        """
        Initialize the preprocessor with the raw data file.

        Parameters:
        -----------
        filepath : str
            Path to the raw CSV data file (used if use_kaggle=False)
        use_kaggle : bool
            Whether to download data from Kaggle
        kaggle_dataset : str
            Kaggle dataset identifier (format: 'owner/dataset-name')
        """
        self.filepath = filepath
        self.use_kaggle = use_kaggle
        self.kaggle_dataset = kaggle_dataset
        self.kaggle_path = None
        self.raw_data = None
        self.processed_data = None

    def download_from_kaggle(self):
        """
        Download dataset from Kaggle using kagglehub.

        Returns:
        --------
        str
            Path to the downloaded dataset directory
        """
        print(f"Downloading dataset from Kaggle: {self.kaggle_dataset}")
        self.kaggle_path = kagglehub.dataset_download(self.kaggle_dataset)
        print(f"Dataset downloaded to: {self.kaggle_path}")
        return self.kaggle_path

    def find_csv_in_kaggle_path(self):
        """
        Find the CSV file in the Kaggle download directory.

        Returns:
        --------
        str
            Path to the CSV file
        """
        if self.kaggle_path is None:
            raise ValueError(
                "Kaggle data not downloaded. Call download_from_kaggle() first.")

        # Look for CSV files in the downloaded directory
        csv_files = [f for f in os.listdir(
            self.kaggle_path) if f.endswith('.csv')]

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.kaggle_path}")

        # Use the first CSV file found (typically there's only one)
        csv_file = csv_files[0]
        full_path = os.path.join(self.kaggle_path, csv_file)
        print(f"Found CSV file: {csv_file}")

        return full_path

    def load_data(self):
        """Load the raw dataset from CSV or Kaggle."""
        if self.use_kaggle:
            self.download_from_kaggle()
            data_path = self.find_csv_in_kaggle_path()
        else:
            data_path = self.filepath

        self.raw_data = pd.read_csv(data_path)
        print(f"Loaded {len(self.raw_data)} records from {data_path}")
        return self.raw_data

    def engineer_features(self):
        """
        Create derived features from raw data:
        - Age from Year_Birth
        - Customer_Days from Dt_Customer
        - Dummy variables for Marital_Status and Education
        - Total spending amounts
        - Campaign acceptance totals
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.raw_data.copy()

        # Create Customer Age Column
        df['Age'] = date.today().year - df['Year_Birth']

        # Number of days as a customer
        df['Customer_Days'] = (pd.to_datetime(
            "now") - pd.to_datetime(df['Dt_Customer'])) // np.timedelta64(1, 'D')

        # Clean and consolidate Marital_Status
        df.loc[df['Marital_Status'].isin(
            ['Alone', 'Absurd', 'YOLO']), 'Marital_Status'] = 'Single'
        dummy_marital = pd.get_dummies(df['Marital_Status'], prefix='marital')
        df = pd.concat([df, dummy_marital], axis=1)

        # Create Education dummy variables
        dummy_education = pd.get_dummies(df['Education'], prefix='education')
        df = pd.concat([df, dummy_education], axis=1)

        # Drop unused columns
        df.drop(columns=['ID', 'Marital_Status', 'Education',
                'Year_Birth', 'Dt_Customer'], inplace=True)

        # Calculate total spending
        df['MntTotal'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                             'MntFishProducts', 'MntSweetProducts']].sum(axis=1)

        # Regular vs Gold Products
        df['MntRegularProds'] = df['MntTotal'] - df['MntGoldProds']

        # Total campaigns accepted
        df['AcceptedCmpOverall'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                                       'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)

        self.processed_data = df
        print(f"Feature engineering complete. New shape: {df.shape}")
        return df

    def remove_missing_values(self):
        """Remove rows with missing values."""
        if self.processed_data is None:
            raise ValueError(
                "Data not processed. Call engineer_features() first.")

        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data.dropna()
        removed_count = initial_count - len(self.processed_data)

        print(f"Removed {removed_count} rows with missing values")
        return self.processed_data

    def remove_outliers(self):
        """
        Remove outliers from Income and Age using IQR method.
        """
        if self.processed_data is None:
            raise ValueError(
                "Data not processed. Call engineer_features() first.")

        df = self.processed_data
        initial_count = len(df)

        # Remove Income outliers
        Q1_income = df['Income'].quantile(0.25)
        Q3_income = df['Income'].quantile(0.75)
        IQR_income = Q3_income - Q1_income
        df = df[df['Income'] < Q3_income + 1.5 * IQR_income]
        income_removed = initial_count - len(df)

        # Remove Age outliers
        Q1_age = df['Age'].quantile(0.25)
        Q3_age = df['Age'].quantile(0.75)
        IQR_age = Q3_age - Q1_age
        df = df[df['Age'] < Q3_age + 1.5 * IQR_age]
        age_removed = len(self.processed_data) - income_removed - len(df)

        print(
            f"Removed {income_removed} Income outliers and {age_removed} Age outliers")

        self.processed_data = df
        return df

    def get_processed_data(self):
        """Return the fully processed dataset."""
        if self.processed_data is None:
            raise ValueError(
                "No processed data available. Run preprocessing pipeline first.")
        return self.processed_data

    def save_processed_data(self, output_path='ifood_df.csv'):
        """
        Save the processed dataset to CSV.

        Parameters:
        -----------
        output_path : str
            Path where the processed data will be saved
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save.")

        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    def run_pipeline(self, output_path='ifood_df.csv'):
        """
        Execute the complete preprocessing pipeline:
        1. Load data
        2. Engineer features
        3. Remove missing values
        4. Remove outliers
        5. Save processed data

        Parameters:
        -----------
        output_path : str
            Path where the processed data will be saved

        Returns:
        --------
        pd.DataFrame
            The fully processed dataset
        """
        print("=" * 50)
        print("Starting Data Preprocessing Pipeline")
        print("=" * 50)

        self.load_data()
        self.engineer_features()
        self.remove_missing_values()
        self.remove_outliers()
        self.save_processed_data(output_path)

        print("=" * 50)
        print("Preprocessing Complete!")
        print("=" * 50)

        return self.processed_data


if __name__ == "__main__":
    # Example usage with Kaggle download
    print("Option 1: Using Kaggle dataset")
    preprocessor = DataPreprocessor(
        use_kaggle=True, kaggle_dataset='rodsaldanha/arketing-campaign')
    processed_data = preprocessor.run_pipeline('ifood_df.csv')
    print(f"\nFinal dataset shape: {processed_data.shape}")
    print(f"Columns: {list(processed_data.columns)}")

    # Alternative: Using local file
    # preprocessor = DataPreprocessor('ml_project1_data.csv', use_kaggle=False)
    # processed_data = preprocessor.run_pipeline('ifood_df.csv')

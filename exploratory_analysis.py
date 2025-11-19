"""
Exploratory Data Analysis Module for iFood CRM Campaign
Provides visualization and correlation analysis capabilities.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class ExploratoryAnalysis:
    """
    Performs exploratory data analysis on customer data.
    """

    def __init__(self, dataframe):
        """
        Initialize with a pandas DataFrame.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The preprocessed customer data
        """
        self.df = dataframe

    def plot_categorical_distribution(self, figsize=(15, 4)):
        """
        Visualize distribution of categorical variables (Marital Status and Education).
        Note: This requires the original raw data with categorical columns.
        """
        fig, ax = plt.subplots(1, 3, figsize=figsize)

        # Check if we have the original categorical columns
        if 'Marital_Status' in self.df.columns and 'Education' in self.df.columns:
            sns.countplot(data=self.df, x='Marital_Status', ax=ax[0])
            ax[0].set_title('Marital Status Distribution')
            ax[0].tick_params(axis='x', rotation=45)

            sns.countplot(data=self.df, x='Education', ax=ax[1])
            ax[1].set_title('Education Distribution')
            ax[1].tick_params(axis='x', rotation=45)

            df_plot = self.df.groupby(['Marital_Status', 'Education']).size().reset_index().pivot(
                columns='Marital_Status', index='Education', values=0)
            df_plot.apply(lambda x: x/x.sum(), axis=1).plot(
                kind='bar', stacked=True, ax=ax[2], colormap='Paired')
            ax[2].set_title('Education by Marital Status')
            ax[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        else:
            print("Warning: Original categorical columns not found in dataset.")

        plt.tight_layout()
        return fig

    def plot_numerical_distributions(self, figsize=(18, 8)):
        """
        Create histograms for all key numerical features.
        """
        numerical_cols = ['Age', 'Customer_Days', 'Income', 'Kidhome', 'Teenhome',
                          'MntTotal', 'MntRegularProds', 'MntGoldProds', 'AcceptedCmpOverall']

        # Filter to only existing columns
        existing_cols = [
            col for col in numerical_cols if col in self.df.columns]

        fig = self.df[existing_cols].hist(
            figsize=figsize, bins=30, edgecolor='black')
        plt.suptitle('Distribution of Numerical Features', y=1.02, fontsize=16)
        plt.tight_layout()
        return fig

    def plot_outlier_detection(self, figsize=(10, 3)):
        """
        Create box plots for Income and Age to visualize outliers.
        """
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        if 'Income' in self.df.columns:
            self.df.boxplot(column=['Income'], ax=ax[0])
            ax[0].set_title('Income Distribution')

        if 'Age' in self.df.columns:
            self.df.boxplot(column=['Age'], ax=ax[1])
            ax[1].set_title('Age Distribution')

        plt.tight_layout()
        return fig

    def calculate_correlation_matrix(self, method='spearman'):
        """
        Calculate correlation matrix for all numerical features.

        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman', or 'kendall')

        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        # Drop non-analytical columns if present
        cols_to_drop = ['Z_CostContact', 'Z_Revenue']
        df_corr = self.df.drop(
            columns=[col for col in cols_to_drop if col in self.df.columns])

        corr_matrix = df_corr.corr(method=method)
        return corr_matrix

    def plot_correlation_heatmap(self, method='spearman', figsize=(10, 10)):
        """
        Visualize correlation matrix as a heatmap.

        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman', or 'kendall')
        figsize : tuple
            Figure size
        """
        corr = self.calculate_correlation_matrix(method=method)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Create color palette
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title(
            f'Correlation Heatmap ({method.capitalize()} Method)', fontsize=14)
        plt.tight_layout()
        return plt.gcf()

    def get_high_correlations(self, threshold=0.8, method='spearman'):
        """
        Extract pairs of features with high correlation.

        Parameters:
        -----------
        threshold : float
            Minimum absolute correlation value to report
        method : str
            Correlation method

        Returns:
        --------
        pd.DataFrame
            Dataframe of highly correlated feature pairs
        """
        corr = self.calculate_correlation_matrix(method=method)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.mask(mask)

        # Unstack and filter
        corr_unstacked = corr_masked.unstack().sort_values(ascending=False)
        high_corr = corr_unstacked[(abs(corr_unstacked) > threshold) & (
            abs(corr_unstacked) < 1)]

        return pd.DataFrame(high_corr, columns=['Correlation'])

    def get_negative_correlations(self, threshold=-0.5, method='spearman'):
        """
        Extract pairs of features with strong negative correlation.

        Parameters:
        -----------
        threshold : float
            Maximum correlation value (should be negative)
        method : str
            Correlation method

        Returns:
        --------
        pd.DataFrame
            Dataframe of negatively correlated feature pairs
        """
        corr = self.calculate_correlation_matrix(method=method)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.mask(mask)

        # Unstack and filter
        corr_unstacked = corr_masked.unstack().sort_values(ascending=True)
        neg_corr = corr_unstacked[corr_unstacked < threshold]

        return pd.DataFrame(neg_corr, columns=['Correlation'])

    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics for the dataset.

        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        return self.df.describe()

    def analyze_response_distribution(self):
        """
        Analyze the distribution of the Response variable (target).

        Returns:
        --------
        dict
            Dictionary containing response statistics
        """
        if 'Response' not in self.df.columns:
            return {"error": "Response column not found in dataset"}

        response_counts = self.df['Response'].value_counts()
        response_pct = self.df['Response'].value_counts(normalize=True) * 100

        return {
            'counts': response_counts.to_dict(),
            'percentages': response_pct.to_dict(),
            'total': len(self.df)
        }

    def plot_response_by_feature(self, feature, figsize=(10, 5)):
        """
        Visualize Response distribution by a specific feature.

        Parameters:
        -----------
        feature : str
            Column name to analyze against Response
        figsize : tuple
            Figure size
        """
        if 'Response' not in self.df.columns:
            print("Error: Response column not found in dataset")
            return None

        if feature not in self.df.columns:
            print(f"Error: {feature} column not found in dataset")
            return None

        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Count plot
        sns.countplot(data=self.df, x=feature, hue='Response', ax=ax[0])
        ax[0].set_title(f'Response Distribution by {feature}')
        ax[0].tick_params(axis='x', rotation=45)

        # Normalized plot
        response_by_feature = pd.crosstab(
            self.df[feature], self.df['Response'], normalize='index')
        response_by_feature.plot(kind='bar', stacked=True, ax=ax[1])
        ax[1].set_title(f'Response Rate by {feature}')
        ax[1].set_ylabel('Proportion')
        ax[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('ifood_df.csv')

    analyzer = ExploratoryAnalysis(df)

    print("Summary Statistics:")
    print(analyzer.generate_summary_statistics())

    print("\nHigh Correlations (>0.8):")
    print(analyzer.get_high_correlations(threshold=0.8))

    print("\nNegative Correlations (<-0.5):")
    print(analyzer.get_negative_correlations(threshold=-0.5))

    if 'Response' in df.columns:
        print("\nResponse Distribution:")
        print(analyzer.analyze_response_distribution())

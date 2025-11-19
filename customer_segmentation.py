"""
Customer Segmentation Module for iFood CRM Campaign
Implements K-Means clustering and hierarchical clustering for customer segmentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans


class CustomerSegmentation:
    """
    Performs customer segmentation using clustering algorithms.
    """

    def __init__(self, dataframe):
        """
        Initialize with a pandas DataFrame.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The preprocessed customer data
        """
        self.df = dataframe.copy()
        self.clustering_data = None
        self.kmeans_model = None
        self.n_clusters = None

    def prepare_clustering_data(self):
        """
        Prepare data for clustering by removing non-feature columns.

        Returns:
        --------
        pd.DataFrame
            Data ready for clustering
        """
        # Drop columns that shouldn't be used for clustering
        cols_to_drop = ['Z_CostContact', 'Z_Revenue', 'Response']
        self.clustering_data = self.df.drop(
            columns=[col for col in cols_to_drop if col in self.df.columns]
        )

        print(
            f"Clustering data prepared with {self.clustering_data.shape[1]} features")
        return self.clustering_data

    def plot_dendrogram(self, method='ward', truncate_mode='lastp', figsize=(12, 6)):
        """
        Create hierarchical clustering dendrogram.

        Parameters:
        -----------
        method : str
            Linkage method ('ward', 'average', 'complete', etc.)
        truncate_mode : str
            Dendrogram truncation mode
        figsize : tuple
            Figure size
        """
        if self.clustering_data is None:
            self.prepare_clustering_data()

        linkage_matrix = linkage(self.clustering_data, method=method)

        plt.figure(figsize=figsize)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Distance')

        dendrogram(
            linkage_matrix,
            truncate_mode=truncate_mode,
            show_leaf_counts=False,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True
        )

        plt.tight_layout()
        return plt.gcf()

    def plot_elbow_curve(self, max_k=10, figsize=(10, 6)):
        """
        Create elbow plot to determine optimal number of clusters.

        Parameters:
        -----------
        max_k : int
            Maximum number of clusters to test
        figsize : tuple
            Figure size

        Returns:
        --------
        list
            Inertia scores for each k value
        """
        if self.clustering_data is None:
            self.prepare_clustering_data()

        inertia_scores = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                max_iter=300,
                n_init=10,
                random_state=0
            )
            kmeans.fit(self.clustering_data)
            inertia_scores.append(kmeans.inertia_)

        # Plot elbow curve
        plt.figure(figsize=figsize)
        plt.plot(range(1, max_k + 1), inertia_scores,
                 marker='o', linestyle='-')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return inertia_scores

    def fit_kmeans(self, n_clusters=3, random_state=0):
        """
        Fit K-Means clustering model.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        np.ndarray
            Cluster labels for each customer
        """
        if self.clustering_data is None:
            self.prepare_clustering_data()

        self.n_clusters = n_clusters
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=random_state
        )

        # Fit and predict clusters
        cluster_labels = self.kmeans_model.fit_predict(self.clustering_data)

        # Add cluster labels to original dataframe
        self.df['cluster'] = cluster_labels
        self.clustering_data['cluster'] = cluster_labels

        print(f"K-Means clustering complete with {n_clusters} clusters")
        print(
            f"Cluster distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

        return cluster_labels

    def get_cluster_profiles(self):
        """
        Generate statistical profiles for each cluster.

        Returns:
        --------
        pd.DataFrame
            Mean values for each feature by cluster
        """
        if 'cluster' not in self.df.columns:
            raise ValueError(
                "Clusters not yet created. Run fit_kmeans() first.")

        cluster_profiles = self.df.groupby('cluster').mean()
        return cluster_profiles

    def plot_cluster_boxplots(self, features=None, figsize=(30, 20)):
        """
        Create box plots for each feature by cluster.

        Parameters:
        -----------
        features : list
            List of features to plot (if None, plots key features)
        figsize : tuple
            Figure size
        """
        if 'cluster' not in self.clustering_data.columns:
            raise ValueError(
                "Clusters not yet created. Run fit_kmeans() first.")

        if features is None:
            features = ['Income', 'Age', 'Recency', 'MntWines', 'MntFruits',
                        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                        'MntRegularProds', 'MntGoldProds', 'NumDealsPurchases',
                        'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                        'NumWebVisitsMonth', 'MntTotal']

        # Filter to existing features
        features = [f for f in features if f in self.clustering_data.columns]

        # Calculate grid dimensions
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
        axes = axes.flatten()

        # Create box plots
        for idx, feature in enumerate(features):
            self.clustering_data.boxplot(
                column=[feature], by='cluster', ax=axes[idx])
            axes[idx].set_title(feature)
            axes[idx].set_xlabel('Cluster')

        # Hide extra subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Feature Distribution by Cluster', fontsize=16, y=1.0)
        plt.tight_layout()
        return fig

    def plot_cluster_countplots(self, categorical_features=None, figsize=(25, 5)):
        """
        Create count plots for categorical features by cluster.

        Parameters:
        -----------
        categorical_features : list
            List of categorical features to plot
        figsize : tuple
            Figure size
        """
        if 'cluster' not in self.clustering_data.columns:
            raise ValueError(
                "Clusters not yet created. Run fit_kmeans() first.")

        if categorical_features is None:
            categorical_features = ['AcceptedCmp5',
                                    'education_Basic', 'Kidhome', 'Teenhome']

        # Filter to existing features
        categorical_features = [
            f for f in categorical_features if f in self.clustering_data.columns]

        n_features = len(categorical_features)
        fig, axes = plt.subplots(1, n_features, figsize=figsize)

        if n_features == 1:
            axes = [axes]

        for idx, feature in enumerate(categorical_features):
            sns.countplot(data=self.clustering_data,
                          x='cluster', hue=feature, ax=axes[idx])
            axes[idx].set_title(f'{feature} by Cluster')
            axes[idx].legend(title=feature)

        plt.tight_layout()
        return fig

    def describe_clusters(self):
        """
        Generate a comprehensive description of each cluster.

        Returns:
        --------
        dict
            Dictionary with cluster descriptions and key characteristics
        """
        if 'cluster' not in self.df.columns:
            raise ValueError(
                "Clusters not yet created. Run fit_kmeans() first.")

        cluster_descriptions = {}

        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster_id]

            description = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.df) * 100,
                'avg_income': cluster_data['Income'].mean() if 'Income' in cluster_data.columns else None,
                'avg_age': cluster_data['Age'].mean() if 'Age' in cluster_data.columns else None,
                'avg_total_spending': cluster_data['MntTotal'].mean() if 'MntTotal' in cluster_data.columns else None,
                'avg_campaigns_accepted': cluster_data['AcceptedCmpOverall'].mean() if 'AcceptedCmpOverall' in cluster_data.columns else None,
                'response_rate': cluster_data['Response'].mean() * 100 if 'Response' in cluster_data.columns else None
            }

            cluster_descriptions[f'Cluster_{cluster_id}'] = description

        return cluster_descriptions

    def export_clustered_data(self, output_path='ifood_df_clustered.csv'):
        """
        Export the dataset with cluster labels.

        Parameters:
        -----------
        output_path : str
            Path where the clustered data will be saved
        """
        if 'cluster' not in self.df.columns:
            raise ValueError(
                "Clusters not yet created. Run fit_kmeans() first.")

        self.df.to_csv(output_path, index=False)
        print(f"Clustered data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('ifood_df.csv')

    segmentation = CustomerSegmentation(df)

    # Prepare data
    segmentation.prepare_clustering_data()

    # Visualize optimal k
    print("Generating elbow plot...")
    segmentation.plot_elbow_curve(max_k=10)
    plt.show()

    # Fit K-Means with k=3
    print("\nFitting K-Means with k=3...")
    segmentation.fit_kmeans(n_clusters=3)

    # Get cluster descriptions
    print("\nCluster Descriptions:")
    descriptions = segmentation.describe_clusters()
    for cluster_name, stats in descriptions.items():
        print(f"\n{cluster_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Export results
    segmentation.export_clustered_data('ifood_df_clustered.csv')

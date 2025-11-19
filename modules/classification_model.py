"""
Classification Model Module for iFood CRM Campaign
Implements Random Forest classifier with hyperparameter tuning and profit optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)


class CampaignClassifier:
    """
    Builds and evaluates a classification model for campaign response prediction.
    Includes profit optimization functionality.
    """

    def __init__(self, dataframe, target_column='Response'):
        """
        Initialize the classifier with preprocessed data.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The preprocessed customer data
        target_column : str
            Name of the target variable column
        """
        self.df = dataframe.copy()
        self.target_column = target_column
        self.features = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_model = None
        self.predictions = None
        self.prediction_probabilities = None

        # Campaign economics (from case study)
        self.cost_per_contact = 3.0  # 6,720 MU / 2,240 customers
        self.pilot_revenue = 3674.0
        self.pilot_contacts = 2240
        self.pilot_conversions = int(2240 * 0.15)  # 15% success rate
        self.revenue_per_conversion = self.pilot_revenue / self.pilot_conversions

        print(f"Campaign Economics:")
        print(f"  Cost per contact: {self.cost_per_contact:.2f} MU")
        print(f"  Revenue per conversion: {self.revenue_per_conversion:.2f} MU")

    def prepare_data(self, test_size=0.40, random_state=5):
        """
        Split data into features and labels, then train/test sets.

        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataframe")

        # Split into features and labels
        self.features = self.df.drop(self.target_column, axis=1)
        self.labels = self.df[self.target_column]

        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=random_state
        )

        print(f"\nData split complete:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        print(f"  Features: {self.features.shape[1]}")
        print(
            f"  Class distribution (train): {dict(self.y_train.value_counts())}")
        print(
            f"  Class distribution (test): {dict(self.y_test.value_counts())}")

    def train_random_forest(self, use_grid_search=True, param_grid=None, cv=5, verbose=1):
        """
        Train Random Forest classifier with optional hyperparameter tuning.

        Parameters:
        -----------
        use_grid_search : bool
            Whether to use GridSearchCV for hyperparameter tuning
        param_grid : dict
            Parameter grid for GridSearchCV
        cv : int
            Number of cross-validation folds
        verbose : int
            Verbosity level for GridSearchCV

        Returns:
        --------
        object
            Trained model
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_features': ['sqrt'],
                'max_depth': [None, 3, 5, 8],
                'criterion': ['gini'],
                'min_samples_split': [2, 3, 4]
            }

        print("\nTraining Random Forest classifier...")

        if use_grid_search:
            print(f"Using GridSearchCV with {cv}-fold cross-validation")
            self.model = GridSearchCV(
                RandomForestClassifier(random_state=5),
                param_grid=param_grid,
                cv=cv,
                verbose=verbose,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
            self.best_model = self.model.best_estimator_

            print(f"\nBest parameters found:")
            for param, value in self.model.best_params_.items():
                print(f"  {param}: {value}")
            print(f"Best cross-validation score: {self.model.best_score_:.4f}")
        else:
            self.model = RandomForestClassifier(random_state=5)
            self.model.fit(self.X_train, self.y_train)
            self.best_model = self.model

        return self.model

    def predict(self, return_probabilities=True):
        """
        Generate predictions on the test set.

        Parameters:
        -----------
        return_probabilities : bool
            Whether to also return prediction probabilities

        Returns:
        --------
        np.ndarray
            Predictions (and probabilities if requested)
        """
        if self.best_model is None:
            raise ValueError(
                "Model not trained. Call train_random_forest() first.")

        self.predictions = self.best_model.predict(self.X_test)

        if return_probabilities:
            self.prediction_probabilities = self.best_model.predict_proba(self.X_test)[
                :, 1]

        print("\nPredictions generated on test set")
        return self.predictions

    def evaluate_model(self):
        """
        Evaluate model performance with multiple metrics.

        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        if self.predictions is None:
            self.predict()

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(
            self.y_test, self.predictions, zero_division=0)
        recall = recall_score(self.y_test, self.predictions, zero_division=0)
        f1 = f1_score(self.y_test, self.predictions, zero_division=0)

        if self.prediction_probabilities is not None:
            roc_auc = roc_auc_score(self.y_test, self.prediction_probabilities)
        else:
            roc_auc = None

        print("\n" + "=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC:   {roc_auc:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

    def plot_confusion_matrix(self, figsize=(8, 6)):
        """
        Plot confusion matrix.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        if self.predictions is None:
            self.predict()

        cm = confusion_matrix(self.y_test, self.predictions)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()

        return plt.gcf()

    def plot_roc_curve(self, figsize=(8, 6)):
        """
        Plot ROC curve.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        if self.prediction_probabilities is None:
            self.predict(return_probabilities=True)

        fpr, tpr, thresholds = roc_curve(
            self.y_test, self.prediction_probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2,
                 linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def get_feature_importance(self, top_n=10):
        """
        Get feature importance from the trained model.

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if self.best_model is None:
            raise ValueError(
                "Model not trained. Call train_random_forest() first.")

        feature_importance = pd.DataFrame({
            'features': self.features.columns,
            'importance': self.best_model.feature_importances_ * 100
        })

        feature_importance = feature_importance.sort_values(
            'importance', ascending=False)

        if top_n:
            return feature_importance.head(top_n)
        return feature_importance

    def plot_feature_importance(self, top_n=10, figsize=(10, 6)):
        """
        Plot feature importance.

        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        figsize : tuple
            Figure size
        """
        importance_df = self.get_feature_importance(top_n=top_n)

        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance',
                    y='features', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance (%)')
        plt.ylabel('Features')
        plt.tight_layout()

        return plt.gcf()

    def calculate_profit(self, predictions=None, probabilities=None, threshold=0.5):
        """
        Calculate expected profit for given predictions.

        Parameters:
        -----------
        predictions : np.ndarray
            Binary predictions (if None, uses threshold on probabilities)
        probabilities : np.ndarray
            Prediction probabilities
        threshold : float
            Probability threshold for conversion

        Returns:
        --------
        dict
            Profit analysis results
        """
        if predictions is None and probabilities is None:
            if self.predictions is None:
                self.predict()
            predictions = self.predictions
            probabilities = self.prediction_probabilities

        # If threshold is provided and probabilities exist, recalculate predictions
        if probabilities is not None and threshold != 0.5:
            predictions = (probabilities >= threshold).astype(int)

        # Calculate confusion matrix components
        cm = confusion_matrix(self.y_test, predictions)
        true_negatives = cm[0, 0]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        true_positives = cm[1, 1]

        # Calculate profit
        total_contacted = true_positives + false_positives
        total_conversions = true_positives

        revenue = total_conversions * self.revenue_per_conversion
        cost = total_contacted * self.cost_per_contact
        profit = revenue - cost

        # Calculate ROI
        roi = (profit / cost * 100) if cost > 0 else 0

        return {
            'total_contacted': total_contacted,
            'total_conversions': total_conversions,
            'conversion_rate': (total_conversions / total_contacted * 100) if total_contacted > 0 else 0,
            'revenue': revenue,
            'cost': cost,
            'profit': profit,
            'roi': roi,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives
        }

    def optimize_threshold(self, thresholds=None):
        """
        Find the optimal probability threshold to maximize profit.

        Parameters:
        -----------
        thresholds : list
            List of thresholds to test (if None, uses 0.1 to 0.9 in 0.05 steps)

        Returns:
        --------
        dict
            Optimization results with best threshold and profit
        """
        if self.prediction_probabilities is None:
            self.predict(return_probabilities=True)

        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        results = []

        for threshold in thresholds:
            profit_analysis = self.calculate_profit(
                probabilities=self.prediction_probabilities,
                threshold=threshold
            )
            profit_analysis['threshold'] = threshold
            results.append(profit_analysis)

        results_df = pd.DataFrame(results)

        # Find best threshold
        best_idx = results_df['profit'].idxmax()
        best_result = results_df.loc[best_idx]

        print("\n" + "=" * 50)
        print("PROFIT OPTIMIZATION RESULTS")
        print("=" * 50)
        print(f"Optimal Threshold: {best_result['threshold']:.2f}")
        print(f"Expected Profit: {best_result['profit']:.2f} MU")
        print(f"ROI: {best_result['roi']:.1f}%")
        print(f"Customers to Contact: {int(best_result['total_contacted'])}")
        print(f"Expected Conversions: {int(best_result['total_conversions'])}")
        print(f"Conversion Rate: {best_result['conversion_rate']:.1f}%")

        return {
            'best_threshold': best_result['threshold'],
            'best_profit': best_result['profit'],
            'best_roi': best_result['roi'],
            'all_results': results_df,
            'best_result': best_result
        }

    def plot_profit_curve(self, optimization_results=None, figsize=(12, 5)):
        """
        Plot profit vs threshold curve.

        Parameters:
        -----------
        optimization_results : dict
            Results from optimize_threshold()
        figsize : tuple
            Figure size
        """
        if optimization_results is None:
            optimization_results = self.optimize_threshold()

        results_df = optimization_results['all_results']

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Profit vs Threshold
        axes[0].plot(results_df['threshold'],
                     results_df['profit'], marker='o', linewidth=2)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].axvline(x=optimization_results['best_threshold'], color='green',
                        linestyle='--', alpha=0.5, label=f"Best: {optimization_results['best_threshold']:.2f}")
        axes[0].set_xlabel('Probability Threshold')
        axes[0].set_ylabel('Expected Profit (MU)')
        axes[0].set_title('Profit Optimization Curve')
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Plot 2: Contacts vs Conversions
        axes[1].plot(results_df['threshold'], results_df['total_contacted'],
                     marker='o', label='Contacts', linewidth=2)
        axes[1].plot(results_df['threshold'], results_df['total_conversions'],
                     marker='s', label='Conversions', linewidth=2)
        axes[1].axvline(x=optimization_results['best_threshold'], color='green',
                        linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Probability Threshold')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Contacts & Conversions by Threshold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        return fig

    def generate_classification_report(self):
        """
        Generate detailed classification report.

        Returns:
        --------
        str
            Classification report
        """
        if self.predictions is None:
            self.predict()

        report = classification_report(self.y_test, self.predictions)
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(report)

        return report


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('ifood_df.csv')

    classifier = CampaignClassifier(df, target_column='Response')

    # Prepare data
    classifier.prepare_data(test_size=0.40, random_state=5)

    # Train model
    classifier.train_random_forest(use_grid_search=True, cv=5, verbose=1)

    # Evaluate
    metrics = classifier.evaluate_model()

    # Feature importance
    print("\n\nTop 10 Most Important Features:")
    print(classifier.get_feature_importance(top_n=10))

    # Optimize for profit
    optimization_results = classifier.optimize_threshold()

    # Generate visualizations
    classifier.plot_confusion_matrix()
    plt.savefig('output_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    classifier.plot_roc_curve()
    plt.savefig('output_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    classifier.plot_feature_importance(top_n=10)
    plt.savefig('output_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    classifier.plot_profit_curve(optimization_results)
    plt.savefig('output_profit_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n\nAll visualizations saved!")

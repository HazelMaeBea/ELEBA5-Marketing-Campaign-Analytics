"""
Main Pipeline Script for iFood CRM Campaign Analysis
Orchestrates the complete workflow: preprocessing, EDA, segmentation, and classification.
"""

import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from exploratory_analysis import ExploratoryAnalysis
from customer_segmentation import CustomerSegmentation
from classification_model import CampaignClassifier


def run_full_pipeline(
    raw_data_path='ml_project1_data.csv',
    processed_data_path='ifood_df.csv',
    clustered_data_path='ifood_df_clustered.csv',
    n_clusters=3,
    show_plots=True,
    use_kaggle=True,
    kaggle_dataset='rodsaldanha/arketing-campaign',
    run_classification=True,
    test_size=0.40,
    optimize_profit=True
):
    """
    Execute the complete analysis pipeline for iFood CRM data.

    Parameters:
    -----------
    raw_data_path : str
        Path to the raw data CSV file (used if use_kaggle=False)
    processed_data_path : str
        Path where processed data will be saved
    clustered_data_path : str
        Path where clustered data will be saved
    n_clusters : int
        Number of customer segments to create
    show_plots : bool
        Whether to display plots during execution
    use_kaggle : bool
        Whether to download data from Kaggle
    kaggle_dataset : str
        Kaggle dataset identifier (format: 'owner/dataset-name')
    run_classification : bool
        Whether to run the classification model
    test_size : float
        Proportion of data for testing (classification)
    optimize_profit : bool
        Whether to run profit optimization

    Returns:
    --------
    dict
        Dictionary containing all results and models
    """

    print("=" * 70)
    print("iFood CRM Campaign Analysis - Full Pipeline")
    print("=" * 70)

    # ==========================================
    # STEP 1: Data Preprocessing
    # ==========================================
    print("\n[STEP 1/3] DATA PREPROCESSING")
    print("-" * 70)

    if use_kaggle:
        print(f"Using Kaggle dataset: {kaggle_dataset}")
        preprocessor = DataPreprocessor(
            use_kaggle=True, kaggle_dataset=kaggle_dataset)
    else:
        print(f"Using local file: {raw_data_path}")
        preprocessor = DataPreprocessor(raw_data_path, use_kaggle=False)

    processed_data = preprocessor.run_pipeline(processed_data_path)

    # ==========================================
    # STEP 2: Exploratory Data Analysis
    # ==========================================
    print("\n[STEP 2/3] EXPLORATORY DATA ANALYSIS")
    print("-" * 70)

    analyzer = ExploratoryAnalysis(processed_data)

    # Summary statistics
    print("\nSummary Statistics:")
    summary = analyzer.generate_summary_statistics()
    print(summary)

    # Correlation analysis
    print("\n\nHigh Correlations (|r| > 0.8):")
    high_corr = analyzer.get_high_correlations(threshold=0.8)
    print(high_corr)

    print("\n\nNegative Correlations (r < -0.5):")
    neg_corr = analyzer.get_negative_correlations(threshold=-0.5)
    print(neg_corr)

    # Response distribution
    if 'Response' in processed_data.columns:
        print("\n\nResponse Variable Distribution:")
        response_stats = analyzer.analyze_response_distribution()
        print(f"  Total Customers: {response_stats['total']}")
        print(f"  Response Counts: {response_stats['counts']}")
        print(f"  Response Percentages: {response_stats['percentages']}")

    # Generate visualizations
    if show_plots:
        print("\nGenerating visualizations...")

        # Numerical distributions
        analyzer.plot_numerical_distributions()
        plt.savefig('output_numerical_distributions.png',
                    dpi=150, bbox_inches='tight')
        print("  Saved: output_numerical_distributions.png")

        # Correlation heatmap
        analyzer.plot_correlation_heatmap()
        plt.savefig('output_correlation_heatmap.png',
                    dpi=150, bbox_inches='tight')
        print("  Saved: output_correlation_heatmap.png")

        plt.close('all')  # Close all figures to free memory

    # ==========================================
    # STEP 3: Customer Segmentation
    # ==========================================
    print("\n[STEP 3/3] CUSTOMER SEGMENTATION")
    print("-" * 70)

    segmentation = CustomerSegmentation(processed_data)
    segmentation.prepare_clustering_data()

    # Generate elbow plot
    print("\nCalculating optimal number of clusters...")
    inertia_scores = segmentation.plot_elbow_curve(max_k=10)
    if show_plots:
        plt.savefig('output_elbow_curve.png', dpi=150, bbox_inches='tight')
        print("  Saved: output_elbow_curve.png")
        plt.close()

    # Generate dendrogram
    print("\nGenerating hierarchical clustering dendrogram...")
    segmentation.plot_dendrogram()
    if show_plots:
        plt.savefig('output_dendrogram.png', dpi=150, bbox_inches='tight')
        print("  Saved: output_dendrogram.png")
        plt.close()

    # Fit K-Means
    print(f"\nFitting K-Means with {n_clusters} clusters...")
    cluster_labels = segmentation.fit_kmeans(n_clusters=n_clusters)

    # Get cluster profiles
    print("\nCluster Profiles:")
    cluster_profiles = segmentation.get_cluster_profiles()
    print(cluster_profiles[['Income', 'Age',
          'MntTotal', 'AcceptedCmpOverall']].round(2))

    # Detailed cluster descriptions
    print("\n\nDetailed Cluster Descriptions:")
    cluster_descriptions = segmentation.describe_clusters()
    for cluster_name, stats in cluster_descriptions.items():
        print(f"\n{cluster_name}:")
        print(
            f"  Size: {stats['size']} customers ({stats['percentage']:.1f}%)")
        if stats['avg_income']:
            print(f"  Avg Income: ${stats['avg_income']:.2f}")
        if stats['avg_age']:
            print(f"  Avg Age: {stats['avg_age']:.1f} years")
        if stats['avg_total_spending']:
            print(f"  Avg Total Spending: ${stats['avg_total_spending']:.2f}")
        if stats['avg_campaigns_accepted']:
            print(
                f"  Avg Campaigns Accepted: {stats['avg_campaigns_accepted']:.2f}")
        if stats['response_rate'] is not None:
            print(f"  Response Rate: {stats['response_rate']:.1f}%")

    # Generate cluster visualizations
    if show_plots:
        print("\nGenerating cluster visualizations...")

        # Box plots
        segmentation.plot_cluster_boxplots()
        plt.savefig('output_cluster_boxplots.png',
                    dpi=150, bbox_inches='tight')
        print("  Saved: output_cluster_boxplots.png")
        plt.close()

        # Count plots
        segmentation.plot_cluster_countplots()
        plt.savefig('output_cluster_countplots.png',
                    dpi=150, bbox_inches='tight')
        print("  Saved: output_cluster_countplots.png")
        plt.close()

    # Export clustered data
    segmentation.export_clustered_data(clustered_data_path)

    # ==========================================
    # STEP 4: Classification Model
    # ==========================================
    classifier = None
    model_metrics = None
    optimization_results = None

    if run_classification:
        print("\n[STEP 4/4] CLASSIFICATION MODEL")
        print("-" * 70)

        classifier = CampaignClassifier(
            processed_data, target_column='Response')

        # Prepare data
        classifier.prepare_data(test_size=test_size, random_state=5)

        # Train model
        print("\nTraining Random Forest with hyperparameter tuning...")
        classifier.train_random_forest(use_grid_search=True, cv=5, verbose=1)

        # Evaluate model
        model_metrics = classifier.evaluate_model()

        # Feature importance
        print("\n\nTop 10 Most Important Features:")
        feature_importance = classifier.get_feature_importance(top_n=10)
        print(feature_importance)

        # Profit optimization
        if optimize_profit:
            print("\n\nOptimizing for maximum profit...")
            optimization_results = classifier.optimize_threshold()

        # Generate visualizations
        if show_plots:
            print("\nGenerating classification visualizations...")

            # Confusion matrix
            classifier.plot_confusion_matrix()
            plt.savefig('output_confusion_matrix.png',
                        dpi=150, bbox_inches='tight')
            print("  ✓ Saved: output_confusion_matrix.png")
            plt.close()

            # ROC curve
            classifier.plot_roc_curve()
            plt.savefig('output_roc_curve.png', dpi=150, bbox_inches='tight')
            print("  Saved: output_roc_curve.png")
            plt.close()

            # Feature importance
            classifier.plot_feature_importance(top_n=10)
            plt.savefig('output_feature_importance.png',
                        dpi=150, bbox_inches='tight')
            print("  ✓ Saved: output_feature_importance.png")
            plt.close()

            # Profit curve
            if optimize_profit and optimization_results:
                classifier.plot_profit_curve(optimization_results)
                plt.savefig('output_profit_curve.png',
                            dpi=150, bbox_inches='tight')
                print("  ✓ Saved: output_profit_curve.png")
                plt.close()

            # Classification report
            classifier.generate_classification_report()

    # ==========================================
    # PIPELINE COMPLETE
    # ==========================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput Files:")
    print(f"  • {processed_data_path} - Preprocessed data")
    print(f"  • {clustered_data_path} - Data with cluster labels")
    if show_plots:
        print(f"  • output_*.png - Visualization plots")

    # Return all results
    return {
        'preprocessor': preprocessor,
        'processed_data': processed_data,
        'analyzer': analyzer,
        'segmentation': segmentation,
        'cluster_labels': cluster_labels,
        'cluster_profiles': cluster_profiles,
        'cluster_descriptions': cluster_descriptions,
        'high_correlations': high_corr,
        'negative_correlations': neg_corr,
        'classifier': classifier,
        'model_metrics': model_metrics,
        'feature_importance': feature_importance if run_classification else None,
        'optimization_results': optimization_results
    }


if __name__ == "__main__":
    # Run the complete pipeline with Kaggle dataset and classification
    results = run_full_pipeline(
        processed_data_path='ifood_df.csv',
        clustered_data_path='ifood_df_clustered.csv',
        n_clusters=3,
        show_plots=True,
        use_kaggle=True,
        kaggle_dataset='rodsaldanha/arketing-campaign',
        run_classification=True,
        test_size=0.40,
        optimize_profit=True
    )

    print("\n\nPipeline execution complete. Results stored in 'results' dictionary.")

    # Print key results
    if results['optimization_results']:
        print("=" * 70)
        print("KEY BUSINESS INSIGHTS")
        print("=" * 70)
        opt_res = results['optimization_results']
        print("\nPROFIT MAXIMIZATION:")
        print(
            f"  Optimal Probability Threshold: {opt_res['best_threshold']:.2f}")
        print(f"  Expected Profit: ${opt_res['best_profit']:.2f}")
        print(f"  Return on Investment: {opt_res['best_roi']:.1f}%")
        print(
            f"  Customers to Contact: {int(opt_res['best_result']['total_contacted'])}")
        print(
            f"  Expected Conversions: {int(opt_res['best_result']['total_conversions'])}")
        print(
            f"  Expected Conversion Rate: {opt_res['best_result']['conversion_rate']:.1f}%")

    # Alternative: Run with local file
    # results = run_full_pipeline(
    #     raw_data_path='ml_project1_data.csv',
    #     processed_data_path='ifood_df.csv',
    #     clustered_data_path='ifood_df_clustered.csv',
    #     n_clusters=3,
    #     show_plots=True,
    #     use_kaggle=False,
    #     run_classification=True,
    #     optimize_profit=True
    # )

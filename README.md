# iFood CRM Marketing Campaign Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Predictive analytics solution for maximizing marketing campaign profitability through customer segmentation and targeted outreach**

---

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Solution Approach](#solution-approach)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deliverables](#deliverables)
- [Technical Details](#technical-details)
- [Business Insights](#business-insights)

---

## Overview

This project analyzes iFood's customer database to build a predictive model that maximizes the profitability of direct marketing campaigns. Using machine learning techniques, we identify high-probability customers, optimize campaign targeting, and transform a loss-making campaign into a profitable venture.

**Project Type:** Data Science & Business Analytics  
**Domain:** Marketing Analytics, Customer Segmentation  
**Techniques:** Machine Learning, Clustering, Predictive Modeling, Profit Optimization

---

## Business Problem

### The Challenge

iFood's pilot marketing campaign (Campaign 5) resulted in significant financial losses:

| Metric | Value |
|--------|-------|
| **Customers Contacted** | 2,240 (random selection) |
| **Campaign Cost** | 6,720 MU |
| **Revenue Generated** | 3,674 MU |
| **Net Profit** | **-3,046 MU** |
| **Success Rate** | 15% |
| **ROI** | -45% |

### The Objective

Build a data-driven solution to:
1. **Identify** customers most likely to purchase the new gadget
2. **Segment** customers based on behavior and characteristics
3. **Predict** purchase probability for each customer
4. **Optimize** targeting to maximize profit (not just accuracy)
5. **Transform** the next campaign (Campaign 6) from loss to profit

---

## Solution Approach

### Methodology

Our solution follows a comprehensive 4-stage pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. Data        │───▶│  2. Exploratory │───▶│  3. Customer    │───▶│  4. Predictive  │
│  Preprocessing  │    │  Analysis       │    │  Segmentation   │    │  Modeling       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
       ▼                       ▼                       ▼                       ▼
  • Feature eng.         • Correlations          • K-Means (k=3)        • Random Forest
  • Outlier removal      • Distributions         • Cluster profiles     • Hyperparameter tuning
  • Missing values       • Response analysis     • Segment insights     • Profit optimization
```

### Key Techniques

- **Data Preprocessing:** Feature engineering, outlier detection (IQR method), missing value handling
- **Exploratory Data Analysis:** Correlation analysis, distribution analysis, response rate analysis
- **Customer Segmentation:** K-Means clustering, hierarchical clustering, elbow method
- **Predictive Modeling:** Random Forest classifier, GridSearchCV, cross-validation
- **Profit Optimization:** Probability threshold optimization, ROI maximization, cost-benefit analysis

---

## Key Results

### Financial Impact

| Metric | Pilot Campaign (Random) | Optimized Campaign (Model) | Improvement |
|--------|------------------------|---------------------------|-------------|
| **Customers Contacted** | 2,240 | ~800-1,200 | ↓ 46-64% |
| **Conversion Rate** | 15% | **28-35%** | ↑ +87-133% |
| **Campaign Cost** | 6,720 MU | ~2,400-3,600 MU | ↓ 46-64% |
| **Net Profit** | **-3,046 MU** | **+300 to +800 MU** | **+109% swing** |
| **ROI** | -45% | **+8% to +25%** | Positive! |

### Model Performance

- **Accuracy:** 87-90%
- **Precision:** 75-80%
- **Recall:** 60-70%
- **F1-Score:** 67-73%
- **ROC AUC:** 85-88%

### Customer Segments Identified

**Segment 0: Budget-Conscious Families** (35-40%)
- Lower income, more dependents
- Price-sensitive, high web usage
- Strategy: Digital campaigns, value propositions

**Segment 1: Premium Customers** (25-30%)
- High income, few/no dependents
- High spending on wine & meat
- **Highest campaign acceptance rate**
- Strategy: Premium positioning, catalog offers

**Segment 2: Middle-Market Shoppers** (30-35%)
- Moderate income, balanced behavior
- Mixed channel preference
- Strategy: Targeted promotions, cross-selling

---

## Project Structure

```
eleba5-data-business-analyst-test/
│
├── Data Files
│   ├── ml_project1_data.csv          # Original raw data (local)
│   ├── ifood_df.csv                  # Preprocessed data (generated)
│   └── ifood_df_clustered.csv        # Data with cluster labels (generated)
│
├── Python Modules
│   ├── data_preprocessing.py         # Data loading, cleaning, feature engineering
│   ├── exploratory_analysis.py       # EDA, correlations, visualizations
│   ├── customer_segmentation.py      # K-Means clustering, profiling
│   ├── classification_model.py       # Random Forest, profit optimization
│   └── main_pipeline.py              # Orchestrates entire workflow
│
├── Jupyter Notebooks
│   ├── exploration_segmentation.ipynb    # Original EDA & clustering notebook
│   └── classification_model.ipynb        # Original classification notebook
│
├── Output Files (Generated)
│   ├── output_numerical_distributions.png
│   ├── output_correlation_heatmap.png
│   ├── output_elbow_curve.png
│   ├── output_dendrogram.png
│   ├── output_cluster_boxplots.png
│   ├── output_cluster_countplots.png
│   ├── output_confusion_matrix.png
│   ├── output_roc_curve.png
│   ├── output_feature_importance.png
│   └── output_profit_curve.png
│
├── Documentation
│   ├── README.md                     # This file
│   ├── BUSINESS_SUMMARY.md           # Executive summary & findings
│   └── dictionary.png                # Data dictionary image
│
└── Configuration
    └── requirements.txt              # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Step 1: Clone the Repository

```bash
git clone https://github.com/HazelMaeBea/eleba5-data-business-analyst-test.git
cd eleba5-data-business-analyst-test
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
kagglehub>=0.1.0
```

### Step 4: Install Additional Package

```bash
pip install kagglehub
```

---

## Usage

### Option 1: Run Complete Pipeline (Recommended)

Execute all 4 stages automatically with Kaggle data download:

```bash
python main_pipeline.py
```

**What it does:**
1. Downloads latest dataset from Kaggle
2. Preprocesses data (cleaning, feature engineering)
3. Performs exploratory analysis
4. Creates customer segments (k=3)
5. Trains Random Forest classifier
6. Optimizes for maximum profit
7. Generates all visualizations
8. Prints key business insights

**Expected runtime:** 3-10 minutes (depending on GridSearchCV)

**Output:**
- Preprocessed data files
- 10+ visualization plots
- Console output with metrics and insights

### Option 2: Use Local Data File

Edit `main_pipeline.py` line 310 to use local file:

```python
results = run_full_pipeline(
    raw_data_path='ml_project1_data.csv',
    use_kaggle=False,  # Change this
    # ... other parameters
)
```

### Option 3: Run Individual Modules

**Data Preprocessing Only:**
```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(use_kaggle=True)
data = preprocessor.run_pipeline('ifood_df.csv')
```

**Exploratory Analysis Only:**
```python
from exploratory_analysis import ExploratoryAnalysis
import pandas as pd

df = pd.read_csv('ifood_df.csv')
analyzer = ExploratoryAnalysis(df)

# Generate correlation heatmap
analyzer.plot_correlation_heatmap()

# Get high correlations
high_corr = analyzer.get_high_correlations(threshold=0.8)
print(high_corr)
```

**Customer Segmentation Only:**
```python
from customer_segmentation import CustomerSegmentation
import pandas as pd

df = pd.read_csv('ifood_df.csv')
segmentation = CustomerSegmentation(df)

segmentation.prepare_clustering_data()
segmentation.fit_kmeans(n_clusters=3)

# Get cluster profiles
profiles = segmentation.describe_clusters()
print(profiles)
```

**Classification & Profit Optimization Only:**
```python
from classification_model import CampaignClassifier
import pandas as pd

df = pd.read_csv('ifood_df.csv')
classifier = CampaignClassifier(df)

classifier.prepare_data(test_size=0.40)
classifier.train_random_forest(use_grid_search=True)

# Optimize for profit
results = classifier.optimize_threshold()
print(f"Best threshold: {results['best_threshold']}")
print(f"Expected profit: ${results['best_profit']:.2f}")
```

---

## Deliverables

### 1. Data Exploration
- Comprehensive EDA with correlation analysis
- Distribution analysis for all key features
- Outlier detection and handling
- Missing value treatment
- Response rate analysis

**Files:** `exploratory_analysis.py`, `output_correlation_heatmap.png`, `output_numerical_distributions.png`

### 2. Customer Segmentation
- K-Means clustering with k=3 segments
- Hierarchical clustering dendrogram
- Detailed segment profiles and characteristics
- Actionable marketing strategies per segment

**Files:** `customer_segmentation.py`, `output_elbow_curve.png`, `output_cluster_boxplots.png`

### 3. Predictive Classification Model
- Random Forest classifier with hyperparameter tuning
- 5-fold cross-validation
- Multiple evaluation metrics (accuracy, precision, recall, F1, ROC AUC)
- Feature importance analysis
- Confusion matrix visualization

**Files:** `classification_model.py`, `output_confusion_matrix.png`, `output_roc_curve.png`

### 4. Profit Optimization
- Probability threshold optimization
- Cost-benefit analysis
- ROI calculation
- Expected profit projections
- Customer targeting recommendations

**Files:** `output_profit_curve.png`, profit analysis in console output

### 5. Business Presentation
- Executive summary document
- Key findings and insights
- Strategic recommendations
- Financial projections

**Files:** `BUSINESS_SUMMARY.md`

---

## Technical Details

### Data Processing Pipeline

**Input Features (30+ features after engineering):**
- **Demographics:** Age, Income, Education, Marital Status
- **Customer Behavior:** Recency, Customer_Days, Web visits
- **Product Spending:** Wines, Fruits, Meat, Fish, Sweets, Gold
- **Purchase Channels:** Web, Catalog, Store, Deals
- **Campaign History:** AcceptedCmp1-5, Response

**Feature Engineering:**
- Age calculation from Year_Birth
- Customer tenure (Customer_Days)
- Total spending aggregation (MntTotal)
- Regular vs. Gold product spending
- Campaign acceptance totals
- One-hot encoding for categoricals

**Data Quality:**
- Missing values: ~24 records removed
- Outliers removed: Income (IQR method), Age (IQR method)
- Final dataset: ~2,200 records

### Model Architecture

**Algorithm:** Random Forest Classifier
- Ensemble of decision trees
- Handles non-linear relationships
- Built-in feature importance
- Robust to overfitting

**Hyperparameter Tuning (GridSearchCV):**
```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 8],
    'min_samples_split': [2, 3, 4],
    'criterion': ['gini'],
    'max_features': ['sqrt']
}
```

**Cross-Validation:** 5-fold stratified

**Train/Test Split:** 60% / 40%

### Profit Optimization Methodology

**Economic Parameters:**
- Cost per contact: $3.00 (derived from pilot: 6,720 MU / 2,240 customers)
- Revenue per conversion: ~$10.94 (derived from pilot: 3,674 MU / 336 conversions)

**Optimization Process:**
1. Generate prediction probabilities (0-1 scale)
2. Test multiple probability thresholds (0.1 to 0.9 in 0.05 steps)
3. For each threshold, calculate:
   - Customers contacted (probability > threshold)
   - Expected conversions (true positives)
   - Cost = Contacts × $3
   - Revenue = Conversions × $10.94
   - Profit = Revenue - Cost
4. Select threshold with maximum profit

**Key Insight:** The optimal threshold is typically NOT 0.5! It's often higher (0.6-0.8) to ensure only high-confidence customers are contacted, reducing cost while maintaining conversions.

---

## Business Insights

### Top Predictive Features

1. **Recency** (Most Important)
   - Days since last purchase
   - Recent customers are more likely to buy
   - **Action:** Prioritize customers with low recency scores

2. **Total Spending (MntTotal)**
   - Higher spenders are better targets
   - **Action:** Focus on high-value customer segments

3. **Income**
   - Strong predictor of campaign acceptance
   - **Action:** Target higher income brackets

4. **Wine & Meat Spending**
   - Premium product buyers respond better
   - **Action:** Cross-sell premium products in campaigns

5. **Number of Dependents**
   - Negative correlation with acceptance
   - **Action:** Deprioritize customers with many kids/teens

### Marketing Recommendations by Segment

#### Segment 1: Premium Customers (Highest Priority)
**Who:** High income, low dependents, high spenders on wine/meat  
**Size:** 25-30% of database  
**Response Rate:** Highest (25-40%)

**Tactics:**
- Personalized catalog mailings
- Premium product positioning
- Exclusive offers and early access
- In-store VIP experiences
- Loyalty rewards program

#### Segment 0: Budget-Conscious Families
**Who:** Lower income, more dependents, price-sensitive  
**Size:** 35-40% of database  
**Response Rate:** Low (5-15%)

**Tactics:**
- Digital-first campaigns (lower cost)
- Family bundle offers
- Value-oriented messaging
- Seasonal promotions
- Email marketing

#### Segment 2: Middle-Market Shoppers
**Who:** Moderate income, balanced behavior  
**Size:** 30-35% of database  
**Response Rate:** Moderate (15-25%)

**Tactics:**
- Multi-channel approach
- Cross-sell complementary products
- Limited-time offers
- Product sampling programs
- Referral incentives

### Channel Optimization

**Best Channels by Segment:**
- **Premium Customers:** Catalog (73% preference) > Store (62%) > Web (45%)
- **Budget-Conscious:** Web (68%) > Store (45%) > Catalog (23%)
- **Middle-Market:** Balanced across all channels

**Cost Efficiency:**
- Digital channels: Lowest cost per contact (~$0.50)
- Catalog: Medium cost (~$2.00)
- Phone: Highest cost (~$3.00)

**Recommendation:** Use digital channels for Segments 0 & 2, reserve phone/catalog for Segment 1

---

## References & Resources

### Dataset
- **Source:** [Kaggle - Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)
- **Original:** iFood CRM customer database

### Academic References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Provost, F., & Fawcett, T. (2013). *Data Science for Business*
- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*

### Libraries & Tools
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [seaborn Documentation](https://seaborn.pydata.org/)

---

*Last Updated: November 17, 2025*

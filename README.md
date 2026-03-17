# 🏥 Healthy Economies, Healthy Communities: The Digital-to-Health Pipeline
### Mastercard 2026 Data Challenge | Fisk University Team
**Target Geography:** Shelby County, TN (IGS < 45)

---

## 📌 Project Overview
Our research addresses the "Healthcare Gap" in Shelby County census tracts with an **Inclusive Growth Score (IGS) below 45**. Using a multi-model Machine Learning approach (XGBoost, Lasso, and Correlation Analysis), we identified that healthcare insurance coverage is not a standalone medical issue, but a downstream result of a **Digital-to-Labor pipeline**.

### The Core Thesis:
In Memphis's most vulnerable tracts, the "Matriarchal Floor" (economic stability of women) is the primary engine of health. However, this engine is currently stalled by a lack of digital infrastructure, creating a "Systemic Bottleneck" between labor engagement and insurance coverage.

---

## 📊 Technical Analysis & Evidence

### 1. Feature Importance (XGBoost)
Our model identified the high-leverage variables that actually drive healthcare access in Shelby County.

| Rank | Feature | Importance | Interpretation |
| :--- | :--- | :--- | :--- |
| **#1** | **Female Above Poverty Score** | **0.3350** | The "Master Variable" and primary predictor of community health. |
| **#2** | **Internet Access Score** | **0.0816** | The functional gatekeeper to workforce and insurance enrollment. |
| **#3** | **Labor Market Engagement** | **0.0610** | The engine of economic stability. |

### 2. The "Domino Effect" (Correlation Matrix)
We uncovered a clear chain of causality within low-IGS tracts ($r$ = correlation coefficient):
> **Internet Access** $\xrightarrow{r=0.56}$ **Labor Engagement** $\xrightarrow{r=0.54}$ **Female Poverty Stability** $\xrightarrow{r=0.60}$ **Health Insurance**

### 3. Lasso Regression Insights
By applying Lasso (L1) regularization, the model "zeroed out" **Residential Real Estate Value**. This confirms that in Shelby's crisis tracts, health disparities are driven by **participation and connectivity**, not property wealth.

---

## 💡 The Proposed Solution: The Trust-Capital Bridge

We propose a business-led intervention that turns local **Minority and Women-Owned Businesses (MWOBs)** into "Digital Health Gateways."

* **The Action:** Targeted **Mastercard-backed Impact Loans** to MWOBs to install high-speed digital kiosks.
* **The Personnel:** Funding for a local "Digital Health Navigator" to assist residents with enrollment.
* **The Impact:** By providing access at the local corner store, we solve the **"Time Tax"** (Travel Time barrier) and leverage existing community trust anchors.

---

## 🛠️ Reproducibility

### Repository Structure
* `shelby_analysis.py`: Main script containing the XGBoost and Lasso logic.
* `correlation_map.py`: Script used to generate the inter-factor relationship matrix.
* `full_feature_correlation.png`: Visual heatmap of the Shelby County landscape.

### Setup & Installation
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/mastercard-challenge-2026.git](https://github.com/YOUR_USERNAME/mastercard-challenge-2026.git)

# Install dependencies
pip install pandas polars scikit-learn xgboost seaborn matplotlib

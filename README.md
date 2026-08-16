## Overview
Despite the name, Logistic Regression is a **classification** algorithm, not a regression one — it predicts the probability that an observation belongs to a given class. It's the natural extension of linear regression to classification problems: rather than predicting an unbounded continuous value, it passes the linear combination of features through a **sigmoid (logistic) function** to squash the output into a valid probability between 0 and 1.

## How It Works
1. Compute the linear combination: **z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ**
2. Apply the **sigmoid function:** **P(y=1) = 1 / (1 + e^(-z))**, mapping z (which can range from −∞ to +∞) into a probability between 0 and 1
3. Classify based on a threshold (default 0.5): predict class 1 if P(y=1) > 0.5, else class 0
4. Coefficients are learned by maximizing the **log-likelihood** of the observed data (equivalently, minimizing **log loss / binary cross-entropy**) via iterative optimization (gradient descent, or solvers like L-BFGS/Newton's method) — there's no closed-form solution like OLS has for linear regression.

## Interpreting Coefficients
Each coefficient βᵢ represents the change in the **log-odds** of the outcome per one-unit increase in that feature. Exponentiating a coefficient (e^β) gives the **odds ratio** — a more intuitive way to communicate effect size (e.g. "each additional year of tenure multiplies the odds of attrition by 0.92").

## Methods & Techniques
- **Multi-class extensions:**
  - **One-vs-Rest (OvR):** trains one binary classifier per class (class vs. all others), predicts the class with the highest probability
  - **Multinomial (Softmax) Logistic Regression:** a single model that directly outputs a probability distribution across all classes simultaneously — generally preferred when classes are mutually exclusive
- **Regularization:** same L1 (Lasso)/L2 (Ridge)/Elastic Net options as linear regression, applied to the log-likelihood objective — critical when there are many features relative to samples, or correlated features
- **Class imbalance handling:** `class_weight='balanced'` (upweights the minority class in the loss function), oversampling (SMOTE), undersampling, or adjusting the classification threshold away from the default 0.5 based on the precision/recall trade-off that matters for the use case
- **Feature scaling:** important when using regularization (same reasoning as Ridge/Lasso above) and helps gradient-based solvers converge faster
- **Decision threshold tuning:** the default 0.5 cutoff is often wrong for the actual business problem — use a **Precision-Recall curve** or **ROC curve** to pick a threshold matching the real cost of false positives vs. false negatives (e.g. in fraud or disease detection, you often want a lower threshold to catch more true positives at the cost of more false alarms)

## Evaluation Metrics
- **Accuracy** — only meaningful when classes are reasonably balanced
- **Precision, Recall, F1-score** — essential when classes are imbalanced
- **ROC-AUC** — measures ranking quality across all thresholds, useful for balanced problems
- **Precision-Recall AUC** — more informative than ROC-AUC on heavily imbalanced datasets
- **Confusion Matrix** — breaks down exactly which classes are being confused with which

## When to Use It
Best for binary or multi-class classification where interpretability matters, features have a roughly linear relationship with the log-odds of the outcome, and you need well-calibrated probability outputs (not just hard class labels) — logistic regression's probabilities tend to be better calibrated out-of-the-box than many tree-based models.

## Strengths & Limitations
| Strengths | Limitations |
|---|---|
| Outputs interpretable probabilities, not just labels | Assumes a linear decision boundary (in log-odds space) |
| Fast to train and predict | Struggles with complex, nonlinear class boundaries without engineered features |
| Coefficients are directly interpretable (odds ratios) | Sensitive to multicollinearity |
| Well-calibrated probabilities | Requires more careful feature engineering than tree-based models to capture interactions |

---
---

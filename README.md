# Box Office Revenue Prediction

This project focuses on predicting the box office revenue of movies using machine learning models. The dataset includes various pre-release attributes such as budget, runtime, popularity, and vote statistics. The project builds and compares multiple regression models to estimate a movie's potential earnings.

---

## Project Objectives

* Predict movie revenue using structured data features.
* Compare multiple machine learning regression models.
* Evaluate model accuracy and performance.
* Perform EDA and feature engineering to enhance prediction accuracy.

---

## Technologies Used

**Programming Language:** Python
**Libraries:** Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
**Tools & Platforms:** Jupyter Notebook, VS Code, Git
**Dataset Source:** Kaggle - [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## Dataset Description

* **Type:** Structured (CSV format)
* **Records:** \~4,800 movies
* **Features:** Budget, runtime, popularity, vote average, vote count, release date, etc.
* **Target Variable:** Revenue
* **Missing Values:** Present in runtime, budget, and release\_date; handled during preprocessing

### Sample Features Used:

| Feature        | Description                          |
| -------------- | ------------------------------------ |
| budget         | Production budget (USD)              |
| runtime        | Movie length in minutes              |
| popularity     | Aggregated popularity metric         |
| vote\_average  | Average user rating (0–10 scale)     |
| vote\_count    | Number of user votes                 |
| release\_year  | Year extracted from release date     |
| release\_month | Month extracted from release date    |
| high\_budget   | Flag for movies with budget > \$100M |

---

## Methodology

1. **Data Collection** – Dataset sourced from Kaggle.
2. **Data Cleaning** – Handled nulls, converted date fields, removed invalid entries.
3. **Feature Engineering** – Created new fields like `release_year`, `high_budget`, and ROI.
4. **Train-Test Split** – 80% training and 20% testing data.
5. **Model Implementation** – Trained:

   * Linear Regression
   * Decision Tree Regressor
   * K-Nearest Neighbors Regressor
6. **Evaluation Metrics** – R² Score, RMSE, MAE.
7. **Visualization** – Revenue distribution, correlation matrix, actual vs. predicted plots.

---

## Model Evaluation

| Model             | R² Score (%) | RMSE     |
| ----------------- | ------------ | -------- |
| Linear Regression | \~48%        | Moderate |
| Decision Tree     | \~53%        | Moderate |
| KNN Regressor     | \~50%        | High     |

### Visuals:

* Revenue distribution (histplot)
* Budget vs. Revenue (scatterplot)
* Correlation heatmap
* Actual vs Predicted revenue plot
* Bar chart for model accuracy comparison

---

## Conclusion

The project demonstrated the application of supervised learning models to predict movie revenue. Among the three models used, the Decision Tree Regressor performed slightly better. The accuracy of all models was moderate, suggesting that additional contextual or unstructured data could further enhance performance.

The process deepened understanding of the machine learning workflow, from preprocessing to model evaluation. It also highlighted the complexity of financial forecasting in the entertainment industry.

---

## Scope for Future Enhancement

* Use of text features such as movie overviews with NLP.
* Incorporate genres, cast, and production companies using encoding techniques.
* Apply advanced models like Random Forest and XGBoost.
* Convert regression into classification (e.g., hit vs flop).
* Deploy model in a web-based interface for user interaction.
* Improve explainability with SHAP or LIME.

---

## How to Run the Project

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/box-office-prediction.git
cd box-office-prediction
```

2. **Set up environment:**

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook:**

```bash
jupyter notebook
```

---

## References

* Kaggle Dataset: [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
* Scikit-learn Documentation: [https://scikit-learn.org/stable/user\_guide.html](https://scikit-learn.org/stable/user_guide.html)
* NumPy Documentation: [https://numpy.org/doc/](https://numpy.org/doc/)
* Matplotlib: [https://matplotlib.org/](https://matplotlib.org/)
* Seaborn: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
* Brownlee, J. (2020). *Machine Learning Mastery With Python*.

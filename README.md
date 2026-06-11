# synent-Task-8-House-Price-ML-Model-Raj-Bhut

## House Price Prediction Machine Learning Model

### Problem Statement
Build an end-to-end machine learning regression pipeline to predict property market prices using structural and spatial features. The objective is to establish a robust framework that accurately evaluates how individual variables like total living area, building age, property condition, and geographic location influence overall real estate valuation.

### Dataset
* **House Price Prediction Dataset.csv:** Features ~2,000 structural records containing property identifiers, configuration counts (bedrooms, bathrooms, floors), construction year, qualitative features (location segment, building condition, garage availability), and the target continuous variable (`Price`).

### Tasks Performed
* Loaded the dataset and executed initial structural inspections, checking shape boundaries and data types inside VS Code.
* Checked for missing values and analyzed descriptive statistics to ensure overall dataset integrity before training.
* Preprocessed categorical matrices (`Location`, `Condition`, and `Garage`) into numerical formats compatible with machine learning algorithms.
* Isolated structural predictors and assigned data subsets through feature selection techniques against the target scalar (`Price`).
* Partitioned records into independent training and testing sets to guarantee unbiased model evaluation.
* Implemented and trained a baseline Linear Regression model alongside an ensemble Random Forest Regressor to capture complex, non-linear relationships.
* Evaluated both models using standard regression metrics, specifically tracking performance via Root Mean Squared Error (RMSE) and R² Score.
* Serialized and exported the champion predictive model configuration to a standalone `.pkl` file for production reuse and quick deployment.

### Tools Used
* Python
* Pandas
* NumPy
* Scikit-Learn
* Pickle
* VS Code

### Results
Successfully engineered a predictive framework that outputs property market valuations, benchmarks basic linear assumptions against complex decision trees, and saves the optimized estimator configuration directly as `task8_house_price_model.pkl`.

### Conclusion
The analysis successfully mapped out the core drivers behind residential property values, demonstrating that an ensemble modeling approach effectively captures non-linear relationships in housing attributes while providing a production-ready file optimized for real-time inference workflows.

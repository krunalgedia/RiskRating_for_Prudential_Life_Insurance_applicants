# Risk Rating Model for Prudential Life Insurance applicants

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Workflow](#workflow)
- [Results](#results)
- [More ideas](#More-ideas)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Project Overview
### Information:
* Prudential, a leading life insurance issuer in the USA, is establishing a Data Science group to tackle complex challenges and identify opportunities.
* The company has a long-standing business experience of 140 years, providing a stable foundation for innovation.
* The current life insurance application process is outdated and time-consuming, taking an average of 30 days.
* Only 40% of U.S. households own individual life insurance, indicating a need for improvement in the industry.

### Goal:
* Prudential aims to revolutionize the life insurance application process by developing a predictive model that accurately classifies risk using automation.
* The goal is to make the application process quicker and less labor-intensive, thereby enhancing public perception of the industry and increasing accessibility to life insurance.

## Data

*The dataset is taken from Kaggle [1] uplaoded by Prudential
*In this dataset, you are provided over a hundred variables describing attributes of life insurance applicants. The task is to predict the "Response" variable for each Id in the test set. "Response" is an ordinal measure of risk that has 8 levels.

## Workflow

1. Importing data
2. Observing data especially target variable for its imbalance and high cardinality columns.
  ![Image](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/target.png)
   -----|
   Target variable (Imbalanced risk rating classes)
![Image](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/bin_cardinality.png)
   -----|
   High Cardinality variables
   
4. Check columns with null values, remove them if they have more than 30% null values. If less than 30%, perform medium imputation in the ML model pipeline ensuring median values used to impute null values are taken only from the train set (even during cross-validation).
5. Make train test Stratified split. Compute class weights and use them in the later models to compute weighted loss while training.
6. Create model pipelines for XGBoost and Logistic Regression, and optimize model hyperparameters using Bayesian optimization. Calculate Mathew's correlation coefficient (MCC) as a metric and get feature importance using the SHAPLEY score.
![Image 1](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/xgblr.png) 
7. Make a simple feed-forward neural network (DNN) and using Logistic regression output as a skip connection to the last layer of DNN, make Combined Actuarial Neural Network (CANN) [2]. CANN was initially introduced in the context of a regression model.

![Image 1](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/dnn.png) | ![Image 2](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/cann.png)
:-------------------------:|:-------------------------:
Feed-forward neural network DNN | Combined Actuarial Neural Network CANN (Logistic Regression as skip connection)

8. Now compare the loss of the DNN and CANN model and also the MCC metric.
  

* notebooks/SBB_TrainTicketParser.ipynb contains the end-to-end code for Document parsing with database integration.

## Results


Following is the result for optimized Logistic regression and XGBoost on the test dataset.

| Model | Mathew's Correlation Co-efficient | 
|--------:|------------:|
|  Logistic Regression | 0.30 |
| XGBoost | 0.41 |

The SHAP value for each of the models is
![Image](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/shap.png)
As expected, the important parameters are around the same for each of the models. We do expect some differences because Logistic regression is a linear regression model with the target variable as logit(probability) while XGBoost is a gradient-boosted decision tree-based non-linear algorithm.

Following is the loss and metric for Logistic regression, DNN, and CANN.
![Image](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/results.png)
The important thing to note here is that CANN begins with a lower loss and better MCC metric than DNN. This is because of the skip connection which uses Logistic regression output. However, given the high non-linear nature of neural networks, eventually both models, DNN and CANN are bound to perform similarly over a few epochs.

## More ideas

Instead of using OCR from the UBIAI tool, it best is to use pyteserract or same OCR tool for train and test set. Further, with Document AI being developed at a rapid pace, it would be worthwhile to test newer multimodal models which hopefully either provide a new solution for not using OCR or inbuilt OCR since it is important to be consistent in preprocessing train and test set for best results.

Also, train on at least >50 tickets, since this was just a small test case to see how well the model can work.

## Dependencies

This project uses the following dependencies:

- **Python:** 3.10.12/3.9.18 
- **PyTorch:** 2.1.0+cu121/2.1.1+cpu
- **Streamlit:** 1.28.2 

- [SBB ticket parser model on Hugging Face](https://huggingface.co/KgModel/sbb_ticket_parser_LayoutLM)
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## References
[1]: Data: [prudential-life-insurance-assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment)

[2] CANN: [CANN](Wüthrich, Mario V., and Michael Merz, ‘EDITORIAL: YES, WE CANN!’, ASTIN Bulletin, 49 (2019), 1–3 <http://dx.doi.org/10.1017/asb.2018.42>)



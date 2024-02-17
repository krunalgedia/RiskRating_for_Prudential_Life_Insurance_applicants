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
   
4. Check columns with null values, remove them if they have more than 30% null values. If less than 30%, perform medium imputation in the ML model pipeline ensuring median values used to impute null values are taken only from train set (even during cross validation).
5. Create model pipelines for XGBoost and Logistic Regression, and optimize model hyperparameters using Bayesian optimization. Calculate Mathew's correlation coefficient as a metric and get feature importance using SHAPLEY score.
![Image 1](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/xgblr.png) 
6. Make a simple feed forward neural network and using Logistic regression output as a skip connection to the last layer of DNN, make Combined Acturial Neural Network (CANN) [2]. CANN was initially introduced in context of a regression model.

![Image 1](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/dnn.png) | ![Image 2](https://github.com/krunalgedia/RiskRating_for_Prudential_Life_Insurance_applicants/blob/main/images_README/cann.png)
:-------------------------:|:-------------------------:
Opening page | Testing ...

8. 
9.  



10. Preparing test set processing, including OCR of prediction documents using Pytesseract and getting the bounding box for all text in the test sample.
11. Running predictions on the bounding boxes of Pytesseract.
12. Update the database with relevant NER extracted from the model prediction on the annotated test sample.

* notebooks/SBB_TrainTicketParser.ipynb contains the end-to-end code for Document parsing with database integration.
* app.py contains the streamlit app code.

## Results

We fine-tuned using Facebook/Meta's LayoutLM (which utilizes BERT as the backbone and adds two new input embeddings: 2-D position embedding and image embedding) [3]. The model was imported from the Hugging Face library [4] with end-to-end code implemented in PyTorch. We leveraged the tokenizer provided by the library itself. For the test case, we perform the OCR using Pytesseract.

With just 4 SBB train tickets we can achieve an average F1 score of 0.81.   

| Epoch | Average Precision | Average Recall | Average F1 | Average Accuracy |
|--------:|------------:|---------:|-----:|-----------:|
|     145 |        0.89 |     0.77 | 0.82 |       0.9  |
|     146 |        0.9  |     0.79 | 0.84 |       0.9  |
|     147 |        0.86 |     0.77 | 0.81 |       0.89 |
|     148 |        0.87 |     0.78 | 0.82 |       0.9  |
|     149 |        0.86 |     0.77 | 0.81 |       0.89 |

The web application serves demo:
![Image 1](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/sample.gif) | ![Image 2](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/test1.gif)
--- | --- 
Opening page | Testing ... 

Once the user uploads the image, the document gets parsed and the information from the document gets updated in the relational database which can be used to verify the traveler's info and also to automate the travel cost-processing task.


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



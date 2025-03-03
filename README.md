# Vehicle-Insurance-Claim-Amount-Predictor
A comprehensive M.L model based upon XGBoost regressor, EfficientNet B0, XGBoost Classifier and Tensorflow which helped me to predict vehicle's claim amount as per the car's condition and it's respective image.

I made a model that could predict claim amount and car condition using the following techniques->

XGboost classifier , EfficientNetB0, XGboost regressor, tensorflow

and acheived an excellent mae of 830 .

Also serialized the models, automated the data ingestion and built a governance layer and continuous learning pipeine.

Now on what basis I made the project was that i had a train.csv(columns like image_path, cost of vehicle, mincoverage, max-coverage, insurance_cmpany, condition, amount) and trainImages folder with respective image names same as image_path of train.csv.

then tested the whole project on test.csv and testImages folder to obtain  a test_predictions.csv with predicted amounts and conditions.


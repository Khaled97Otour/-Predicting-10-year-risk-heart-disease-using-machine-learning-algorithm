# Predicting-10-year-risk-heart-disease-using-machine-learning-algorithm

In this project, I use data from kaggle.com to study the heart disease :https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
this data contain the following:

1. male
2. age
3. education
4. currentSmoker
5. cigsPerDay
6. BPMeds
7. prevalentStroke
8. prevalentHyp
9. diabetes
10. totChol
11. sysBP
12. diaBP
13. BMI
14. heartRate
15. glucose

# building the model:
For classification first I will build a normail logictic model and study the results.
. I read the data and try to find a class by printing the histogram 

![download](https://user-images.githubusercontent.com/93203143/182427940-e268039b-7f29-4e58-947d-fc82b5046d59.png)

. I used diffrent types of classification models (KNN, Decision Tree and random forest) and the results  were the following:
1. by using a logictic classification I was able to have the best results.
|  /            | lr            |l1             |l2             |
| ------------- | ------------- | ------------- | ------------- |
| precision     |0.821592       |0.823000       | 0.819658| 
| recall     | 0.850608 |0.851322|0.850608|
| fscore     | 0.792638 |0.795316|0.793772|
|accuracy    |0.850608|0.851322|0.850608|
|auc         |0.523169|0.527420|0.525084|

![download (1)](https://user-images.githubusercontent.com/93203143/182428196-57872976-d515-47a8-9dbb-396900ac623f.png)

2. I use the KNN method to find the ideal number.However, the F1 scores were to low.
 
![download (2)](https://user-images.githubusercontent.com/93203143/182428430-3822ad5c-64ce-49d2-9a56-1f3d602c68e1.png)
![download (3)](https://user-images.githubusercontent.com/93203143/182428436-4803c93b-2260-4c6b-af01-530c2cff22e6.png)

3. Decision Tree and random Forest shows low F1 and recall scores.
	          train	      test
accuracy	  1.0	      0.742673
precision	 1.0	      0.190678
recall	    1.0	      0.210280
f1	        1.0	      0.200000
 
![download (4)](https://user-images.githubusercontent.com/93203143/182428630-f6befaf6-c14e-467d-90a2-200c7af8033d.png)
![download (5)](https://user-images.githubusercontent.com/93203143/182428640-a3723d8d-6c99-430d-a315-39303cec2377.png)

4. One of the down side is the unbalance of our dataset.
5. The large number of our features are one of the reason of overfitting (complexity).

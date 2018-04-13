#Logistical Regression

If a new dataset is provided with participation in races since 2003, one can load the file and use
>>X_train, y_train, X_test, y_test, X_val, y_val = deal_with_data(csvfile)
in order to load it. 

To test the various alphas or the various complexities and generate graphs for each one. One can use the two methods that are commented in the main method.

>>alphaTests(X_train, y_train, X_val, y_val)
>>sample_Complexity(X_train, y_train, X_val, y_val) 

To train the classifier and see all the scores over the test set, use:
>>weights1 = testSetMeasures(X_train, y_train, X_test, y_test)

To make new new predictions on the 2017 data, reuse the weights you have found and apply them to the 2017 data:
>>getPredictions2017(weights1)



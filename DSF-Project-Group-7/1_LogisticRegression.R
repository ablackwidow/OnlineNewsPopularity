
# --------------- LOGISTIC REGRESSION ---------------

# --------------- 1: use logistic regression on original dataset ---------------

# train glm-model incl. 10-fold cross-validation 
logist_news_og = train(virality ~ .,
                        data = news_train,
                        trControl = trainControl(method = "repeatedcv", number = 10),
                        method = "glm", 
                        family=binomial())

# use glm-model to predict our test-data
pred_og_prob = predict(logist_news_og, news_test, type = "prob")
pred_og <- predict(logist_news_og, news_test, type = "raw")

# display performance measures
confusionMatrix(pred_og, news_test$virality, mode="prec_recall", positive = "1")

#       0     1
# 0 10652  1221
# 1    13     5

# accuracy = 0.8962          
# precision = 0.2777778         
# recall = 0.0040783         
# balanced accuracy = 0.5014297         

# calculate area under the roc curve
pred_og <- as.numeric(pred_og)
roc_og <- roc(news_test$virality, pred_og, plot=T)

# AUC = 0.5014

# --------------- 2: use logistic regression on ROSE dataset---------------

# train glm-model incl. 10-fold cross-validation
logist_news_rose = train(virality ~ .,
                         data = ovun_news_train,
                         trControl = trainControl(method = "repeatedcv", number = 10),
                         method = "glm",
                         family=binomial())

# use glm-model to predict our test-data
pred_rose_prob = predict(logist_news_rose, news_test, type = "prob")
pred_rose = predict(logist_news_rose, news_test, type = "raw")

# display performance measures
confusionMatrix(pred_rose, news_test$virality, mode="prec_recall", positive = "1")

#       0    1
# 0 7421  513
# 1 3244  713    

# accuracy = 0.684
# precision = 0.18019
# recall = 0.58157
# balanced accuracy = 0.63870


# calculate area under the roc curve
pred_rose <- as.numeric(pred_rose)
roc_rose <- roc(news_test$virality, pred_rose, plot=T)

# AUC = 0.6387

# --------------- 3: use logistic regression on undersampled dataset---------------

# train glm-model incl. 10-fold cross-validation
logist_news_down = train(virality ~ .,
                       data = news_down,
                       trControl = trainControl(method = "repeatedcv", number = 10),
                       method = "glm",
                       family=binomial())

# use glm-model to predict our test-data
pred_down_prob = predict(logist_news_down, news_test, type = "prob")
pred_down = predict(logist_news_down, news_test, type = "raw")

# display performance measures
confusionMatrix(pred_down, news_test$virality, mode="prec_recall", positive = "1")

#      0    1
# 0 7249  493
# 1 3416  733

# accuracy = 0.6713
# precision = 0.17667
# recall = 0.59788
# balanced accuracy = 0.63879


# calculate area under the roc curve
pred_down <- as.numeric(pred_down)
roc_down <- roc(news_test$virality, pred_down, plot=T)

# AUC = 0.6388

# --------------- 4: use logistic regression on oversampled dataset---------------

# train glm-model incl. 10-fold cross-validation
logist_news_over = train(virality ~ .,
                         data = up_Train,
                         trControl = trainControl(method = "repeatedcv", number = 10),
                         method = "glm",
                         family=binomial())

# use glm-model to predict our test-data
pred_over_prob = predict(logist_news_over, news_test, type = "prob")
pred_over = predict(logist_news_over, news_test, type = "raw")

# display performance measures
confusionMatrix(pred_over, news_test$virality, mode="prec_recall", positive = "1")

#      0    1
# 0 7350  505
# 1 3315  721

# accuracy = 0.6787
# precision = 0.17864
# recall = 0.58809
# balanced accuracy = 0.63863


# calculate area under the roc curve
pred_over <- as.numeric(pred_over)
roc_over <- roc(news_test$virality, pred_over, plot=T)

# AUC = 0.6386

# --------------- 5: use logistic regression on SMOTE dataset---------------

# train glm-model incl. 10-fold cross-validation
logist_news_smote = train(virality ~ .,
                         data = smote_Train,
                         trControl = trainControl(method = "repeatedcv", number = 10),
                         method = "glm",
                         family=binomial())

# use glm-model to predict our test-data
pred_smote = predict(logist_news_smote, news_test, type = "raw")

# display performance measures
confusionMatrix(pred_smote, news_test$virality, mode="prec_recall", positive = "1")

#      0    1
# 0 8609  801
# 1 2056  425

# accuracy = 0.7597
# precision = 0.17130
# recall = 0.34666
# balanced accuracy = 0.57694


# calculate area under the roc curve
pred_smote <- as.numeric(pred_smote)
roc_smote <- roc(news_test$virality, pred_smote, plot=T)

# AUC= 0.5769

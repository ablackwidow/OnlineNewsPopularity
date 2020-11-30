
# --------------- LINEAR DISCRIMINANT ANALYSIS (LDA) ---------------

# in this script we will go through the process of Linear Discriminant Analysis 

# load packages 
library(magrittr)
library(MASS)
library(tidyverse) 
library(caret) 
library(rsample) 
library(DMwR)
library(ROSE)
library(boot)
library(ROCR)
library(naivebayes)

# --------------- 1: use LDA on original dataset ---------------

# train lda-model incl. 10-fold repeated cross-validation 
LdaKFcvOriginal <- train(virality ~., 
                        data = news_train,  
                        method = "lda", 
                        trControl = trainControl(## 10-fold CV
                          method = "repeatedcv",
                          number = 10,
                          ## repeated ten times
                          repeats = 10)
                        )

LdaKFcvresults <- predict(LdaKFcvOriginal, newdata=news_test)
table(LdaKFcvresults)

a  <-  as.factor(LdaKFcvresults)
b <-  as.factor(news_test$virality)

print(confusionMatrix(data = a, reference= b, mode = "prec_recall", positive="1"))

#       0     1
# 0 10608  1204
# 1    57    22

# accuracy = 0.894           
# precision = 0.278481        
# recall = 0.017945 r   
# balanced accuracy = 0.506300        

# plot ROC
LDAKFCVproba <- predict(LdaKFcvOriginal, newdata=news_test, type="prob")

probabilities <- LDAKFCVproba[,-1]
ROCR <- prediction(probabilities, news_test$virality)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for LDA, 10-FoldCV", print.auc = T)
abline(a=0, b=1)

# compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AROUC") 

# AUC = 0.6879

# train lda-model incl. leave-one-out cross-validation 
modelTrained <- train(virality ~., 
                      data = news_train,  
                      method = "lda", 
                      trControl = trainControl(method = "LOOCV"))
# save(modelTrained, file = "LDAModels")

LdaLOOCVresults <- predict(modelTrained, newdata=news_test)
table(LdaLOOCVresults)

a <- as.factor(LdaLOOCVresults)
b <- as.factor(news_test$virality)

print(confusionMatrix(data = a, reference= b, mode = "prec_recall", positive="1"))
# be sure that the measures take 1 as positive class! => since we are interested in
# prediciting especially viral articles!

#       0     1
# 0 10612  1165
# 1    72    43

# accuracy = 0.896           
# precision : 0.373913        
# recall : 0.035596        
# balanced accuracy : 0.514428        

# plot ROC
LDALOOCVproba <- predict(modelTrained, newdata=news_test, type="prob")

probabilities <- LDALOOCVproba[,-1]
ROCR <- prediction(probabilities, news_test$virality)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for LDA, originalLOOCV", print.auc = T)
abline(a=0, b=1)

# compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AROUC")  

# AUC = 0.6988 


# --------------- 2: use logistic regression on ROSE dataset ---------------

LdaKFcvRose <- train(virality ~., 
                     data = ovun_news_train,  
                     method = "lda", 
                     trControl = trainControl(## 10-fold CV
                       method = "repeatedcv",
                       number = 10,
                       ## repeated ten times
                       repeats = 10)
                     )

LdaKFcvresults <- predict(LdaKFcvRose, newdata=news_test)
table(LdaKFcvresults)

a <- as.factor(LdaKFcvresults)
b <- as.factor(news_test$virality)

print(confusionMatrix(data = a, reference= b, mode = "prec_recall", positive="1"))

#      0    1
# 0 7505  504
# 1 3179  704

# accuracy = 0.6903          
# precision = 0.1813          
# recall = 0.5828          
# balanced accuracy = 0.6426          

# plot ROC
LDAKFCVproba <- predict(LdaKFcvRose, newdata=news_test, type="prob")

probabilities <- LDAKFCVproba[,-1]
ROCR <- prediction(probabilities, news_test$virality)
per <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for LDA, 10-FoldCV", print.auc = T)
abline(a=0, b=1)

# compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- ound(auc, 4)
legend(.6, .2, auc, title = "AROUC") 

# AUC = 0.6956


# --------------- 3: use LDA on undersampled dataset ---------------
LdaKFcvUnder <- train(virality ~., 
                     data = news_down,  
                     method = "lda", 
                     trControl = trainControl(## 10-fold CV
                       method = "repeatedcv",
                       number = 10,
                       ## repeated ten times
                       repeats = 10)
                     )

LdaKFcvresults<-predict(LdaKFcvUnder, newdata=news_test)
table(LdaKFcvresults)

a <- as.factor(LdaKFcvresults)
b <- as.factor(news_test$virality)

print(confusionMatrix(data = a, reference= b, mode = "prec_recall", positive="1"))

#      0    1
# 0 7369  521
# 1 3296  705

# accuracy = 0.679           
# precision = 0.17621         
# recall = 0.57504         
# balanced accuracy = 0.63300         
            

# plot ROC
LDAKFCVproba <- predict(LdaKFcvUnder, newdata=news_test, type="prob")

probabilities <- LDAKFCVproba[,-1]
ROCR <- prediction(probabilities, news_test$virality)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for LDA, 10-FoldCV", print.auc = T)
abline(a=0, b=1)

# compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AROUC") 

# AUC = 0.6968


# --------------- 4: use logistic regression on oversampled dataset ---------------

LdaKFcvOver <- train(virality ~., 
                     data = up_Train,  
                     method = "lda", 
                     trControl = trainControl(## 10-fold CV
                       method = "repeatedcv",
                       number = 10,
                       ## repeated ten times
                       repeats = 10)
                     )

LdaKFovercvresults <- predict(LdaKFcvOver, newdata=news_test)
table(LdaKFovercvresults)

a <-  as.factor(LdaKFovercvresults)
b <-  as.factor(news_test$virality)

print(confusionMatrix(data = a, reference= b, mode = "prec_recall", positive="1"))

#      0    1
# 0 7473  501
# 1 3211  707

# accuracy = 0.6879          
# precision = 0.18045         
# recall = 0.58526         
# balanced accuracy = 0.64236         
     


# plot ROC
LDAKFCVproba <- predict(LdaKFcvOver, newdata=news_test, type="prob", positive="1")

probabilities <- LDAKFCVproba[,-1]
ROCR<- prediction(probabilities, news_test$virality)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for LDA, 10-FoldCV", print.auc = T)
abline(a=0, b=1)

# compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AROUC") 
# AUROC: 0.6971


# --------------- 5: use logistic regression on SMOTE dataset ---------------

LdaKFcvSmote <- train(virality ~., 
                      data = smote_Train,  
                      method = "lda", 
                      trControl = trainControl(## 10-fold CV
                        method = "repeatedcv",
                        number = 10,
                        ## repeated ten times
                        repeats = 10)
                      )

LdaKFcvresults <- predict(LdaKFcvSmote, newdata=news_test)
table(LdaKFcvresults)


a <- as.factor(LdaKFcvresults)
b <- as.factor(news_test$virality)

print(confusionMatrix(data = a, reference= b, mode = "prec_recall", positive="1"))

#      0    1
# 0 8690  775
# 1 1994  433

# accuracy = 0.7672          
# precision = 0.17841         
# recall = 0.35844         
# balanced accuracy = 0.58590         


# plot ROC
LDAKFCVproba <- predict(LdaKFcvDown, newdata=news_test, type="prob", positive="1")

probabilities <- LDAKFCVproba[,-1]
ROCR <- prediction(probabilities, news_test$virality)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for LDA, 10-FoldCV", print.auc = T)
abline(a=0, b=1)

# compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AROUC") 

# AUC = 0.6968

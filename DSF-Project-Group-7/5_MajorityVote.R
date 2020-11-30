
# --------------- MAJORITY VOTING ---------------

library(tidyverse)
library(caret)  
library(rsample)
library(mlbench)
library(caretEnsemble)
library(ROSE)
library(randomForest)
library(ranger)
library(MASS)
library(keras)
library(dplyr)
rm(list = ls())
graphics.off()

set.seed(123)

news0 = read.table("OnlineNewsPopularity.csv", header = TRUE, sep = ",")


# classify the data between viral and non-viral
# where 1 stands for viral and 0 for non-viral

quantile_viral <- quantile(news0$shares, probs = 0.9)
quantile_viral <- as.numeric(quantile_viral)

news1 <- news0 %>% 
  mutate(virality = ifelse(as.numeric(news0$shares) >= quantile_viral, 1, 0)) %>% 
  dplyr::select(-url, -timedelta, -shares)

table(news1$virality) 

# put column virality in front of data-frame
virality <- news1$virality
news1 <- cbind(virality, news1[ , 1:58])
class(news1$virality)

###
#partition
set.seed(123)
perc_train <- 0.7  

initial_split <- initial_split(news1, prop = perc_train, strata = news1$virality)  
news_train <- training(initial_split)  
news_test <- testing(initial_split)

###
#shuffle the data
rows <- sample(nrow(news_train))
news_train <- news_train[rows, ]


# --------------- logistic regression ---------------

# using undersampled logistic regression-model, since it has the best weighted average for logistic regression

#do the downsampling
news_train$virality <- as.factor(news_train$virality)
news_down <- downSample(x = news_train[,2:59],
                        y = news_train$virality)

set.seed(123)
news_down <- news_down[sample(1:nrow(news_down)), ]  

# drop class
news_down$virality <- news_down$Class

# take target as first column
news_down <- news_down %>% dplyr::select(-Class)
news_down <- news_down %>%  dplyr::select(virality,everything())

table(news_down$virality)

logist_news_down <- train(virality ~ .,
                         data = news_down,
                         trControl = trainControl(method = "cv", number = 10),
                         method = "glm",
                         family=binomial())

pred_down_probabilities <- predict(logist_news_down, news_test)
pred_down <- predict(logist_news_down, news_test, type = "raw")

confusionMatrix(as.factor(pred_down), as.factor(news_test$virality), mode="prec_recall", positive = "1")


# --------------- random forest ---------------

#load the trained model
load("optimal_ranger_ovun.RData")
optimal_ranger_ovun

#optain classes prediction
pred_t <- predict(optimal_ranger_ovun, news_test)
pred_t <- pred_t[["predictions"]]

pred_t <- as.vector(pred_t)


# --------------- neural networks ---------------

#UNDERSAMPLED Neural networks, We take this one because the best weighted average for ANN
# was with the normal dataset. The normal dataset will not help us much since it only predicts non-viral
#article, therefore we chose the undersampled one.

set.seed(123)

perc_train <- 0.7  

initial_split <- initial_split(news1, prop = perc_train, strata = news1$virality)  

news_train <- training(initial_split)  

news_test <- testing(initial_split)

 news2 <- news1
 totalCol <- ncol(news2)
 
 trainM <- as.matrix(news_train)
 dimnames(trainM) <- NULL

 testM <- as.matrix(news_test)
 dimnames(testM) <- NULL
 virality_train <- as.numeric(trainM[,1])
 virality_test <- as.numeric(testM[,1])
 news_train <- normalize(trainM[,2:totalCol])
 news_test <- normalize(testM[,2:totalCol])
 
# one hot encoding
  trainLabels <- to_categorical(virality_train)
  testLabels <- to_categorical(virality_test)

#load the trained model
neurnetMV <- load_model_hdf5("neurnetUNCV.hdf5")

neurnetMV %>%
  evaluate(news_test, testLabels)

# for probabilities
probabilities <- neurnetMV %>%
  predict_proba(news_test)

# confusion matrix
pred <- neurnetMV %>%
  predict_classes(news_test)

pred <- as.vector(pred)


# --------------- hard voting ---------------

#combining classes predictions


#adding all y_pred together
#the code print the first vector as 1 and 2, so I put a -1 with to have the correct result
pred_df <- as.data.frame(cbind(as.numeric(pred_down)-1, as.numeric(pred_t), as.numeric(pred)))
#majority vote with mutate

pred_df <- pred_df %>% 
  mutate(sum = rowSums(.[1:3]))

#Weighted sum with more weight to the random forest, because it was the best model by far
pred_dfW <- pred_df %>% 
  mutate(sum = 0.825*V1 + 1.35*V2 + 0.825*V3)

#Set the threshold at 2, since it is the majority
thres <- 2
thres <- as.numeric(thres)

# Do the combined prediction

pred_dfW <- pred_dfW %>% 
  mutate(majority_vote = ifelse(as.numeric(pred_dfW$sum) >= thres, 1, 0))%>% 
  dplyr::select(-V1,-V2, -V3, -sum)

#confusion matrixof the combined prediction
a <- as.factor(pred_dfW$majority_vote)
b <- as.factor(news_test$virality)
CM <- confusionMatrix(data = a, reference = b,mode = "prec_recall", positive = "1")
CM 

# save
# save(pred_dfW, file = "pred_dfW.RData")

#plot ROC and calculate AUC
library(ROCR)
ROCR= prediction(pred_dfW$majority_vote, news_test$virality)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve Neural Networks", print.auc = T)
abline(a=0, b=1)

#compute area under the curve

auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AUC")

#       0     1
# 0 10287   538
# 1   397   670

# accuracy = 0.9214          
# precision = 0.62793         
# recall = 0.55464         
# balanced accuracy = 0.75874         

# AUC = 0.7587




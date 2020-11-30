
# --------------- CROSS-VALIDATION OF NEURAL NETWORKS ---------------

library(keras) #package needed for the neural networks
library(tfdatasets)
library(dplyr)
library(e1071)
library(caret)
library(ROSE)
library(rsample)
library(tidyverse)
# remove global environment and grapghs
rm(list = ls())
graphics.off()

set.seed(123)

# load the data
news0 <- read.table("OnlineNewsPopularity.csv", header = TRUE, sep = ",")


# classify the data between viral and non-viral
# where 1 stands for viral and 0 for non-viral

quantile_viral <- quantile(news0$shares, probs = 0.9)
quantile_viral <- as.numeric(quantile_viral)

# remove url, timedelta, shares which are not needed
news1 <- news0 %>% 
  mutate(virality = ifelse(as.numeric(news0$shares) >= quantile_viral, 1, 0)) %>% 
      select(-url, -timedelta, -shares) 
table(news1$virality) 

# have to number of column so that you don't have to count
# manually later
totalCol <- ncol(news1)


# --------------- 3: use cross-validation on undersampled dataset ---------------

# partition
set.seed(123)
perc_train <- 0.7  

initial_split <- initial_split(news1, prop = perc_train, strata = news1$virality)  
train <- training(initial_split)  
test <- testing(initial_split)

# For cross-validation, we will do a k-fold cross validation
fold <- createFolds(train$virality, 10)

# In this part we will create neural networks with the
# keras package, which include tenserflow
# create an empty structure for the network
neurnetUNCV <- keras_model_sequential()

# define the structure of the nn, defining the number of hidden layers
# and the activation function
neurnetUNCV %>%
  layer_dense(units =  256, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>% #avoid overfitting 
  layer_dense(units = 32, activation = 'relu', input_shape = c(totalCol-1))%>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetUNCV)


# compile characteristics of the learning algorithm
neurnetUNCV %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))


#do a for loop for cross validation
for (i in 1:length(fold)){
  
  # print the current iteration
  print(i)
  
  
  #create train and test sample with the fold
  kftrain <- train[-fold[[i]],]
  kfvalidation <- train[fold[[i]],]
  
  #resample the data, so that it contains an equal proportion of the two classes
  kftrain$virality = as.factor(kftrain$virality)
  kftrain_down <- downSample(x = kftrain,
                          y = kftrain$virality)
  kftrain_down <- kftrain_down[,1:59]
  
  kftrain_down$virality = as.numeric(kftrain_down$virality)
  kftrain_down$virality = kftrain_down$virality - 1
  
  # trasnform to a matrix because algorithm works with 
  #matrices
  trainM <- as.matrix(kftrain_down)
  dimnames(trainM) <- NULL
  
  #shuffle the data so that, the algorithm does not learn all 
  # the non-viral first and then all the viral ones
  rows <- sample(nrow(trainM))
  kftrain <- trainM[rows, ]
  
  #transform to a matrix for the same reason than
  # for train
  kfValidationM <- as.matrix(kfvalidation)
  dimnames(kfValidationM) <- NULL
  
  #noramlize the data so that the values are contained
  # in the the same interval, and that a variable is not
  # is not considered more than others just because values are 
  #higheer
  x_train <- normalize(kftrain[,1:totalCol-1])
  y_train <- as.numeric(kftrain[,totalCol])
  x_test <- normalize(kfValidationM[,1:totalCol-1])
  y_test <- as.numeric(kfValidationM[,totalCol])
  
  # need to transform target variable to categorical variable
  # in the model
  trainLabelsFold <- to_categorical(y_train)
  testLabelsFold <- to_categorical(y_test)
  
  
  #fit the model
  history <- neurnetUNCV %>%
    fit(x_train, 
        trainLabelsFold, 
        epoch = 100, 
        batch_size = 32,
        validation_split = 0.2)
  
  # evaluate
  neurnetUNCV %>%
    evaluate(x_test, testLabelsFold)
  
  summary(neurnetUNCV)
}

# save_model_hdf5(neurnetUNCV, file = "neurnetUNCV.hdf5")

# TESTING
testM <- as.matrix(test)
dimnames(testM) <- NULL
x_testD <- normalize(testM[,1:totalCol-1])
y_testD <- as.numeric(testM[,totalCol])
testLabels <- to_categorical(y_testD)

neurnetUNCV %>%
  evaluate(x_testD, testLabels)

# predict probabilities
probabilities <- neurnetUNCV %>%
  predict_proba(x_testD)

# predit the classes
pred <- neurnetUNCV %>%
  predict_classes(x_testD)

# confusion matrix
a <- as.factor(pred)
b <- as.factor(y_testD)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM 

#      0    1
# 0 7313  516
# 1 3352  710

# accuracy = 0.6747          
# precision = 0.17479         
# recall = 0.57912         
# balanced accuracy = 0.63241         

#AUC = 0.6796
                              

#plot ROC

library(ROCR)
probabilities1 <- probabilities[, -1]
ROCR <- prediction(probabilities1, y_testD)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve Neural Networks", print.auc = T)
abline(a=0, b=1)

#compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AUC")


# --------------- 2: use cross-validation on ROSE dataset ---------------

# partition
set.seed(123)
perc_train <- 0.7  

initial_split = initial_split(news1, prop = perc_train, strata = news1$virality)  
train = training(initial_split)  
test = testing(initial_split)

# For cross-validation, we will do a k-fold cross validation
fold <- createFolds(train$virality, 10)

# In this part we will create neural networks with the
# keras package, which include tenserflow
# create an empty structure for the network
neurnetBOTHCV1 <- keras_model_sequential()

#####
#First try overfitting
# neurnetBOTHCV %>%
#   layer_dense(units = 256, activation = 'relu', input_shape = c(totalCol-1)) %>%
#   layer_dropout(rate = 0.2)%>% #avoid overfitting 
#   layer_dense(units = 128, activation = 'relu')%>%
#   layer_dropout(rate = 0.2)%>%
#   layer_dense(units = 2, activation = 'softmax')
# summary(neurnetBOTHCV)


#SECOND TRY

# define the structure of the nn, defining the number of hidden layers
# and the activation function
neurnetBOTHCV1 %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>% #avoid overfitting 
  layer_dense(units = 64, activation = 'relu')%>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetBOTHCV1)

#compile characteristics of the learning algorithm
neurnetBOTHCV1 %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

# create vector and matrix to fill it
# with the result

#do a for loop for cross validation
for (i in 1:length(fold)){
  
  # print the current iteration
  print(i)
  
  #create train and validation sample with the fold
  kftrain <- train[-fold[[i]],]
  kfvalidation <- train[fold[[i]],]
  
  #resample the data, so that it contains an equal proportion of the two classes
  kfovun_news_train2 <- ovun.sample(virality ~ ., data = kftrain, method = "both") 
  kfovun_news_train2 <- kfovun_news_train2$data 
  table(kfovun_news_train2$virality) 
  kftrain = kfovun_news_train2
  
  # trasnform to a matrix because algorithm works with 
  #matrices
  trainM <- as.matrix(kftrain)
  dimnames(trainM) <- NULL
  
  #shuffle the data so that, the algorithm does not learn all 
  # the non-viral first and then all the viral ones
  rows <- sample(nrow(trainM))
  kftrain <- trainM[rows, ]
  
  #transform to a matrix for the same reason than
  # for train
  kfValidationM <- as.matrix(kfvalidation)
  dimnames(kfValidationM) <- NULL
  
  # noramlize the data so that the values are contained
  # in the the same interval, and that a variable is not
  # is not considered more than others just because values are 
  #higheer
  x_train <- normalize(kftrain[,1:totalCol-1])
  y_train <- as.numeric(kftrain[,totalCol])
  x_test <- normalize(kfValidationM[,1:totalCol-1])
  y_test <- as.numeric(kfValidationM[,totalCol])
  
  # need to transform target variable to categorical variable
  # in the model
  trainLabelsFold <- to_categorical(y_train)
  testLabelsFold <- to_categorical(y_test)
  
  
  #fit the model
  history <- neurnetBOTHCV1 %>%
    fit(x_train, 
        trainLabelsFold, 
        epoch = 100, 
        batch_size = 100,
        validation_split = 0.2)
  
  
  neurnetBOTHCV1 %>%
    evaluate(x_test, testLabelsFold)
}
  
#T ESTING
testM <- as.matrix(test)
dimnames(testM) <- NULL
x_testD <- normalize(testM[,1:totalCol-1])
y_testD <- as.numeric(testM[,totalCol])
testLabels <- to_categorical(y_testD)

neurnetBOTHCV1 %>%
  evaluate(x_testD, testLabels)

# predict probabilities
probabilities = neurnetBOTHCV1 %>%
  predict_proba(x_testD)

# predit the classes
pred <- neurnetBOTHCV1 %>%
  predict_classes(x_testD)


serializedNeurNetBOTHCV1 = serialize_model(neurnetBOTHCV1, include_optimizer = TRUE)
# save(serializedNeurNetBOTHCV1, file="serializedNeurNetBOTHCV1.RData")


# confusion matrix
a <- as.factor(pred)
b <- as.factor(y_testD)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM

library(ROCR)
probabilities1 = probabilities[, -1]
ROCR <- prediction(probabilities1, y_testD)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve Neural Networks", print.auc = T)
abline(a=0, b=1)

# compute area under the curve

auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AUC")

# saved trial:
# 0    1
# 0 7275  522
# 1 3390  704

# accuracy = 0.671          
# precision = 0.1720         
# recall = 0.5742          
# balanced accuracy = 0.6282         

#AUC = 0.6769


# --------------- 3: use cross-validation on undersampled dataset ---------------

# partition
set.seed(123)
perc_train <- 0.7  

initial_split <- initial_split(news1, prop = perc_train, strata = news1$virality)  
train <- training(initial_split)  
test <- testing(initial_split)

# For cross-validation, we will do a k-fold cross validation
fold <- createFolds(train$virality, 10)

#I n this part we will create neural networks with the
# keras package, which include tenserflow
# create an empty structure for the network
neurnetUPCV1 <- keras_model_sequential()

# define the structure of the nn, defining the number of hidden layers
# and the activation function
neurnetUPCV1 %>%
  layer_dense(units =  1024, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>% #avoid overfitting 
  layer_dense(units = 512, activation = 'relu')%>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetUPCV1)

# compile characteristics of the learning algorithm
neurnetUPCV1 %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

# create vector and matrix to fill it
# with the result

# do a for loop for cross validation
for (i in 1:length(fold)){
  
  # print the current iteration
  print(i)
  
  #create train and test sample with the fold
  kftrain <- train[-fold[[i]],]
  kfvalidation <- train[fold[[i]],]
  
  #resample the data, so that it contains an equal proportion of the two classes
  kftrain$virality = as.factor(kftrain$virality)
  upTrain <- upSample(x=kftrain[(-c(59))], y=kftrain$virality)
  #take as features all x values, but virality (-c(1)
  upTrain$virality <- upTrain$Class
  #Rename Class into virality
  upTrain <- upTrain %>% select(-c(59))#Drop Class
  table(upTrain$virality)
  
  upTrain$virality = as.numeric(upTrain$virality)
  kftrain = upTrain
  kftrain$virality = kftrain$virality - 1
  
  # trasnform to a matrix because algorithm works with 
  #matrices
  trainM <- as.matrix(kftrain)
  dimnames(trainM) <- NULL
  
  #shuffle the data so that, the algorithm does not learn all 
  # the non-viral first and then all the viral ones
  rows <- sample(nrow(trainM))
  kftrain <- trainM[rows, ]
  
  #transform to a matrix for the same reason than
  # for train
  kfValidationM <- as.matrix(kfvalidation)
  dimnames(kfValidationM) <- NULL
  
  
  #noramlize the data so that the values are contained
  # in the the same interval, and that a variable is not
  # is not considered more than others just because values are 
  #higheer
  x_train <- normalize(kftrain[,1:totalCol-1])
  y_train <- as.numeric(kftrain[,totalCol])
  x_test <- normalize(kfValidationM[,1:totalCol-1])
  y_test <- as.numeric(kfValidationM[,totalCol])
  
  # need to transform target variable to categorical variable
  # in the model
  trainLabelsFold <- to_categorical(y_train)
  testLabelsFold <- to_categorical(y_test)
  
  #fit the model
  history <- neurnetUPCV1 %>%
    fit(x_train, 
        trainLabelsFold, 
        epoch = 150, 
        batch_size = 512,
        validation_split = 0.2)
  
  #evaluate
  neurnetUPCV1 %>%
    evaluate(x_test, testLabelsFold)
  
}

# save_model_hdf5(neurnetUPCV1, file="neurnetUPCV1.hdf5")
  
# TESTING
testM <- as.matrix(test)
dimnames(testM) <- NULL
x_testD <- normalize(testM[,1:totalCol-1])
y_testD <- as.numeric(testM[,totalCol])
testLabels <- to_categorical(y_testD)

neurnetUPCV1 %>%
  evaluate(x_testD, testLabels)

# predict probabilities
probabilities = neurnetUPCV1 %>%
  predict_proba(x_testD)

# predit the classes
pred <- neurnetUPCV1 %>%
  predict_classes(x_testD)

# confusion matrix

a <- as.factor(pred)
b <- as.factor(y_testD)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM

# plot ROC
library(ROCR)
probabilities1 <- probabilities[, -1]
ROCR <- prediction(probabilities1, y_testD)
perf <-  performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve Neural Networks", print.auc = T)
abline(a=0, b=1)

# compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AUC")

#      0    1
# 0 8074  715
# 1 2591  511

# accuracy = 0.722         
# precision = 0.16473       
# recall = 0.41680   
# balanced accuracy = 0.58693       

# AUC = 0.6198
                                        


# --------------- 5: use cross-validation on SMOTE dataset ---------------


#partition
set.seed(123)
perc_train <- 0.7  

initial_split <- initial_split(news1, prop = perc_train, strata = news1$virality)  
train <- training(initial_split)  
test <- testing(initial_split)

#For cross-validation, we will do a k-fold cross validation
fold <- createFolds(train$virality, 10)


library(DMwR)

neurnetSMCV1 <- keras_model_sequential()

# First try
# neurnetSMCV %>%
#   layer_dense(units = 256, activation = 'relu', input_shape = c(totalCol-1)) %>%
#   layer_dropout(rate = 0.2)%>%
#   layer_dense(units = 128, activation = 'relu') %>%
#   layer_dropout(rate = 0.2)%>%
#   layer_dense(units = 2, activation = 'softmax')
# summary(neurnetSMCV)


# Second try

# define the structure of the nn, defining the number of hidden layers
# and the activation function
neurnetSMCV1 %>%
  layer_dense(units =  128, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>% #avoid overfitting 
  layer_dense(units = 64, activation = 'relu')%>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetSMCV1)


#compile characteristics of the learning algorithm
neurnetSMCV1 %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

# create vector and matrix to fill it
# with the result

#do a for loop for cross validation
for (i in 1:length(fold)){
  
  # print the current iteration
  print(i)
  
  
  #create train and test sample with the fold
  kftrain <- train[-fold[[i]],]
  kfvalidation <- train[fold[[i]],]
  
  #resample the data, so that it contains an equal proportion of the two classes
  kftrain$virality = as.factor(kftrain$virality)
  kfsmote_Train <- SMOTE(virality ~ ., data= kftrain,
                       dup_size =length(which(kftrain$virality==0))/length(which(kftrain$virality ==1))
  )
  
  kfsmote_Train$virality = as.numeric(kfsmote_Train$virality)
  kftrain = kfsmote_Train
  kftrain$virality = kftrain$virality - 1
  
  # trasnform to a matrix because algorithm works with 
  #matrices
  trainM <- as.matrix(kftrain)
  dimnames(trainM) <- NULL
  
  # shuffle the data so that, the algorithm does not learn all 
  # the non-viral first and then all the viral ones
  rows <- sample(nrow(trainM))
  kftrain <- trainM[rows, ]
  
  # transform to a matrix for the same reason than
  # for train
  kfValidationM <- as.matrix(kfvalidation)
  dimnames(kfValidationM) <- NULL
  
  # noramlize the data so that the values are contained
  # in the the same interval, and that a variable is not
  # is not considered more than others just because values are higher
  x_train <- normalize(kftrain[,1:totalCol-1])
  y_train <- as.numeric(kftrain[,totalCol])
  x_test <- normalize(kfValidationM[,1:totalCol-1])
  y_test <- as.numeric(kfValidationM[,totalCol])
  
  # need to transform target variable to categorical variable
  # in the model
  trainLabelsFold <- to_categorical(y_train)
  testLabelsFold <- to_categorical(y_test)
  
  #fit the model
  history <- neurnetSMCV1 %>%
    fit(x_train, 
        trainLabelsFold, 
        epoch = 150, 
        batch_size = 128,
        validation_split = 0.2)
  
  
  neurnetSMCV1 %>%
    evaluate(x_test, testLabelsFold)
  
}

#TESTING
testM <- as.matrix(test)
dimnames(testM) <- NULL
x_testD <- normalize(testM[,1:totalCol-1])
y_testD <- as.numeric(testM[,totalCol])
testLabels <- to_categorical(y_testD)

# save_model_hdf5(neurnetSMCV1, file="neurNetSMCV1.hdf5")
myLoad = load_model_hdf5(file="neurNetSMCV1.hdf5")

neurnetSMCV1 %>%
  evaluate(x_testD, testLabels)

# predict probabilities
probabilities = neurnetSMCV1 %>%
  predict_proba(x_testD)

# predit the classes
pred <- neurnetSMCV1 %>%
  predict_classes(x_testD)

# confusion matrix
a <- as.factor(pred)
b <- as.factor(y_testD)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM


probabilities1 <- probabilities[, -1]
ROCR <- prediction(probabilities1, y_testD)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve Neural Networks", print.auc = T)
abline(a=0, b=1)

#compute area under the curve
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc = round(auc, 4)
legend(.6, .2, auc, title = "AUC")

#      0    1
# 0 7925  629
# 1 2740  597

# accuracy = 0.7167          
# precision = 0.17890         
# recall = 0.48695         
# balanced accuracy = 0.61502         

# AUC = 6586


#####
#Second try less overfitting
#######

#      0    1
# 0 8246  671
# 1 2419  555

# accuracy = 0.7401         
# precision = 0.18662        
# recall = 0.45269        
# balanced accuracy = 0.61294        

# AUC = 0.666
                                             

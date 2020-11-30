
# --------------- Artificial Neural Networks ---------------

library(keras)
library(tfdatasets)
library(dplyr)
library(e1071)
library(caret)
library(ROSE)
library(rsample)
library(tidyverse)
library(tfruns)


# remove global environment and graphs
rm(list = ls())
graphics.off()

set.seed(123)

# load the data
news0 <- read.table("OnlineNewsPopularity.csv", header = TRUE, sep = ",")


# classify the data between viral and non-viral
# where 1 stands for viral and 0 for non-viral

quantile_viral <- quantile(news0$shares, probs = 0.9)
quantile_viral <- as.numeric(quantile_viral)

# remove columns that are not needed
news1 <- news0 %>% 
  mutate(virality = ifelse(as.numeric(news0$shares) >= quantile_viral, 1, 0)) %>% 
  dplyr::select(-url, -timedelta, -shares) 
table(news1$virality) 


# --------------- 1: use neural network on original dataset ---------------

# shuffle the data
rows <- sample(nrow(news1))
news1 <- news1[rows, ]

# change to matrix
news2 <- as.matrix(news1)
dimnames(news2) <- NULL

totalCol <- ncol(news2)

# normalize
news2[,totalCol] <- as.numeric(news2[,totalCol]) # for virality
news2[,1:totalCol-1] <- normalize(news2[,1:totalCol-1]) # for parameters
# summary(news2)

# partition
ind <- sample(2, totalCol, replace = T, prob = c(0.7, 0.3))
news_train <- news2[ind==1, 1:totalCol-1]
news_test <- news2[ind==2, 1:totalCol-1]
virality_train <- news2[ind==1, totalCol]
virality_test <- news2[ind==2, totalCol]

# one hot encoding
trainLabels <- to_categorical(virality_train)
testLabels <- to_categorical(virality_test)

# create model
neurnet <- keras_model_sequential()

# add layers and compile the model
neurnet %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnet)

neurnet %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# fit model
history <- neurnet %>%
  fit(news_train, 
      trainLabels, 
      epoch = 100, 
      batch_size = 32,
      validation_split = 0.2)

# evaluate
neurnet %>%
  evaluate(news_test, testLabels)

# for probabilities
probabilities <- neurnet %>%
  predict_proba(news_test)

# predict classes
pred <- neurnet %>%
  predict_classes(news_test)

# confusion matrix
a <- as.factor(pred)
b <- as.factor(virality_test)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM 

# save_model_hdf5(neurnet, file="neurnet.hdf5")

#      0    1
# 0 9642 1101
# 1    3    4

# accuracy = 0.8973         
# precision = 0.5714286      
# recall = 0.0036199      
# balanced accuracy = 0.5016544      

# accuracy seems good because it classifies everything as non-viral, what we do not want

# plot ROC and add AUC
library(ROCR)
probabilities1 <- probabilities[, -1]
ROCR= prediction(probabilities1, virality_test)
perf= performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for neural networks", print.auc = T)
abline(a=0, b=1)
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AUC")

# AUC = 0.6973

#TRY WITH RESAMPLED DATA SETS


# --------------- 2: use neural network on ROSE dataset ---------------

news1 <- news0 %>% 
  mutate(virality = ifelse(as.numeric(news0$shares) >= quantile_viral, 1, 0)) %>% 
  dplyr::select(-url, -timedelta, -shares) 
table(news1$virality) 


news2 <- news1
totalCol <- ncol(news2)

# partition
ind <- sample(2, totalCol, replace = T, prob = c(0.7, 0.3))
train <- news2[ind==1,]
test <- news2[ind==2, ]

# both oversampling and undersampling with package rose
ovun_news_train2 <- ovun.sample(virality ~ ., data = train, method = "both") 
ovun_news_train2 <- ovun_news_train2$data 
table(ovun_news_train2$virality) 
train <- ovun_news_train2


# transform as matrix
trainM <- as.matrix(train)
dimnames(trainM) <- NULL

# shuffle the data, so that it will not learn all the non-viral first
# and then all the viral ones
rows <- sample(nrow(trainM))
trainM <- trainM[rows, ]

# transform as matrix
testM <- as.matrix(test)
dimnames(testM) <- NULL

# normalize the data
virality_train <- as.numeric(trainM[,totalCol])
virality_test <- as.numeric(testM[,totalCol])
news_train <- normalize(trainM[,1:totalCol-1])
news_test <- normalize(testM[,1:totalCol-1])

# one hot encoding
trainLabels <- to_categorical(virality_train)
testLabels <- to_categorical(virality_test)

# Tuning parameters from ROSE dataset

#SEE "BuildModel" FILE

runs = tuning_run("BuildModel.R", flags = list(dense_units = c(32, 64, 128),
                                               dense_units2 =c(16,32,64),
                                               dropout1=c(0.1, 0.2),
                                               dropout2=c(0.1, 0.2),
                                               epoch = c(100, 150)))

head(runs)
results <- runs[, c(3, 5:11)]
#####  ACC    VALACC  
#     0.6532	0.6539	128	64	0.1	0.1	32	150
# 10	0.6550	0.6536	128	64	0.1	0.2	32	150
# 19	0.6517	0.6528	128	64	0.2	0.1	32	150
# 64	0.6496	0.6523	128	64	0.1	0.1	32	100
# 67	0.6467	0.6518	128	32	0.1	0.1	32	100
# 16	0.6466	0.6510	128	16	0.1	0.2	32	150
###
# We can see that the most important parameters are the numbers of hidden layers
# Accuracy and Validation accuracy values are close to each other
# which indicate that there is no specific overfitting problem

# save(runs,file = "runs.RData")

##### ADDED AUC
runs2 <- tuning_run("BuildModel.R", flags = list(dense_units = c(128, 256),
                                                dense_units2 =c(64, 128),
                                                dropout1=c(0.1, 0.2),
                                                dropout2=c(0.1, 0.2)))
# save(runs2,file = "runs2.RData")
head(runs2)
results <- runs2[, c(3:4, 6:37)]

#       ACC    VALACC    AUC    AUCVAL
#     0.6596		0.6610	0.7154	0.7205	256	128	0.1	0.2	
# 13	0.6589  	0.6549	0.7168	0.7209	256	128	0.1	0.1	
# 15	0.6583		0.6572	0.7143	0.716	  256	64	0.1	0.1	
# 8	  0.6559		0.6538	0.7096	0.7166	256	64	0.1	0.1
# the one with the highest accuracy were also the ones with the highest AUC

runstest <- tuning_run("BuildModel.R", flags = list(dense_units = c(128, 256),
                                                   dense_units2 =c(64, 128)))


# Create the chosen model

neurnetBOTH <- keras_model_sequential()

# add layers and compile the model

# First try
# neurnetBOTH %>%
#   layer_dense(units = 256, activation = 'relu', input_shape = c(totalCol-1)) %>%
#   layer_dropout(rate = 0.1)%>% #avoid overfitting 
#   layer_dense(units = 128, activation = 'relu', input_shape = c(totalCol-1))%>%
#   layer_dropout(rate = 0.1)%>%
#   layer_dense(units = 2, activation = 'softmax')
# summary(neurnetTune)
#The difference between the training and testing was too high which means that it is overfitting
#Accuracy : 0.5695 vs accuracy in training: 0.6607, so increase the dropout rate to 0.2, this paparameter will
# not be tried in parameter tuning

neurnetBOTH %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>% #avoid overfitting 
  layer_dense(units = 128, activation = 'relu', input_shape = c(totalCol-1))%>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetTune)

neurnetBOTH %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

# fit model
history <- neurnetBOTH %>%
  fit(news_train, 
      trainLabels, 
      epoch = 150, 
      batch_size = 100,
      validation_split = 0.2)

# evaluate
neurnetBOTH %>%
  evaluate(news_test, testLabels)

# for probabilities
probabilities <- neurnetBOTH %>%
  predict_proba(news_test)

# confusion matrix
pred <- neurnetBOTH %>%
  predict_classes(news_test)

a <- as.factor(pred)
b <- as.factor(virality_test)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM 

# plot ROC
library(ROCR)
probabilities1 = probabilities[, -1]
ROCR= prediction(probabilities1, virality_test)
perf= performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for neural networks", print.auc = T)
abline(a=0, b=1)
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc = round(auc, 4)
legend(.6, .2, auc, title = "AUC")

#      0    1
# 0 5827  391
# 1 2633  556

# accuracy = 0.6785        
# precision = 0.1743        
# recall = 0.5871        
# balanced accuracy = 0.6379

# AUC = 0.6832


#SEE THE FILE CVNeurnet TO SEE THE CROSS-VALIDATION


# --------------- 3: use neural network on undersampled dataset ---------------

news2 <- news1
totalCol <- ncol(news2)

# partition
ind <- sample(2, totalCol, replace = T, prob = c(0.7, 0.3))
train <- news2[ind==1,]
test <- news2[ind==2, ]

# downsample the data
train$virality = as.factor(train$virality)
news_down <- downSample(x = train,
                        y = train$virality)
news_down <- news_down[,1:59]

news_down$virality <- as.numeric(news_down$virality)
train <- news_down
train$virality <- train$virality - 1

# transform as matrix
trainM <- as.matrix(train)
dimnames(trainM) <- NULL

rows <- sample(nrow(trainM))
trainM <- trainM[rows, ]

testM <- as.matrix(test)
dimnames(testM) <- NULL

# normalize the data
virality_train <- as.numeric(trainM[,totalCol])
virality_test <- as.numeric(testM[,totalCol])
news_train <- normalize(trainM[,1:totalCol-1])
news_test <- normalize(testM[,1:totalCol-1])

# one hot encoding
trainLabels <- to_categorical(virality_train)
testLabels <- to_categorical(virality_test)

# TUNE parameters
# SEE "BuildModel" FILE

# Change epoch to 100 and dropout to 0.2 so that it will not overfit too much, batch_size 50
runsDOWN = tuning_run("BuildModel.R", flags = list(dense_units = c(32, 64, 128, 256),
                                                   dense_units2 =c(16,32,64,128)))
#ACC    VALACC   AUC    AUCVAL                     
#     0.6450	0.6243	0.6888	0.652	  256	32	
# 15	0.6414	0.6198	0.6824	0.6579	64	16	
# 5		0.6400	0.6134	0.6906	0.6606	256	64	
# 14	0.6400	0.6189	0.6807	0.661		128	16
## not a clear tendency

# save(runsDOWN,file = "runsDOWN.RData")


# Second try with batch size

runsDOWN1 = tuning_run("BuildModel.R", flags = list(dense_units = c(32, 64, 128),
                                                    dense_units2 =c(16,32,64),
                                                    batch_size = c(32, 64,128)))
head(runsDOWN1)
# save(runsDOWN1,file = "runsDOWN1.RData")
#       ACC    VALACC   AUC   AUCVAL        #batch_size
# 22  0.6428	0.6234	0.6882	0.6496	128 32  32  
# 13	0.6396	0.6162	0.6868	0.6662	128 32  64
# 23	0.6396	0.6252	0.685   0.6516  64  32  32 
# 19	0.6389	0.6225	0.6903	0.6595	128 64  32 
#the batch size is not so imprtant the smaller

runsDOWN2 = tuning_run("BuildModel.R", flags = list(dense_units = c(256),
                                                    dense_units2 =c(32,64,128),
                                                    batch_size = c(32, 64)))
#     ACC    VALACC   AUC   AUCVAL
#   0.6489	0.6243	0.6955	0.6673	256	64	32	
# 1	0.6480	0.6189	0.6994  0.664		256	128	64
# 2	0.6450	0.6016	0.6951	0.6336	256	64	64		
# 3	0.6421	0.6134	0.6906	0.6588	256	32	64		
# 6 0.6421	0.6180	0.6847	0.6623	256	32	32		
# 4	0.6344	0.6189	0.689	  0.6548  256	128	32

#We will retain the first rows of runsDown2 as the best parameters as it has the best performance so far, 
# even we were not able to increase it significantly

# save(runsDOWN2,file = "runsDOWN2.RData")

#############
#Build the actual model

# create model
neurnetUN <- keras_model_sequential()

# add layers and compile the model
neurnetUN %>%
  layer_dense(units =  256, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>% #avoid overfitting 
  layer_dense(units = 32, activation = 'relu')%>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetTune)

neurnetUN %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

# fit model
history <- neurnetUN %>%
  fit(news_train, 
      trainLabels, 
      epoch = 100, 
      batch_size = 32,
      validation_split = 0.2)

# evaluate
neurnetUN %>%
  evaluate(news_test, testLabels)

# for probabilities
probabilities = neurnetUN %>%
  predict_proba(news_test)


# predict classes
pred <- neurnetUN %>%
  predict_classes(news_test)

# confusion matrix
a <- as.factor(pred)
b <- as.factor(virality_test)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM


#      0    1
# 0 5977  372
# 1 4842  903

# accuracy = 0.5689        
# precision = 0.15718       
# recall = 0.70824       
# balanced accuracy = 0.63034       

# AUC = 0.679


# SEE THE OTHER FILE TO SEE CROSS-VALIDATION

library(ROCR)
probabilities1 = probabilities[, -1]
ROCR= prediction(probabilities1, virality_test)
perf= performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for neural networks", print.auc = T)
abline(a=0, b=1)
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AUC")


# --------------- 4: use neural network on oversampled dataset ---------------

news2 <- news1
totalCol <- ncol(news2)

# partition
ind <- sample(2, totalCol, replace = T, prob = c(0.7, 0.3))
train <- news2[ind==1,]
test <- news2[ind==2, ]

# oversample the data

# take as features all x values, but virality (-c(1)
train$virality <- as.factor(train$virality)
upTrain <- upSample(x=train[(-c(59))], y=train$virality)

# rename Class into virality
upTrain$virality <- upTrain$Class

upTrain <- upTrain %>% select(-c(59))#Drop Class
table(upTrain$virality)

upTrain$virality <- as.numeric(upTrain$virality)
train <- upTrain
train$virality <- train$virality - 1

# transform data as matrix
trainM <- as.matrix(train)
dimnames(trainM) <- NULL

rows <- sample(nrow(trainM))
trainM <- trainM[rows, ]

testM <- as.matrix(test)
dimnames(testM) <- NULL

# normalize the data
virality_train <- as.numeric(trainM[,totalCol])
virality_test <- as.numeric(testM[,totalCol])
news_train <- normalize(trainM[,1:totalCol-1])
news_test <- normalize(testM[,1:totalCol-1])

# one hot encoding
trainLabels <- to_categorical(virality_train)
testLabels <- to_categorical(virality_test)

######
#TUNE PARAMETERS
####

# SEE "BuildModel" FILE

runsUP <- tuning_run("BuildModel.R", flags = list(dense_units = c(32, 64),
                                                 dense_units2 =c(16, 32),
                                                 batch_size = c(128, 256)))

head(runsUP)
results1 = runsUP[c(3:27)]

####
# accuracy and in terms of AUC it is with the highest units of hidden layer at 
# the begining
#  ACC    VALACC   AUC   AUCVAL
# 0.6525 0.6540 0.7052 0.7099 64 32 128
# 0.6501 0.6491 0.6987 0.7063 32 16 128
# 0.6497 0.6526 0.7005 0.7099 64 16 128
# 0.6491 0.6564 0.7035 0.71   64 32 256

# save(runsUP, file = "runsUP.RData")

runsUP1 <- tuning_run("BuildModel.R", flags = list(dense_units = c(128, 256),
                                                  dense_units2 =c(64, 128),
                                                  batch_size = c(256, 512)))

results2 <- runsUP1[c(3:27)]
# save(runsUP1, file = "runsUP1.RData")

# ACC    VALACC   AUC   AUCVAL
# 0.6626 0.6622 0.7218 0.7223 256 128 256
# 0.6604 0.6582 0.7204 0.71   256 128 512
# 0.6591 0.6592 0.72   0.7229 256 64  256
# 0.6575 0.6533 0.7152 0.7155 256 64  512

# batch_size does not change much, set batch size 256

runsUP2 <- tuning_run("BuildModel.R", flags = list(dense_units = c(512),
                                                  dense_units2 =c(128,256)))

results3 <- runsUP2[c(3:27)]
# save(runsUP2, file = "runsUP2.RData")

# ACC    VALACC  AUC   AUCVAL
# 0.6652 0.6647 0.7282 0.7281 512 256
# 0.6628 0.6555 0.7248 0.7253 512 128

runsUP3 <- tuning_run("BuildModel.R", flags = list(dense_units = c(1024),
                                                  dense_units2 =c(512)))
# save(runsUP3, file = "runsUP3.RData")
# 0.6691 0.6616 0.7336 0.734 1024 512


# We will keep the last try

#############
#Build the actual model

# create model
neurnetUP <- keras_model_sequential()

# add layers and compile the model
neurnetUP %>%
  layer_dense(units =  1024, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>% #avoid overfitting 
  layer_dense(units = 512, activation = 'relu')%>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetUP)


neurnetUP %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

# fit model
history <- neurnetUP %>%
  fit(news_train, 
      trainLabels, 
      epoch = 150, 
      batch_size = 256,
      validation_split = 0.2)


# evaluate
neurnetUP %>%
  evaluate(news_test, testLabels)

# for probabilities
probabilities <- neurnetUP %>%
  predict_proba(news_test)


# confusion matrix
pred <- neurnetUP %>%
  predict_classes(news_test)

a <- as.factor(pred)
b <- as.factor(virality_test)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM 

#      0    1
# 0 7461  469
# 1 4639  869

# accuracy = 0.6199          
# precision = 0.15777         
# recall = 0.64948         
# balanced accuracy = 0.63304

# AUC = 0.6717


# plot ROC and calculate AUC
library(ROCR)
probabilities1 <- probabilities[, -1]
ROCR <- prediction(probabilities1, virality_test)
perf <- performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for neural networks", print.auc = T)
abline(a=0, b=1)
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc, 4)
legend(.6, .2, auc, title = "AUC")


# SEE THE OTHER FILE TO SEE THE CROSS-VALIDATED MODEL


# --------------- 5: use neural network on SMOTE dataset ---------------

news2 <- news1
totalCol <- ncol(news2)

# partition
ind <- sample(2, totalCol, replace = T, prob = c(0.7, 0.3))
train <- news2[ind==1,]
test <- news2[ind==2, ]

# SMOTE technique to resample the data
library(DMwR)
train$virality = as.factor(train$virality)
smote_Train <- SMOTE(virality ~ ., data= train,
                     dup_size =length(which(train$virality==0))/length(which(train$virality ==1))
)

smote_Train$virality <- as.numeric(smote_Train$virality)
train <- smote_Train
train$virality <- train$virality - 1

# transform as matrix
trainM <- as.matrix(train)
dimnames(trainM) <- NULL

# shuffle the data
rows <- sample(nrow(trainM))
trainM <- trainM[rows, ]

# transform as matrix
testM <- as.matrix(test)
dimnames(testM) <- NULL

# normalize the data
virality_train <- as.numeric(trainM[,totalCol])
virality_test <- as.numeric(testM[,totalCol])
news_train <- normalize(trainM[,1:totalCol-1])
news_test <- normalize(testM[,1:totalCol-1])

# one hot encoding
trainLabels <- to_categorical(virality_train)
testLabels <- to_categorical(virality_test)

############################
# Tuning  parameters
########################
#SEE "BuildModel.R" FILE

runsSmote <- tuning_run("BuildModel.R", flags = list(dense_units = c(32, 64, 128),
                                                    dense_units2 =c(16,32,64),
                                                    dropout1=c(0.1, 0.2),
                                                    dropout2=c(0.1, 0.2),
                                                    epoch = c(100, 150)))
head(runsSmote)
results <- runsSmote[c(3,5:11)]
    #ACC VALACC  
# 0.6846 0.6662 128 64 0.1 0.2 150
# 0.6831 0.6621 128 32 0.1 0.2 150
# 0.6825 0.6678 128 64 0.1 0.1 150
# 0.6817 0.6618 128 16 0.1 0.1 150
# As we can see the number of units in the hidden layers is the most important

# save(runsSmote, file = "runsSmote.RData")
# We will keep the first line and we will not try higher parameters because of the risk of overfiting

######
# Build the actual model

# create model
neurnetSM1 <- keras_model_sequential()

# add layers and compile the model
neurnetSM %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetSM)

# second try
neurnetSM1 %>%
layer_dense(units = 256, activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetSM1)


neurnetSM1 %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

# fit model
history <- neurnetSM1 %>%
  fit(news_train, 
      trainLabels, 
      epoch = 150, 
      batch_size = 128,
      validation_split = 0.2)

# evaluate
neurnetSM1 %>%
  evaluate(news_test, testLabels)

# for probabilities
probabilities <- neurnetSM1 %>%
  predict_proba(news_test)

# predict classes
pred <- neurnetSM1 %>%
  predict_classes(news_test)

# confusion matrix
a <- as.factor(pred)
b <- as.factor(virality_test)
CM <- confusionMatrix(data = a, reference = b, mode = "prec_recall", positive = "1")
CM 

# plot ROC and calculate AUC
library(ROCR)
probabilities1 = probabilities[, -1]
ROCR= prediction(probabilities1, virality_test)
perf= performance(ROCR,"tpr", "fpr")
plot(perf, main = "ROC Curve for neural networks", print.auc = T)
abline(a=0, b=1)
auc <- performance(ROCR, measure = "auc")
auc <- unlist(slot(auc,"y.values"))
auc = round(auc, 4)
legend(.6, .2, auc, title = "AUC")

# first try

#      0    1
# 0 8101  633
# 1 2718  642

# accuracy = 0.7229          
# precision = 0.19107         
# recall = 0.50353         
# balanced accuracy = 0.62615         

# AUC = 0.6782


# second try

#      0    1
# 0 8537  700
# 1 2282  575

# accuracy = 0.7534          
# precision = 0.20126         
# recall = 0.45098         
# balanced accuracy = 0.62003         

# AUC = 0.6743

# We will keep the second parameters to do cross validation


#SEE THE OTHER FILE TO SEE THE CROSS-VALIDATED MODEL




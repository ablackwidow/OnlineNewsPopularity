
# --------------- preparing the work-surface  ---------------

# install.packages("neuralnet")
# install.packages("ROSE")
# install.packages("pROC")
# install.packages("DMwR")
# install.packages("tidymodels", repo = 'https://mac.R-project.org')
# install.packages("caret")


# load packages 
library(neuralnet) 
library(tidyverse) 
library(caret) 
library(rsample) 
library(randomForest)
library(ranger)
library(ROSE)
library(pROC)
library(DMwR)
library(tidymodels)
library(ROCR)
library(e1071)

# remove global environment and grapghs 
rm(list = ls()) 
graphics.off() 


# --------------- method 1: create dataset using original data ---------------

# data preprocessing 

# load data
news0 <- read.table("OnlineNewsPopularity.csv", header = TRUE, sep = ",") 

# classify the data between viral and non-viral 
quantile_viral <- quantile(news0$shares, probs = 0.9) 
quantile_viral <-  as.numeric(quantile_viral)

#add column for virality (1 = viral, 0 = non-viral) and drop url, shares and timedelta
news1  <-  news0 %>%
  mutate(virality = ifelse(as.numeric(news0$shares) >= quantile_viral, 1, 0)) %>%
  select(-url, -timedelta, -shares)

table(news1$virality)

#justification to take out timedelta
ggplot()+
  geom_point(mapping = aes(x = news0$timedelta, y = news0$shares))


# make categorical variables factors
news1$data_channel_is_bus <- as.factor(news1$data_channel_is_bus)
news1$data_channel_is_entertainment <- as.factor(news1$data_channel_is_entertainment)
news1$data_channel_is_lifestyle <- as.factor(news1$data_channel_is_lifestyle)
news1$data_channel_is_socmed <- as.factor(news1$data_channel_is_socmed)
news1$data_channel_is_tech <- as.factor(news1$data_channel_is_tech)
news1$data_channel_is_world <- as.factor(news1$data_channel_is_world)
news1$weekday_is_monday <- as.factor(news1$weekday_is_monday)
news1$weekday_is_tuesday <- as.factor(news1$weekday_is_tuesday)
news1$weekday_is_wednesday <- as.factor(news1$weekday_is_wednesday)
news1$weekday_is_thursday <- as.factor(news1$weekday_is_thursday)
news1$weekday_is_friday <- as.factor(news1$weekday_is_friday)
news1$weekday_is_saturday <- as.factor(news1$weekday_is_saturday)
news1$weekday_is_sunday <- as.factor(news1$weekday_is_sunday)
news1$is_weekend <- as.factor(news1$is_weekend)


#create testing and training sample 
set.seed(123) 
perc_train <- 0.7 

initial_split = initial_split(news1, prop = perc_train, strata = news1$virality) 
news_train = training(initial_split) 
news_test = testing(initial_split) 

#change virality to be a factor
news_train$virality = as.factor(news_train$virality)
news_test$virality = as.factor(news_test$virality)

# look for more or less equal distribution of virality and non-virality in both samples.  
percentage_news_train <- table(news_train$virality) / sum(table(news_train$virality)) * 100 
print(percentage_news_train) 

percentage_news_test <- table(news_test$virality) / sum(table(news_test$virality)) * 100 
print(percentage_news_test) 


# --------------- method 2: create balanced dataset using ROSE ---------------

# using the ROSE package we use a combination of over- and under-sampling 

# to get a more even training set
ovun_news_train <- ovun.sample(virality ~ ., data = news_train, method = "both")
ovun_news_train <- ovun_news_train$data

table(ovun_news_train$virality)

# reshuffle columns
rows <- sample(nrow(ovun_news_train))
ovun_news_train <- ovun_news_train[rows, ]

#change categorical variables to factors
ovun_news_train$virality <- as.factor(ovun_news_train$virality)

# check if dataset is more or less balanced
table(ovun_news_train$virality)


# --------------- method 3: create balanced dataset using undersampling ---------------

# create downsampled dataset 
news_down <- downSample(x = news_train,
                        y = news_train$virality)

# reshuffle columns and delete column "class"
set.seed(1)

news_down <- news_down[sample(1:nrow(news_down)), ]  
news_down <- news_down %>%
  select(-Class)

# check if dataset is balanced
table(news_down$virality)


# --------------- method 4: create balanced dataset using oversampling ---------------

# create oversampled dataset and duplicate column "class" as column "virality"
upTrain <- upSample(x=news_train[,-c(59)], y=news_train$virality)
upTrain$virality <- upTrain$Class

# drop column "class"
upTrain <- upTrain %>% select(-c(59))

# take target (column "virality") as first column
up_Train <- upTrain %>% select(virality,everything())

# reshuffle columns
up_Train <- upTrain [sample(1:nrow(upTrain)), ]

# check if dataset is balanced
table(up_Train$virality)


# --------------- method 5: create balanced dataset using SMOTE ---------------

# using SMOTE to balance the data set by artificially generating data
library(DMwR)
smote_Train <- SMOTE(virality ~ ., data=news_train,
                     dup_size =length(which(news_train$virality==0))/length(which(news_train$virality ==1))
)

# reshuffle columns
smote_Train <- smote_Train[sample(1:nrow(smote_Train)), ]

# check if dataset is more or less balanced
table(smote_Train$virality)


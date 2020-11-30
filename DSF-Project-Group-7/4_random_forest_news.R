
# --------------- RANDOM FOREST ---------------

library(tidyverse)
library(tidymodels)
library(randomForest)
library(ranger)
library(pROC)
library(ROCR)
library(caret)


# --------------- 1: use random forest on original dataset ---------------

# we start by training a simple random forest to get an idea how it performs

set.seed(123)
rf_original <- randomForest(
  formula = virality ~ .,
  data    = news_train
)

rf_original

plot(rf_original)

# from the plot we can see that the out-of-bag (OOB) error for the positive class (green),
# negative class (red) and overall (black) for different number of trees. As we can
# see the OOB error rate for the positive class increases with increasing number of trees
# which means that the simple model does not classify well even in training

# predicting
test_rf_o <- predict(rf_original, news_test)

#confusion matrix
cm_o <- confusionMatrix(test_rf_o, news_test$virality,
                              mode = "prec_recall", positive = "1")

#       0     1
# 0 10656  1219
# 1     9     7

# precision = 0.4375
# recall = 0.0057
# balanced accuracy = 0.5024

# ROC Curve
test_rf_o <- as.numeric(test_rf_o)
roc_og <- roc(news_test$virality, test_rf_o, plot=T)

auc_roc(test_rf_o, news_test$virality)

# AUC = 0.5017

# The AUC is hardly above 0.5 which is very bad and only marginally better than
# misclassifying all viral articles

# to improve the performance, we tune the hyperparameters mtry and min_n
# the random forest is unable to determine the optimal values for the hyper-
# parameters itself, therefore we try different combinations to maximise the AUC
# (we do not try to maximise accuracy as this would lead to misclassifying the viral
# articles)

# write workflow
tune_spec <- rand_forest(
  mode = "classification",
  mtry = tune(),
  trees = 500,
  min_n = tune()) %>%
  set_engine("ranger")

form <- formula(virality ~.)

rf_wf <- workflow() %>%
  add_formula(form) %>%
  add_model(tune_spec)

# to get a first idea how mtry and min_n affect our model, we run an initial tune, 
# without specifying the interval in which we want mtry and min_n to be

# cross validation folds
set.seed(123)
news_fold_og <- vfold_cv(news_train)

set.seed(123)
doParallel::registerDoParallel()
og_rf_grid <- tune_grid(
  rf_wf,
  resamples = news_fold_og,
  grid = 20
)

# plot AUC for different combinations of min_n and mtry
grid_plot_og <- og_rf_grid %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

grid_plot_og

# From the plot we can see that the AUC is larger for smaller mtry.
# For min_n, there is no clear correlation between number of min_n and AUC.
# Some of the variation for min_n could be due to the influence of mtry.
# We consider a smaller range of mtry but since the results for min_n were 
# inconclusive, we consider the full range tried in the general tune.

# now we tune again, but this time in a more specific range
# we try 25 combinations to get the combination with the highest AUC

# redefine grid for tuning
rf_grid_tar_og <- grid_regular(
  mtry(range = c(2, 10)),
  min_n(range = c(2, 40)),
  levels = 5
)

# tune
doParallel::registerDoParallel()
og_rf_grid_reg <- tune_grid(
  rf_wf,
  resamples = news_fold_og,
  grid = rf_grid_tar_og
)

# plot AUCs
auc_plot_og_hyp_tun <- ovun_rf_grid_reg %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

auc_plot_og_hyp_tun

og_rf_grid_reg %>%
  select_best("roc_auc")

# train final model with tuned hyperparameters,
# using the best combination determined by select_best
# In our case, mtry = 2 , min.node.size = 40

OOB_Error_og <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_Error_og)) {
  
  optimal_ranger_og <- ranger(
    formula         = virality ~ ., 
    data            = news_train, 
    num.trees       = 500,
    mtry            = 2,
    min.node.size   = 40,
    sample.fraction = .7,
    importance      = 'impurity'
  )
  
  OOB_Error_og[i] <- optimal_ranger_og$prediction.error
  
  print(i)
}

# load("optimal_ranger_og.RData")

# testing
pred_t_og <- predict(optimal_ranger_og, news_test)
pred_t_og <- pred_t_og[["predictions"]]

# confusion matrix
cm_og_t <- confusionMatrix(pred_t_og, news_test$virality,
                           mode = "prec_recall", positive = "1")

# Confusion matrix
#        0      1
# 0  10665   1226
# 1      0      0

# accuracy = 0.8969
# balanced accuracy = 0.5
# recall = 0.0000
# precision = NA

# ROC
pred_tn_og <- as.numeric(pred_t_og)
pred_og <- prediction(pred_tn_og, news_test[["virality"]])
perf_og_rf <- performance(pred_og, x.measure = "tpr", measure = "fpr")
plot(perf_og_rf, colorize = F, print.cutoffs.at = seq(0, 0.1, by = 0.01))

# AUC
auc_og <- performance(pred_og, measure = "auc")
auc_og <- unlist(slot(auc_og,"y.values"))
auc_og = round(auc_og, 4)
legend(.6, .2, auc_og, title = "AUC")

# AUC = 0.5

# since every viral article is misclassified as viral, the model is
# useless for us and even worse than the basic random forest

# next, we train the random forest with balanced training data


# --------------- 2: use random forest on ROSE dataset ---------------

# the first balanced training set we try for random forests is obtained through 
# a combination of over- and undersampling using the ROSE package
# it gives us a training set of the same size and a roughly equal distribution 
# of viral and non-viral articles

# to get an idea for initial performance of the random forest compared to the
# imbalanced training data, we run the basic model first

set.seed(123)
rf_ovun <- randomForest(
  formula = virality ~ .,
  data = ovun_news_train
)

rf_ovun
plot(rf_ovun)

# initially, it looks better than the random forest trained on imbalanced data
# as the viral articles are not systematically misclassified
# however, the OOB error is rather small (1.42%) which could result in
# overfitting. The number of trees does not have a significant influence on the
# out of bag error

# testing
test_rf_ovun <- predict(rf_ovun,news_test)

# confusion matrix
confusionMatrix(table(test_rf_ovun, news_test$virality))

#         0           1
# 0     10277       1073
# 1       388        153

# precision = 0.2743
# recall = 0.1175
# balanced accuracy = 0.5409

# ROC Curve
test_rf_ovun <- as.numeric(test_rf_ovun)
roc(news_test$virality, test_rf_ovun, plot = T)

# AUC = 0.5442

# The ROC and AUC are a little better than the ones obtained with the imbalanced
# training data but still not satisfactory

# to improve performance of the random forest, we tune the hyperparameters mtry
# and min_n again

# write workflow
tune_spec <- rand_forest(
  mode = "classification",
  mtry = tune(),
  trees = 500,
  min_n = tune()) %>%
  set_engine("ranger")

form <- formula(virality ~.)

rf_wf <- workflow() %>%
  add_formula(form) %>%
  add_model(tune_spec)

# to get a first idea how mtry and min_n affect our model, we run an initial tune
# but without specifying the interval in which we want mtry and min_n to be

# we do this again as we cannot be sure that mtry and min_n have the same affect
# on our balanced training data as they did on the imbalanced training data

set.seed(123)
doParallel::registerDoParallel()
ovun_rf_grid <- tune_grid(
  rf_wf,
  resamples = news_fold,
  grid = 20
)

# plot possible AUC for different min_n and mtry

grid_plot_ovun <- ovun_rf_grid %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

grid_plot_ovun

# from the plot we can see that the AUC is larger for smaller min_n.
# for mtry, there is no clear correlation between number of mtry and AUC
# some of the variation for mtry could be due to the influence of min node
# it still looks like smaller mtry will result in higher AUC

# specific tune
# now we tune again, but this time in a more specific range
# we try 16 combinations to get the combination with the highest AUC

rf_grid_tar <- grid_regular(
  mtry(range = c(2, 10)),
  min_n(range = c(2, 8)),
  levels = 4
)

# tune again with the targeted grid
doParallel::registerDoParallel()
ovun_rf_grid_reg <- tune_grid(
  rf_wf,
  resamples = news_fold,
  grid = rf_grid_tar
)

ovun_rf_grid_reg %>%
  collect_metrics()

# plot AUC again for 16 combinations
auc_plot_ovun_hyp_tun <- ovun_rf_grid_reg %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

auc_plot_ovun_hyp_tun

# select combination with highest AUC
ovun_rf_grid_reg %>%
  select_best("roc_auc")

# train final model with tuned hyperparameters
# using the best combination determined by select best
# in our case, mtry = 4, min.node.size = 2

OOB_Error_ovun <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_Error_ovun)) {
  
  optimal_ranger_ovun <- ranger(
    formula         = virality ~ ., 
    data            = ovun_news_train, 
    num.trees       = 500,
    mtry            = 4,
    min.node.size   = 2,
    sample.fraction = .7,
    importance      = 'impurity'
  )
  
  OOB_Error_ovun[i] <- optimal_ranger_ovun$prediction.error
  
  print(i)
}

# load("optimal_ranger_ovun.RData")

pred_t_ovun <- predict(optimal_ranger_ovun, news_test)
pred_t_ovun <- pred_t_ovun[["predictions"]]

# confusion matrix
pre_rec_ovun <- print(confusionMatrix(pred_t_ovun, news_test$virality,
                                      mode = "prec_recall", positive = "1"))

#         0     1
# 0   10560   408
# 1     105   818

# accuracy = 0.9572
# precision = 0.8862
# recall = 0.6672
# balanced accuracy = 0.8287

# ROC
pred_tn_ovun <- as.numeric(pred_t_ovun)
pred_ovun <- prediction(pred_tn_ovun, news_test[["virality"]])
perf_ovun_rf <- performance(pred_ovun, x.measure = "tpr", measure = "fpr")
plot(perf_og_rf, colorize = F, print.cutoffs.at = seq(0, 0.1, by = 0.01))

# AUC
auc_ovun <- performance(pred_ovun, measure = "auc")
auc_ovun <- unlist(slot(auc_og,"y.values"))
auc_og = round(auc_og, 4)
legend(.6, .2, auc_og, title = "AUC")

# AUC = 0.8287

# We see significant improvement from the basic random forest 
# as well as to the random forest trained on imbalanced data


# --------------- 3: use random forest on undersampled dataset ---------------

# train forest without tuning
set.seed(123)
rf_under <- randomForest(
  formula = virality ~ .,
  data    = news_down
)

rf_under

rf_under_plot <- plot(rf_under)


# compared to the previous basic models, the OOB error is much higher (33.55%)
# which means that our model will not overfit as much

test_rf_under <- predict(rf_under, news_test)

confusionMatrix(test_rf_under, news_test$virality, mode = "prec_recall",
                positive = "1")

#         0       1
# 0    6755     413
# 1    3910     813

# precision = 0.1721
# recall = 0.6631
# balanced accuracy = 0.6483

# ROC
test_rf_under_n <- as.numeric(test_rf_under)
roc(news_test$virality, test_rf_under_n, plot = T)

# AUC = 0.6483

# the performance of our basic model is better than that of the 
# original random forest but still worse than the random forest ROSE

# tune hyperparameters

# we again tune the hyperparameters of the random forest but now we go straight
# into the specific range of mtry and min_n since we assume that the effect of
# both are similar to the effect on the ROSE training set as both have balanced
# class distribution

# therefore, we can reuse the first step of the tuning process

# create cross-validation folds

set.seed(123)
news_fold_under <- vfold_cv(news_down)

# write workflow (reuse from ROSE, so we do not need to run it again)
# 
# tune_spec <- rand_forest(
#  mode = "classification",
#  mtry = tune(),
#  trees = 500,
#  min_n = tune()) %>%
#  set_engine("ranger")
# 
# form <- formula(virality ~.)
# 
# rf_wf <- workflow() %>%
#  add_formula(form) %>%
#  add_model(tune_spec)
# 
# we use the same grid as before
# rf_grid_tar <- grid_regular(
#  mtry(range = c(2, 10)),
#  min_n(range = c(2, 8)),
#  levels = 4
#)

#tune again with the targeted grid
doParallel::registerDoParallel()
set.seed(123)
under_rf_grid_reg <- tune_grid(
  rf_wf,
  resamples = news_fold_under,
  grid = rf_grid_tar
)

# plot AUC again
auc_plot_under_hyp_tun <- under_rf_grid_reg %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

auc_plot_under_hyp_tun

under_rf_grid_reg %>%
  select_best("roc_auc")


# train final model with tuned hyperparameters using the best combination determined by select_best
# In our case, mtry = 2 , min.node.size = 6

OOB_Error_under <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_Error_under)) {
  
  optimal_ranger_under <- ranger(
    formula         = virality ~ ., 
    data            = news_down, 
    num.trees       = 500,
    mtry            = 2,
    min.node.size   = 6,
    sample.fraction = .7,
    importance      = 'impurity'
  )
  
  OOB_Error_under[i] <- optimal_ranger_under$prediction.error
  
  print(i)
}

# load("optimal_ranger_under.RData")

under_pred_t <- predict(optimal_ranger_under, news_test)
under_pred_t <- under_pred_t[["predictions"]]


# confusion matrix
cm_under_t <- confusionMatrix(under_pred_t_cm, news_test$virality,
                              positive = "1", mode = "prec_recall")

#        0    1
# 0   6880  416
# 1   3785  810

# accuracy = 0.6467
# precision = 0.1763
# recall = 0.6607
# balanced accuracy = 0.6529


# ROC
under_pred_tn <- as.numeric(under_pred_prob)
under_pred <- prediction(under_pred_tn, news_test$virality)
perf_under_rf <- performance(under_pred, measure = "tpr", x.measure = "fpr")
roc_optimal_under <- plot(perf_under_rf, colorize = T,
                          print.cutoffs.at = seq(0, 0.1, by = 0.01)) 

# AUC = 0.6487

# tuning only improved our model a little bit and it is still significantly
# worse than the random forest ROSE


# --------------- 4: use random forest on oversampled dataset ---------------

# train random forest without tuning to see how it performs

set.seed(123)
rf_over <- randomForest(
  formula = virality ~ .,
  data    = upTrain
)

rf_over
plot(rf_over)

# the OOB error is very small (0.17%) and again there is no significant effect
# of the number of trees on the OOB error. We expect the basic model to overfit 
# due to the very small OOB error

# test
test_rf_over <- predict(rf_over,news_test)

# confusion matrix
confusionMatrix(test_rf_ovun, news_test$virality, mode = "prec_recall", positive = "1")

#         0      1
# 0   10284   1082
# 1     381    144

# precision = 0.2743
# recall = 0.1175
# balanced accuracy = 0.5401


# ROC
test_rf_over_n <- as.numeric(test_rf_over)
roc(news_test$virality, test_rf_over_n, plot = T)

# AUC = 0.5078

# again, the basic model overfits and does not predict many viral articles correctly
# we try to improve the performance by tuning the hyperparameters

# tune
# we again tune the hyperparameters of the random forest but now we go straight
# into the specific range of mtry and min_n since we assume that the effect of
# both are similar to the effect on the ROSE training set as both have balanced
# class distribution
# therefore, we can reuse the first step of the tuning process

# create cross-validation folds
set.seed(123)
news_fold_over <- vfold_cv(upTrain)

# write workflow (reuse from ROSE, so we do not need to run it again)

#tune_spec <- rand_forest(
#  mode = "classification",
#  mtry = tune(),
#  trees = 500,
#  min_n = tune()) %>%
#  set_engine("ranger")

# form <- formula(virality ~.)

# rf_wf <- workflow() %>%
#  add_formula(form) %>%
#  add_model(tune_spec)

# we use the same grid as before
# rf_grid_tar <- grid_regular(
#  mtry(range = c(2, 10)),
#  min_n(range = c(2, 8)),
#  levels = 4
#)

#tune again with the targeted grid
doParallel::registerDoParallel()
over_rf_grid_reg <- tune_grid(
  rf_wf,
  resamples = news_fold_over,
  grid = rf_grid_tar
)
 
over_rf_grid_reg %>%
  collect_metrics()
 
# plot AUC for different combinations
auc_plot_over_hyp_tun <- over_rf_grid_reg %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")
 
auc_plot_over_hyp_tun
 
over_rf_grid_reg %>%
select_best("roc_auc")

# the best AUC is very high which could mean that our model will overfit even with
# cross validation and tuned hyperparamaters
 
# train final model with tuned hyperparameters using the best combination determined by select_best
# In our case, mtry = 10, min.node.size = 2

OOB_Error_over <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_Error_over)) {
  
  optimal_ranger_over <- ranger(
    formula         = virality ~ ., 
    data            = up_Train, 
    num.trees       = 500,
    mtry            = 10,
    min.node.size   = 2,
    sample.fraction = .7,
    importance      = 'impurity'
  )
  
  OOB_Error_over[i] <- optimal_ranger_over$prediction.error
  
  print(i)
}

# load("optimal_ranger_over.RData")

over_pred_t <- predict(optimal_ranger_over, news_test)
over_pred_t <- over_pred_t[["predictions"]]


# confusion matrix
cm_over_t <- confusionMatrix(table(over_pred_t, news_test$virality,
                                   mode = "prec_recall", positive = "1"))

#         0       1
# 0   10595    1197
# 1    553      199

# accuracy = 0.8934
# precision: 0.2646
# recall: 0.1623
# balanced accuracy: 0.5649

# ROC
pred_tn_over <- as.numeric(pred_t_over)
pred_over <- prediction(pred_tn_over, news_test[["virality"]])
perf_over_rf <- performance(pred_over, x.measure = "tpr", measure = "fpr")
plot(perf_over_rf, colorize = F, print.cutoffs.at = seq(0, 0.1, by = 0.01))


# AUC
auc_over <- performance(pred_over, measure = "auc")
auc_over <- unlist(slot(auc_over,"y.values"))
auc_over <- round(auc_over, 4)
legend(.6, .2, auc_over, title = "AUC")

# AUC = 0.5552

# compared to the basic model, we make a small improvement, however, the random
# forest still overfits a lot. Compared to the random forest ROSE 
# and random forest trained on undersampled data, it is significantly worse


# --------------- 5: use random forest on SMOTE dataset ---------------

# the last resampled training data we try is SMOTE

# train basic model

set.seed(123)
rf_SMOTE <- randomForest(
  formula = virality ~ .,
  data    = smote_news1
)

rf_SMOTE
plot(rf_SMOTE)

# from the plot we see that the OOB error for the positive class is higher than
# for the negative class. Overall the OOB error is 10.84% and in between the very
# low OOB error from the oversampled training data and the higher one from the
# undersampled training data

test_rf_SMOTE <- predict(rf_SMOTE,news_test)

# confusion matrix
confusionMatrix(test_rf_SMOTE, news_test$virality, mode = "prec_recall", positive = "1")

#         0       1
# 0    9655     935
# 1    1010     291

# precision = 0.2237
# recall = 0.2374
# balanced accuracy = 0.5713

# ROC
test_rf_SMOTE_n <- as.numeric(test_rf_SMOTE)
roc(news_test$virality, test_rf_SMOTE_n, plot = T)

# AUC = 0.5713

# the performance of the basic model is not particularly good 
# but already better than the tuned random forest trained on oversampled data

# tune hyperparameters to improve performance

# we again tune the hyperparameters of the random forest but now we go straight
# into the specific range of mtry and min_n since we assume that the effect of
# both are similar to the effect on the ROSE training set as both have balanced
# class distribution

# therefore, we can reuse the first step of the tuning process

# cross-validation samples

set.seed(123)
news_fold_SMOTE <- vfold_cv(smote_Train)

# write workflow

# tune_spec <- rand_forest(
#   mode = "classification",
#   mtry = tune(),
#   trees = 500,
#   min_n = tune()) %>%
#   set_engine("ranger")

# form <- formula(virality ~.)

# rf_wf <- workflow() %>%
#  add_formula(form) %>%
#  add_model(tune_spec)

# specific tune
# since the training set is similar to the first one we used,
# we go straight to tuning in a specific range
# we again try 16 combinations to get the combination with the highest AUC

# we use the same grid as before.
# rf_grid_tar <- grid_regular(
#   mtry(range = c(2, 10)),
#   min_n(range = c(2, 8)),
#   levels = 4
# )

# tune again with the targeted grid
doParallel::registerDoParallel()
SMOTE_rf_grid_reg <- tune_grid(
  rf_wf,
  resamples = news_fold_SMOTE,
  grid = rf_grid_tar
)


# plot AUC
auc_plot_SMOTE_hyp_tun <- SMOTE_rf_grid_reg %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

auc_plot_SMOTE_hyp_tun

over_rf_grid_reg %>%
  select_best("roc_auc")

# train final model with tuned hyperparameters and cross validation,
# using the best combination determined by select_best.
# In our case, mtry = 4, min.node.size = 2

OOB_Error_SMOTE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_Error_SMOTE)) {

  optimal_ranger_SMOTE <- ranger(
    formula         = virality ~ .,
    data            = smote_news1,
    num.trees       = 500,
    mtry            = 4,
    min.node.size   = 2,
    sample.fraction = .7,
    importance      = 'impurity'
  )

  OOB_Error_SMOTE[i] <- optimal_ranger_SMOTE$prediction.error

  print(i)
}

# load("optimal_ranger_SMOTE.RData")

SMOTE_pred_t <- predict(optimal_ranger_SMOTE, news_test)
SMOTE_pred_t <- SMOTE_pred_t[["predictions"]]

#confusion matrix
pre_rec_SMOTE <- print(confusionMatrix(SMOTE_pred_t, news_test$virality,
                                       mode = "prec_recall", positive = "1"))

#      0       1
# 0  9997    990
# 1   668    236

# accuracy = 0.8606
# precision = 0.2611
# recall = 0.1925   
# balanced accuracy = 0.5649 


# ROC
pred_tn_SMOTE <- as.numeric(SMOTE_pred_t)
pred_SMOTE <- prediction(pred_tn_SMOTE, news_test[["virality"]])
perf_SMOTE_rf <- performance(pred_SMOTE, x.measure = "tpr", measure = "fpr")
plot(perf_og_rf, colorize = F, print.cutoffs.at = seq(0, 0.1, by = 0.01))


# AUC
auc_SMOTE <- performance(pred_SMOTE, measure = "auc")
auc_SMOTE <- unlist(slot(auc_SMOTE,"y.values"))
auc_SMOTE <- round(auc_SMOTE, 4)
legend(.6, .2, auc_og, title = "AUC")

# AUC = 0.7022

# compared to the basic model, balanced accuracy and precision improved while
# recall got worse. 
# The AUC is significantly higher after tuning, 
# however, it still performs worse than the random forest ROSE.




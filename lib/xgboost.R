library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

setwd("~/ADS/Fall2017-project3-fall2017-project3-grp8")

labs <- read_csv("data/our_data/training_set/label_train.csv")

df_sift <- read_csv("data/our_data/training_set/sift_train.csv")

folds <- createFolds(df_sift, k = 10, list = TRUE, returnTrain = FALSE)

train <- df_sift[-folds$Fold02,]
test <- df_sift[folds$Fold02,]

# very simple xgboost

xgb <- xgboost(data = data.matrix(train[,-1]),
               label = train[, 1],
               eta = 0.2,
               max_depth = 10, 
               nrounds = 25,
               early_stopping_rounds = 25,
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softmax",
               num_class = 3,
               nthread = 3
              )

test_error <- sum(predict(xgb, data.matrix(test[,-1])) != test[,1])/length(test[,1])

# test error is around 20%... okay.


#trying xgb.cv

xgb <- xgb.cv(data = data.matrix(df_sift[,-1]),
              label = df_sift[,1],
              eta = 0.2,
              max_depth = 10,
              nrounds = 25,
              nfold = 10,
              num_class = 3,
              early_stopping_rounds = 25,
              prediction = TRUE,
              metrics = "merror",
              objective = "multi:softmax",
              stratified = TRUE)

# using the above
# train-merror:0.000000+0.000000	test-merror:0.228333+0.024866 

xgb2 <- xgb.cv(data = data.matrix(df_sift[,-1]),
              label = df_sift[,1],
              eta = 0.2,
              max_depth = 15,
              nrounds = 25,
              nfold = 10,
              num_class = 3,
              early_stopping_rounds = 25,
              prediction = TRUE,
              metrics = "mlogloss",
              objective = "multi:softmax",
              stratified = TRUE)

df_gist <- read_csv("data/our_data/training_set/features_GIST.csv", col_names = FALSE)

df_gist <- cbind(labs[,2], df_gist)

xgb2_gist <- xgb.cv(data = data.matrix(df_gist[,-1]),
               label = df_gist[,1],
               eta = 0.3,
               max_depth = 10,
               nrounds = 25,
               nfold = 10,
               num_class = 3,
               early_stopping_rounds = 25,
               prediction = TRUE,
               metrics = "merror",
               objective = "multi:softmax",
               stratified = TRUE)

df_hog <- read_csv("data/our_data/training_set/features_HOG.csv")

df_hog <- cbind(labs[,2], df_hog[,-1])

# currently the best one is below, i'll do a grid search and find the best parameters
xgbbest_hog <- xgb.cv(data = data.matrix(df_hog[,-1]),
                      label = df_hog[,1],
                      eta = 0.35,
                      max_depth = 10,
                      nrounds = 60,
                      nfold = 10,
                      num_class = 3,
                      early_stopping_rounds = 60,
                      prediction = TRUE,
                      metrics = "merror",
                      objective = "multi:softmax",
                      stratified = TRUE)

xgb2_hog <- xgb.cv(data = data.matrix(df_hog[,-1]),
                   label = df_hog[,1],
                   eta = 0.25,
                   gamma = 1,
                   min_child_weight = 1.1,
                   subsample = 0.6,
                   max_depth = 10,
                   nrounds = 60,
                   nfold = 10,
                   num_class = 3,
                   early_stopping_rounds = 60,
                   prediction = TRUE,
                   metrics = "merror",
                   objective = "multi:softmax",
                   stratified = TRUE)


xgb2_gist <- xgb.cv(data = data.matrix(df_gist[,-1]),
                    label = df_gist[,1],
                    eta = 0.3,
                    gamma = 1,
                    min_child_weight = 1.1,
                    subsample = 0.6,
                    max_depth = 10,
                    nrounds = 30,
                    nfold = 10,
                    num_class = 3,
                    early_stopping_rounds = 30,
                    prediction = TRUE,
                    metrics = "merror",
                    objective = "multi:softmax",
                    stratified = TRUE)

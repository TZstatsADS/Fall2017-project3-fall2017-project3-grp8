library(xgboost)
library(readr)
library(stringr)
library(caret)

setwd("~/ADS/Fall2017-project3-fall2017-project3-grp8")

labs <- read_csv("data/our_data/training_set/label_train.csv")

labs[,3] <- 0
labs[,3][labs[,2] == 0,] <- 'A'
labs[,3][labs[,2] == 1,] <- 'B'
labs[,3][labs[,2] == 2,] <- 'C'

df_sift <- read_csv("data/our_data/training_set/sift_train.csv")
df_sift <- cbind(labs[,2], df_sift)

df_gist <- read_csv("data/our_data/training_set/features_GIST.csv", col_names = FALSE)
df_gist <- cbind(labs[,2], df_gist)

df_hog <- read_csv("data/our_data/training_set/features_HOG.csv")
df_hog <- cbind(labs[,c(2,3)], df_hog[,-1])

test_size <- floor(0.25 * nrow(df_hog))
test_indices <-  sort(sample(seq_len(nrow(df_hog)), size=test_size))

train <- df_hog[-test_indices,]
test <- df_hog[test_indices,]


control <- trainControl(method="cv", number = 5, search = "grid", verboseIter = TRUE, returnData = FALSE
                        , allowParallel = TRUE)

xgb_grid_1 = expand.grid(
  nrounds = 100,
  max_depth = c(5, 10),
  eta = c(0.1, 1, 10),
  gamma = c(0.1, 1, 10),
  colsample_bytree = c(0.1, 0.5, 1),
  min_child_weight = c(0.1, 1, 10),
  subsample = c(0.1, 0.5, 1)
)

model1 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_1, objecitve = "multi:softprob", tuneLength=3)

preds <- predict(model1, test[, -c(1,2)])
acc <- 1 - sum(preds != test[,2])/length(test[,2])

# The final values used for the model were nrounds = 100, max_depth = 10, eta = 0.1, gamma = 1,
# colsample_bytree = 1, min_child_weight = 1 and subsample = 0.5.
# CV Accuracy 0.784, Test Acc 0.768

xgb_grid_2 = expand.grid(
  nrounds = 100,
  max_depth = c(8, 10, 12),
  eta = c(0.1, 0.2),
  gamma = c(0.5, 1, 3),
  colsample_bytree = c(1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.4, 0.5, 0.6)
)

model2 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_2, objecitve = "multi:softprob", tuneLength=3)

# Fitting nrounds = 100, max_depth = 12, eta = 0.2, gamma = 0.5, colsample_bytree = 1, min_child_weight = 5, subsample = 0.4 on full training set
# CV Accuracy 0.2075599, Test Acc 0.238667

preds <- predict(model2, test[, -c(1,2)])
acc <- 1 - sum(preds != test[,2])/length(test[,2])

xgb_grid_3 = expand.grid(
  nrounds = 100,
  max_depth = c(12, 14),
  eta = c(0.1, 0.15, 0.2),
  gamma = c(0.25, 0.5, 0.75),
  colsample_bytree = c(1),
  min_child_weight = c(4, 5, 6),
  subsample = c(0.4, 0.45, 0.5)
)

model3 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_3, objecitve = "multi:softprob", tuneLength=3)

1 - max(model3[["results"]][["Accuracy"]])

preds <- predict(model3, test[, -c(1,2)])
err <- sum(preds != test[,2])/length(test[,2])

# Fitting nrounds = 100, max_depth = 14, eta = 0.15, gamma = 0.5, colsample_bytree = 1, min_child_weight = 6, subsample = 0.5 on full training set

# CV err 0.2111046, err = 0.2306667

xgb_grid_4 = expand.grid(
  nrounds = 100,
  max_depth = c(13, 14, 15),
  eta = c(0.15, 10),
  gamma = c(0.5, 20),
  colsample_bytree = c(1),
  min_child_weight = c(6, 12),
  subsample = c(0.5, 0.7)
)

model4 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_4, objecitve = "multi:softprob", tuneLength=3)

preds <- predict(model4, test[, -c(1,2)])
err <- sum(preds != test[,2])/length(test[,2])

# Fitting nrounds = 100, max_depth = 15, eta = 0.15, gamma = 0.5, colsample_bytree = 1, min_child_weight = 6, subsample = 0.7 on full training set

#CV err 0.2186661, err = 0.22

xgb_grid_5 = expand.grid(
  nrounds = 100,
  max_depth = c(14, 15, 16),
  eta = c(0.13, 0.15, 0.17),
  gamma = c(0.4, 0.5, 0.6),
  colsample_bytree = c(1),
  min_child_weight = c(5.5, 6, 6.5),
  subsample = c(0.5, 0.6, 0.7)
)

model5 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_5, objecitve = "multi:softprob", tuneLength=3)

1 - max(model5[["results"]][["Accuracy"]])

preds <- predict(model5, test[, -c(1,2)])
err <- sum(preds != test[,2])/length(test[,2])

# nrounds = 100, max_depth = 16, eta = 0.17, gamma = 0.5, colsample_bytree = 1, min_child_weight = 6.5, subsample = 0.6 on full training set
# CV Error  0.2097821, test err = 0.2093333

xgb_grid_6 = expand.grid(
  nrounds = 100,
  max_depth = c(15, 16, 17),
  eta = c(0.16, 0.17, 0.18),
  gamma = c(0.45, 0.5, 0.55),
  colsample_bytree = c(1),
  min_child_weight = c(6.3, 6.5, 6.7),
  subsample = c(0.55, 0.6, 0.65)
)

model6 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_6, objecitve = "multi:softprob", tuneLength=3)

preds <- predict(model6, test[, -c(1,2)])
err <- sum(preds != test[,2])/length(test[,2])

# max_depth = 17, eta = 0.18, gamma = 0.45, colsample_bytree = 1, min_child_weight = 6.5, subsample = 0.65 on full training set

# CV Accuracy - 0.2093459, test err = 0.2186667

# might be starting to overfit here

xgb_grid_7 = expand.grid(
  nrounds = 100,
  max_depth = c(16, 17, 18),
  eta = c(0.17, 0.18, 0.19),
  gamma = c(0.44, 0.45, 0.46),
  colsample_bytree = c(1),
  min_child_weight = c(6.4, 6.5, 6.6),
  subsample = c(0.6, 0.65, 0.7)
)

model7 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_7, objecitve = "multi:softprob", tuneLength=3)

1 - max(model7[["results"]][["Accuracy"]])
preds <- predict(model7, test[, -c(1,2)])
err <- sum(preds != test[,2])/length(test[,2])

# max_depth = 18, eta = 0.18, gamma = 0.45, colsample_bytree = 1, min_child_weight = 6.6, subsample = 0.7 on full training set
# cv err = 0.2084235, test err = 0.2146667

# Best model is model 5

# nrounds = 100, max_depth = 16, eta = 0.17, gamma = 0.5, colsample_bytree = 1, min_child_weight = 6.5, subsample = 0.6 on full training set


xgbbest <- xgboost(data = data.matrix(train[,-c(1,2)]),
                      label = train[,1],
                      max_depth = 18,
                      eta = 0.18,
                      gamma = 0.45,
                      colsample_bytree = 1,
                      min_child_weight = 6.6,
                      subsample = 0.7,
                      nrounds = 100,
                      num_class = 3,
                      early_stopping_rounds = 100,
                      prediction = TRUE,
                      metrics = "merror",
                      objective = "multi:softmax",)

preds <- predict(xgbbest, data.matrix(test[, -c(1,2)]))
err <- sum(preds != test[,1])/length(test[,1])

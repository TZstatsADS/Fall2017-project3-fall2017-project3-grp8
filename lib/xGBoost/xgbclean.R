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

# Tune 1: CV Error 0.216, Test Error 0.232
# Tune 2: CV Error 0.2075599, Test Error 0.238667
# Tune 3: CV Error 0.2111046, Test Error 0.2306667
# Tune 4: CV Error 0.2186661, Test Error 0.22
# Tune 5: CV Error  0.2097821, Test Error = 0.2093333
# Tune 6: CV Error 0.2093459, Test err = 0.2186667
### might be starting to overfit here
# Tune 7: CV err = 0.2084235, Test err = 0.2146667

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

# Best model is model 5
# max_depth = 18, eta = 0.18, gamma = 0.45, colsample_bytree = 1, min_child_weight = 6.6, subsample = 0.7 on full training set

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
                      objective = "multi:softmax")

preds <- predict(xgbbest, data.matrix(test[, -c(1,2)]))
err <- sum(preds != test[,1])/length(test[,1])

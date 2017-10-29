## author: Shiqi Duan
## date: "October 28, 2017"

# load libraries
if(!require("randomForest")){
  install.packages("randomForest")
}
if(!require("caret")){
  install.packages("caret")
}
if(!require("e1071")){
  install.packages("e1071")
}
if(!require("ggplot2")){ 
}

library(randomForest)
library(randomForest)
library(caret)
library(e1071)
library(knitr)
library(ggplot2)


#### Controls

K <- 5 # number of CV folds
proportion = 0.75 # training set proportion
seed = 123 # set seed

#### Import and split the data into test and training set (75% training set and 25% test set)

setwd("/Users/duanshiqi/Documents/GitHub/Fall2017-project3-fall2017-project3-grp8/lib")
features <- read.csv("../data/our_data/training_set/features_HOG.csv",header=TRUE,as.is=TRUE)[,-1]
label.train <- read.csv("../data/our_data/training_set/label_train.csv",header=TRUE,as.is=TRUE)[,-1]


n <- dim(features)[1]
set.seed(seed)
index <- sample(n, n*proportion, replace = FALSE)

x.train <- features[index,]
y.train <- label.train[index]

x.test <- features[-index,]
y.test <- label.train[-index]


#### Model selection with cross-validation: choosing different values of training model parameters
m <- length(y.train)
m.fold <- floor(m/K)
set.seed(seed)
s <- sample(rep(1:K, c(rep(m.fold, K-1), m-(K-1)*m.fold))) 

cv.error <- rep(NA, K)
opt.mtry <- rep(NA,K) # optimal mtry for certain fold training data

n_trees <- seq(500,1000,100)
cv.ntree.error <- rep(NA, length(n_trees)) # lowest cv-error for certain ntree value
best.ntree.mtry <- rep(NA, length(n_trees)) # best mtry for certain ntree value

for (j in 1:length(n_trees)){
  for (i in 1:K){
    train.data <- x.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- x.train[s == i,]
    test.label <- y.train[s == i]
  
    fit <- tuneRF(train.data, as.factor(train.label),
                mtreeTry = n_trees[j], 
                doBest = TRUE)
 
    # Get the 'mtry' for trained model
    opt.mtry[i] <- fit$mtry
    pred <- predict(fit, test.data)
    cv.error[i] <- mean(pred != test.label)
  }
  
  # Get the lowest error rate of cross validation
  cv.ntree.error[j] <- min(cv.error)
  best.ntree.mtry[j] <- opt.mtry[which.min(cv.error)]
}

#### Evaluation
# Visualize Cross Validation
ggplot(data = data.frame(cv.ntree.error)) + geom_point(aes(x = n_trees, y = cv.ntree.error), color = "blue")

# Get the best parameters
best <- which.min(cv.ntree.error)
best.ntree <- n_trees[best] # 900
best.mtry <- best.ntree.mtry[best] #14

# Training error
tm_train<-system.time(fit.1 <- randomForest(x.train, as.factor(y.train), 
                                  mtry = best.mtry, ntree = best.ntree, importance = TRUE))

train_error <- mean(fit.1$predicted != y.train)
train_error #0.2488
save(fit.1, file="../output/RFs_fit_HOG.RData")

# Test error
tm_test<-system.time(test_pred <- predict(fit.1, x.test))
test_error <- mean(test_pred != y.test)
test_error #0.2506
save(test_pred, file="../output/pred_test_RFs_HOG.RData")

### Summarize Running Time
cat("Time for training model=", tm_train[1], "s \n") # 34.427s
cat("Time for testing model=", tm_test[1], "s \n") # 0.149s

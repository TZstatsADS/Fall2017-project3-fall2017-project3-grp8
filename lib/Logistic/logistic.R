### Author: Shiqi Duan
### Project 3
### ADS Fall 2017

train.lg <- function(dat_train, label_train, par = NULL){
  
  ### Train a multinomial logistic regression using processed features from training images
  
  ### Input: 
  ###  -  processed features from images, 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("nnet")
  
  ### combine the features and the labels together
  #y<-label_train
  mydata<-cbind(dat_train,label_train)
  mydata$label_train<-factor(label_train)
  mydata$label_train<-relevel(mydata$label_train,ref="0")
  
  ### train with multinomial logistic regression model
  if(is.null(par)){
    maxit <- 100
  } else {
    maxit <- par$maxit
  }
  fit_lg <- multinom(label_train~., data = mydata, MaxNWts = 20000, maxit = maxit)

  return(list(fit=fit_lg))
}


test.lg <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("nnet")
  
  pred <- predict(fit_train$fit, dat_test)
  
  return(pred)
}


cv.function <- function(X.train, y.train, iter, K){
  
  set.seed(0)
  
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i]
    
    par <- list(maxit=iter)
    fit <- train.lg(train.data, train.label, par)
    pred <- test.lg(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)  
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
  
}



### GIST features ###

setwd("/Users/duanshiqi/Documents/Github/Fall2017-project3-fall2017-project3-grp8/lib")
X<-read.csv("../data/our_data/training_set/features_GIST.csv",header=FALSE,as.is=TRUE)
y<-read.csv("../data/our_data/training_set/label_train.csv",header=TRUE,as.is=TRUE)[,-1]

# split into training set and validation(test) set
proportion = 0.75 # training set proportion
set.seed(0)
n = length(y)
index <- sample(n, n*proportion, replace = FALSE)

x.train <- X[index,]
y.train <- y[index]

x.test <- X[-index,]
y.test <- y[-index]

# Choosing between different values of maximum interation for Logistic
iter_values <- seq(10,30,1)
err_cv <- array(dim=c(length(iter_values), 2))
K <- 5  # number of CV folds
for(k in 1:length(iter_values)){
  cat("k=", k, "\n")
  err_cv[k,] <- cv.function(x.train, y.train, iter_values[k], K)
}

# Visualize CV results
#pdf("../fig/cv_LG_results.pdf", width=7, height=5)
plot(iter_values, err_cv[,1], xlab="Maximum Iteration", ylab="CV Error",
     main="Cross Validation Error", ylim=c(0.2, 0.4))
points(iter_values, err_cv[,1], col="blue", pch=16)
lines(iter_values, err_cv[,1], col="blue")
arrows(iter_values, err_cv[,1]-err_cv[,2],iter_values, err_cv[,1]+err_cv[,2], 
       length=0.1, angle=90, code=3)

# Choose the best parameter value
min(err_cv[,1])  # 0.2733
iter_best <- iter_values[which.min(err_cv[,1] + err_cv[,2])]
par_best <- list(maxit=iter_best) # maxit=13 is the best

# train the model with the entire training set
tm_train <- system.time(fit_train <- train.lg(x.train, y.train, par_best))
pred_train <- test.lg(fit_train, x.train)
train.error <- mean(pred_train != y.train)
train.error #0.2364
save(fit_train, file="../output/fit_train_Logistic_Gist.RData")

### Make prediction 
tm_test <- system.time(pred_test <- test.lg(fit_train, x.test))
test.error <- mean(pred_test != y.test)
test.error #0.2346
save(pred_test, file="../output/pred_test_Logistic_Gist.RData")

### Summarize Running Time
#cat("Time for constructing training features=", tm_feature_train[1], "s \n")
#cat("Time for constructing testing features=", tm_feature_test[1], "s \n")
cat("Time for training model=", tm_train[1], "s \n") # 1.284s
cat("Time for making prediction=", tm_test[1], "s \n")  # 0.057s



### HOG features ###
X<-read.csv("../data/our_data/training_set/features_HOG.csv",header=TRUE,as.is=TRUE)[,-1]

# split into training set and validation(test) set
proportion = 0.75 # training set proportion
set.seed(0)
n = length(y)
index <- sample(n, n*proportion, replace = FALSE)

x.train <- X[index,]
y.train <- y[index]

x.test <- X[-index,]
y.test <- y[-index]

# Choosing between different values of maximum interation for Logistic
iter_values <- seq(100,460,10)
err_cv <- array(dim=c(length(iter_values), 2))
K <- 5  # number of CV folds
for(k in 1:length(iter_values)){
  cat("k=", k, "\n")
  err_cv[k,] <- cv.function(x.train, y.train, iter_values[k], K)
}

# Visualize CV results
plot(iter_values, err_cv[,1], xlab="Maximum Iteration", ylab="CV Error",
     main="Cross Validation Error", ylim=c(0.2, 0.4))
points(iter_values, err_cv[,1], col="blue", pch=16)
lines(iter_values, err_cv[,1], col="blue")
arrows(iter_values, err_cv[,1]-err_cv[,2],iter_values, err_cv[,1]+err_cv[,2], 
       length=0.1, angle=90, code=3)

# Choose the best parameter value
min(err_cv[,1]) # min err_cv is 0.2173
iter_best <- iter_values[which.min(err_cv[,1] + err_cv[,2])]
par_best <- list(maxit=iter_best) # maxit=190 is the best

# train the model with the entire training set
tm_train <- system.time(fit_train <- train.lg(x.train, y.train, par_best))
pred_train <- test.lg(fit_train, x.train)
train.error <- mean(pred_train != y.train)
train.error #0.1933
save(fit_train, file="../output/fit_train_Logistic_HOG.RData")

### Make prediction 
tm_test <- system.time(pred_test <- test.lg(fit_train, x.test))
test.error <- mean(pred_test != y.test)
test.error #0.228
save(pred_test, file="../output/pred_test_Logistic_HOG.RData")

### Summarize Running Time
cat("Time for training model=", tm_train[1], "s \n") # 0.601s
cat("Time for making prediction=", tm_test[1], "s \n") #0.009s



### SIFT features ###

X<-read.csv("../data/our_data/training_set/sift_train.csv",header=TRUE,as.is=TRUE)[,-1]

# split into training set and validation(test) set
proportion = 0.75 # training set proportion
set.seed(0)
n = length(y)
index <- sample(n, n*proportion, replace = FALSE)

x.train <- X[index,]
y.train <- y[index]

x.test <- X[-index,]
y.test <- y[-index]

# Choosing between different values of maximum interation for Logistic
iter_values <- seq(30,50,5)
err_cv <- array(dim=c(length(iter_values), 2))
K <- 5  # number of CV folds
for(k in 1:length(iter_values)){
  cat("k=", k, "\n")
  err_cv[k,] <- cv.function(x.train, y.train, iter_values[k], K)
}

# Visualize CV results
plot(iter_values, err_cv[,1], xlab="Maximum Iteration", ylab="CV Error",
     main="Cross Validation Error", ylim=c(0.2, 0.4))
points(iter_values, err_cv[,1], col="blue", pch=16)
lines(iter_values, err_cv[,1], col="blue")
arrows(iter_values, err_cv[,1]-err_cv[,2],iter_values, err_cv[,1]+err_cv[,2], 
       length=0.1, angle=90, code=3)

# Choose the best parameter value
min(err_cv[,1]) # min err_cv is 0.2733
iter_best <- iter_values[which.min(err_cv[,1])]
par_best <- list(maxit=iter_best) # maxit=40 is the best

# train the model with the entire training set
tm_train <- system.time(fit_train <- train.lg(x.train, y.train, par_best))
pred_train <- test.lg(fit_train, x.train)
train.error <- mean(pred_train != y.train)
train.error #0.0675
save(fit_train, file="../output/fit_train_Logistic_SIFT.RData")

### Make prediction 
tm_test <- system.time(pred_test <- test.lg(fit_train, x.test))
test.error <- mean(pred_test != y.test)
test.error #0.272
save(pred_test, file="../output/pred_test_Logistic_SIFT.RData")

### Summarize Running Time
cat("Time for training model=", tm_train[1], "s \n") # 190.928s
cat("Time for making prediction=", tm_test[1], "s \n") #3.095s

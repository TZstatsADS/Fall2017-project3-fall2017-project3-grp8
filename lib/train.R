train.gbm = function(train.data,par = NULL){
  
  ### Train a GBM classifier using features of training images
  
  ### Input:
  ### - training data including features of training images 
  ###                       and class labels of training images
  ### Output:
  ### - training gbm model specification
  
  ### load libraries
  library("gbm")
  
  ### train with GBM
  if(is.null(par)){
    n_trees = 100
    interaction_depth = 1
    shrinkage = 0.05
  }
  else{
    n_trees = par$n_trees
    interaction_depth = par$interaction_depth
    shrinkage = par$shrinkage
  }
  
  fit_gbm = gbm(y ~ .,
              data = train.data,
              distribution = 'multinomial',
              n.trees = n_trees,
              interaction.depth = interaction_depth,
              shrinkage = shrinkage)
  best_iter <- gbm.perf(fit_gbm, method = "OOB", plot.it = FALSE)
  
  return(list(fit = fit_gbm, iter = best_iter))
}
                              
  

train.svm_rbf<-function(train.data, par = NULL){
  
  ### Train a SVM classifier with RBF kernel using features of training images
  
  ### Input:
  ### - training data including features of training images 
  ###                       and class labels of training images
  ### Output:
  ### - training svm_rbf model specification
  
  ### load libraries
  library("e1071")
  
  ### train with svm RBF
  
  if(is.null(par)){
    cost = 5
    gamma = 100
  }
  else{
    cost = par$cost
    gamma = par$gamma
  }
  
  fit_svm_rbf <- svm(y~., data = train.data, 
                     kernel = "radial", scale = FALSE,
                     cost = cost, gamma = gamma)
  return(fit_svm_rbf)
}


train.xgb <- function(train.data, par = NULL){
  
  ### Train a Xgboost classifier using features of training images
  
  ### Input:
  ### - training data including features of training images 
  ###                       and class labels of training images
  ### Output:
  ### - training Xgboost model specification
  
  ### load libraries
  library("xgboost")
  
  ### train with Xgboost
  
  
  if(is.null(par)){
    max_depth = 16
    eta = 0.17
    gamma = 0.5
    min_child_weight = 6.5
    subsample = 0.6
  } 
  else {
    eta <- par$eta
    max_depth <- par$max_depth
    gamma <- par$gamma
    min_child_weight = par$min_child_weight
    subsample <- par$subsample
    }
  
  fit_xgb <- xgboost(data = data.matrix(train.data[, -1]),
                     label = as.numeric(train.data[, 1]) - 1,
                     nrounds = 100,
                     max_depth = max_depth,
                     eta = eta,
                     gamma = gamma,
                     colsample_bytree = 1, 
                     early_stopping_rounds = 100,
                     min_child_weight = min_child_weight,
                     subsample = subsample,
                     eval_metric = "merror",
                     objective = "multi:softmax",
                     num_class = 3,
                     nthread = 3)
  
  return(fit_xgb)
}


train.lg <- function(train.data, par = NULL){
  
  ### Train a multinomial logistic regression using processed features from training images
  
  ### Input: 
  ###  -  processed features from images, 
  ###           and class labels for training images
  ### Output: training logistic model specification
  
  ### load libraries
  library("nnet")
  
  ### combine the features and the labels together
  #y<-label_train
  #mydata<-cbind(dat_train,label_train)
  #mydata$label_train<-factor(label_train)
  #mydata$label_train<-relevel(mydata$label_train,ref="0")
  
  ### train with multinomial logistic regression model
  if(is.null(par)){
    maxit <- 190
  } else {
    maxit <- par$maxit
  }
  fit_lg <- multinom(y~., data = train.data, MaxNWts = 20000, maxit = maxit)
  
  return(fit_lg)
}
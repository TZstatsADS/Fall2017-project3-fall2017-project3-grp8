
test.gbm = function(fit_train, data.test) {
  ### Fit the GBM model with testing data
  
  ### Input: 
  ###  - the fitted GBM model using training data
  ###  - processed features from testing images
  ### Output: prediction labels
  
  ### load libraries
  library("gbm")
  
  predBST = predict(fit_train$fit,
                    n.trees=fit_train$iter, 
                    newdata=data.test,
                    type='response')
  
  p.predBST <- apply(predBST, 1, which.max) - 1
  
  return(p.predBST)
  
}


test.svm_rbf <- function(fit_train, data.test){
  
  ### Fit the svm_rbf model with testing data
  
  ### Input: 
  ###  - the fitted svm_rbf model using training data
  ###  -  processed features from testing images 
  ### Output: prediction labels
  
  ### load libraries
  library("e1071")
  pred <- predict(fit_train, data.test)
  
  return(pred)
}



test.svm_linear <- function(fit_train, data.test){
  
  ### Fit the svm_linear model with testing data
  
  ### Input: 
  ###  - the fitted svm_linear model using training data
  ###  -  processed features from testing images 
  ### Output: prediction labels
  
  ### load libraries
  library("e1071")
  pred <- predict(fit_train, data.test)
  
  return(pred)
}





test.xgb <- function(fit_train, data.test){
  
  ### Fit the Xgboost model with testing data
  
  ### Input: 
  ###  - the fitted logistic model using training data
  ###  -  processed features from testing images 
  ### Output: prediction labels
  
  ### load libraries
  library("xgboost")
  
  pred <- predict(fit_train, data.matrix(data.test))
  
  return(pred)
}


test.lg <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: prediction labels
  
  ### load libraries
  library("nnet")
  
  pred <- predict(fit_train, dat_test)
  
  return(pred)
}
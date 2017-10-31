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
    n_trees = 2000
    interaction_depth = 3
    shrinkage = 0.01
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



train.svm_linear<-function(train.data, par = NULL){
  
  ### Train a SVM classifier with linear kernel using features of training images
  
  ### Input:
  ### - training data including features of training images 
  ###                       and class labels of training images
  ### Output:
  ### - training svm_linear model specification
  
  ### load libraries
  library("e1071")
  
  ### train with linear RBF
  
  if(is.null(par)){
    cost = 5
    gamma = 100
  }
  else{
    cost = par$cost
    gamma = par$gamma
  }
  
  fit_svm_linear <- svm(y~., data = train.data, 
                     kernel = "linear", scale = FALSE,
                     cost = cost, gamma = gamma)
  return(fit_svm_linear)
}

train.lg <- function(data.train, par){
  
  ### Train a multinomial logistic regression classifier using features of training images
  
  ### Input:
  ### - training data including features of training images 
  ###                       and class labels of training images
  ### Output:
  ### - training logistic model specification
  
  ### load libraries
  library("nnet")
  
  ### train with logistic
  
  if(is.null(par)){
    maxit <- 100
  } 
  else {
    maxit <- par$maxit
  }
  
  fit_lg <- multinom(y ~ ., data = train.data, MaxNWts = 20000, maxit = maxit)
  
  return(fit_lg)
}


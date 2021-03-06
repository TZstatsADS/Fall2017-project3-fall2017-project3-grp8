########################
### Cross Validation ###
########################

cv.gbm <- function(dat_train, par, K){
  
  n <- nrow(dat_train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold))) 
  
  err.gbm = rep(NA, K)  

  for (i in 1:K) {
    fit_gbm = train.gbm(train.data = dat_train[s != i,], par = par)
    
    pred = test.gbm(fit_train = fit_gbm, data.test = dat_train[s == i, -1])
    err.gbm[i] = mean(pred != dat_train[s == i, 1])
  }

  return(c(mean(err.gbm),sd(err.gbm)))
}


cv.svm_rbf <- function(dat_train, par.ranges, K){
  # tune svm with multiple classes using the one-versus-one approach
  tune.out = tune(svm, train.x = dat_train[, -1], train.y = dat_train[, 1], 
                  kernel = "radial",
                  scale = FALSE, 
                  ranges = par.ranges, 
                  tunecontrol = tune.control(cross = K))
  
  best.para = tune.out$best.parameters 
  smallest.cv_error = tune.out$best.performance
  performance.tune <- tune.out$performances
  
  return(list(best.par = best.para, 
              smallest.cv_error = smallest.cv_error, 
              performances = performance.tune))
}


cv.xgb <- function(dat_train, K, par){
  
  xgb <- xgb.cv(data = data.matrix(dat_train[, -1]),
                label = as.numeric(dat_train[, 1]) -1,
                eta = par$eta,
                max_depth = par$max_depth,
                nrounds = 100,
                gamma = par$gamma,
                colsample_bytree = 1,
                min_child_weight = par$min_child_weight,
                subsample = par$subsample,
                nfold = K,
                num_class = 3,
                early_stopping_rounds = 100,
                metrics = "merror",
                objective = "multi:softmax",
                stratified = TRUE)
  iter <- xgb$best_iteration
  cv.err <- xgb$evaluation_log[iter, ]$test_merror_mean
  cv.sd <- xgb$evaluation_log[iter, ]$test_merror_std
  return (list(iter = iter, cv_error = cv.err, cv_sd = cv.sd))
}


cv.lg <- function(dat_train, par, K){
  
  n <- dim(dat_train)[1]
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    
    train.data <- dat_train[s != i,]
    data.test <- dat_train[s == i, -1]
    y.test <- dat_train[s == i, 1]
    
    fit_log <- train.lg(train.data, par)
    pred <- test.lg(fit_log, data.test)  
    cv.error[i] <- mean(pred != y.test)  
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
  
}
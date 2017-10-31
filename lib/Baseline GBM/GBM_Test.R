library(gbm)

test_baseline_gbm = function(data,
                              model,
                              n_trees) {
  
  predBST = predict(model,
                    n.trees=n_trees, 
                    newdata=data,
                    type='response')
  
  p.predBST <- apply(predBST, 1, which.max) - 1
  
  return(p.predBST)
  
}
library(gbm)

train_baseline_gbm = function(data,
                               n_trees,
                               interaction_depth,
                               shrinkage) {
  
  model = gbm(y ~ .,
              data = data,
              distribution = 'multinomial',
              n.trees = n_trees,
              interaction.depth = interaction_depth,
              shrinkage = shrinkage)
  
  iter = gbm.perf(model, method = "OOB", plot.it = FALSE)
  
  return(list(model = model, iter = iter))
  
}

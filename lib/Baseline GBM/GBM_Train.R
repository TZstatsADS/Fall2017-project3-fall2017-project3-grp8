library(gbm)

train_baseline_gbm(data,
                   n_trees,
                   interaction_depth,
                   shrinkage) {
  
  model = gbm(y ~ .,
              data = data,
              distribution = 'multinomial',
              n.trees = n_trees,
              interaction.depth = interaction_depth,
              shrinkage = shrinkage)
  
  return(model)
  
}

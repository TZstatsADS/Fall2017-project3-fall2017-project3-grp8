feature<-function(img_dir, export=TRUE){
  
  ### Construct process features for training/testing images
  ### HOG: calculate the Histogram of Oriented Gradient for an image
  
  ### Input: a directory that contains images ready for processing
  ### Output: an .RData file contains processed features for the images
  
  
  ### load libraries
  library("EBImage")
  library("OpenImageR")
  
  dir_names <- list.files(img_dir)
  n_files <- length(dir_names)
  
  ### calculate HOG of images
  dat <- matrix(NA, n_files, 54) 
  for(i in 1:n_files){
    img <- readImage(paste0(img_dir,  dir_names[i]))
    dat[i,] <- HOG(img)
  }
  
  ### output constructed features
  if(export){
    save(dat, file=paste0("../output/HOG.RData"))
  }
  return(dat)
}

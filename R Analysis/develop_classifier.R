# Read in Required Libraries
suppressMessages(library(MVA))
suppressMessages(library(corrplot))
suppressMessages(library(reshape2))
suppressMessages(library(MASS))
suppressMessages(library(e1071))
suppressMessages(library(biotools))
suppressMessages(library(class))

# Bug fix for large plots
# graphics.off()
# par("mar")
# par(mar=c(1,1,1,1))

# Specify the constants
latent_dim = 32
set.seed(314159)

##########################################################
# Helper Function for Error Generation
##########################################################
apparent_error = function(preds, actual) {
  cm <- confusionmatrix(actual, preds)
  return(list("cm" = cm, "ae" = sum(diag(cm))/length(preds)))
}

##########################################################
# Specify the Type of Analysis that should be Done
##########################################################
# Data Files
file_pres <- c("modvae")#, "betavae_2", "betavae_3", "betavae_4","betavae_5", 
               #"betavae_10", "betavae_100", "betavae_250", "betavae_500", "vae")


# Specify the general dimensionality reduction process
pre_pca = TRUE   # Conduct a PCA on the latent vectors?
concat = FALSE   # Concatenate the vectors?
abs_diff = TRUE  # Join the vectors using an absolute difference?
post_pca = TRUE # Conduct a PCA on the joined latent vectors?

# Specify which classifiers to use
use_lda = FALSE   # Use an LDA classifier?
use_qda = FALSE   # Use a QDA classifier?
use_svm = FALSE   # Use an SVM for classification?
use_lr = FALSE    # Use logistic regression for classification?
use_knn = TRUE   # Use a KNN for classification?

# Specify whether looking for diagnostic output as well
diagnostic = TRUE  # i.e. to generate scree plots, etc.

# KNN Parameter Specification
total <- 100    # Number of replicate trials
try <- 25       # Max 'n' to try up until

# Top Values Selected from Cross Validation
prepca_pcs <- c(5, 14, 2, 12, 13, 13, 9, 5, 5, 12)
postpca_pcs_abs <- c(7, 11, 4, 11, 11, 13, 12, 9, 8, 12)
postpca_pcs_con <- c(5, 14, 2, 12, 12, 13, 9, 5, 5, 12)
postpca_pcs_w_pre_con <- c(3, 6, 2, 12, 5, 13, 9, 1, 3, 6)
postpca_pcs_w_pre_abs <- c(1, 3, 1, 5, 5, 6, 3, 2, 2, 4)

knn_n <- c(9, 2, 3, 2, 1, 4, 2, 3, 1, 1)
knn_n_pre <- c(2, 5, 3, 2, 2, 3, 1, 1, 1, 1)
knn_n_post_abs <- c(2, 5, 1, 4, 1, 2, 2, 2, 2, 1)
knn_n_post_con <- c(2, 2, 1, 2, 2, 2, 2, 2, 2, 4)
knn_n_pre_post_abs <- c(10, 7, 3, 3, 5, 3, 3, 10, 1, 6)
knn_n_pre_post_con <- c(2, 2, 2, 5, 1, 2, 4, 1, 6, 1)

for (fn in 1:length(file_pres)) {
  file_pre<-file_pres[fn]
  print(paste("Running for:", file_pre))
  data_file_1 = paste(file_pre, "_train_1.dat", sep="") # Data files should have pair ID as first column
  data_file_2 = paste(file_pre, "_train_2.dat", sep="") # and response (equal or not) as second column
  t_data_file_2 = paste(file_pre, "_test_1.dat", sep="") 
  t_data_file_1 = paste(file_pre, "_test_2.dat", sep="")
  
  # Specify the parameters necessary
  num_pcs_pre = prepca_pcs[fn] # Number of Principal Components to Keep
  if (pre_pca) { 
    num_pcs_post = postpca_pcs_w_pre_abs[fn] 
    if (concat) {
      num_pocs_post <- postpca_pcs_w_pre_con[fn]
    }
  } else {
    num_pcs_post = postpca_pcs_abs[fn] 
    if (concat) {
      num_pcs_post <- postpca_pcs_con[fn]
    }
  } # Number of Principal Components to Keep
  
  # Number of neighbours in the KNN classification
  if(pre_pca && post_pca) { 
    n_knn <- knn_n_pre_post_pca_abs[fn]
    if (concat) {
      n_knn <- knn_n_pre_post_pca_con[fn]
    }
  } else if (pre_pca) { 
    n_knn <- knn_n_pre[fn] 
  } else if (post_pca) { 
    n_knn <- knn_n_post_abs[fn]
    if (concat) {
      n_knn <- knn_n_post_con[fn]
    }
  } else { n_knn <- knn_n[fn] }
  
  ##########################################################
  # Check for validitity of input parameters
  ##########################################################
  if (! concat && ! abs_diff) {
    stop("A method for combining the vectors must be specified.")
  } else if (concat && abs_diff) {
    stop("You cannot use both concatenation and absolute differencing.")
  }
  
  if (pre_pca &&
      (! is.numeric(num_pcs_pre) || num_pcs_pre <= 0 || num_pcs_pre > latent_dim)) {
    stop("To use pre-PCA you must specify a valid number of principal components.")
  }
  
  if (post_pca &&
      (! is.numeric(num_pcs_post) || num_pcs_post <= 0 || num_pcs_post > latent_dim)) {
    stop("To use PCA you must specify a valid number of principal components.")
  }
  
  
  ##########################################################
  # Run the analysis
  ##########################################################
  # Load in the training Data
  data_1 <- read.table(data_file_1, header=T)
  data_2 <- read.table(data_file_2, header=T)
  data_l <- rbind(data_1, data_2)
  full_data <- merge(data_1, data_2, by=c(1, 2)) # Must be id + resp in first 2 columns
  
  # Load in the test Data
  t_data_1 <- read.table(t_data_file_1, header=T)
  t_data_2 <- read.table(t_data_file_2, header=T)
  t_data_l <- rbind(t_data_1, t_data_2)
  t_full_data <- merge(t_data_1, t_data_2, by=c(1, 2)) # Must be id + resp in first 2 columns
  
  if (diagnostic) {
    # Generate a Correlation and Pairs Plots for the Data
    corrplot(cor(data_l[,-c(1,2)]),type = "upper", method="square", tl.pos = "n",
                   title="Correlation Plot", mar=c(0,0,1,0))
    # pairs(data_l, main="Pairs Plot (All Observations)")
  }

  # Conduct the pre-PCA if necessary
  if (pre_pca) {
    obs_data_pca <- prcomp(data_l[,-c(1,2)], scale=T, center=T) # Conduct the pre-PCA
    
    # Generate scree plots if necessary
    if (diagnostic) {
      plot(x = obs_data_pca$sdev^2, type="o", main = paste("Scree Plot (pre-PCA)", file_pre))
      abline(a = mean(obs_data_pca$sdev^2),b = 0) # Add the Average
    }

    # Use the obs_data_pca to predict the PCA for d1 and d2
    data_1[,-c(1,2)] <- predict(obs_data_pca, newdata=data_1[,-c(1,2)])
    data_1 <- data_1[1:(2+num_pcs_pre)]
    data_2[,-c(1,2)] <- predict(obs_data_pca, newdata=data_2[,-c(1,2)])
    data_2 <- data_2[1:(2+num_pcs_pre)]
    data_l <- rbind(data_1, data_2)
    full_data <- merge(data_1, data_2, by=c(1, 2))
    
    t_data_1[,-c(1,2)] <- predict(obs_data_pca, newdata=t_data_1[,-c(1,2)])
    t_data_1 <- t_data_1[1:(2+num_pcs_pre)]
    t_data_2[,-c(1,2)] <- predict(obs_data_pca, newdata=t_data_2[,-c(1,2)])
    t_data_2 <- t_data_2[1:(2+num_pcs_pre)]
    
    t_full_data <- merge(t_data_1, t_data_2, by=c(1, 2))
  }
  
  # If concat is used, simply use full_data/obs_data as it stands
  # if abs_diff is used, then compute the absolute differences
  if (abs_diff) {
    full_data <- cbind(full_data[,c(1,2)], data_1[,-c(1,2)]-data_2[,-c(1,2)])
    data_l <- full_data
    
    t_full_data <- cbind(t_full_data[,c(1,2)], t_data_1[,-c(1,2)]-t_data_2[,-c(1,2)])
  }
  
  # Conduct the post-PCA if necessary
  if (post_pca) {
    obs_data_pca_post <- prcomp(data_l[,-c(1,2)], scale=T, center=T) 

    # Generate scree plots if necessary
    if (diagnostic) {
      plot(x = obs_data_pca_post$sdev^2, type="o", main = paste("Scree Plot (post-PCA)", file_pre))
      abline(a = mean(obs_data_pca_post$sdev^2),b = 0) # Add the Average
    }

    if (concat) {
      # Use the obs_data_pca to predict the PCA for d1 and d2
      data_1[,-c(1,2)] <- predict(obs_data_pca_post, newdata=data_1[,-c(1,2)])
      data_1 <- data_1[1:(2+num_pcs_post)]
      data_2[,-c(1,2)] <- predict(obs_data_pca_post, newdata=data_2[,-c(1,2)])
      data_2 <- data_2[1:(2+num_pcs_post)]
      
      full_data <- merge(data_1, data_2, by=c(1, 2))
      
      t_data_1[,-c(1,2)] <- predict(obs_data_pca_post, newdata=t_data_1[,-c(1,2)])
      t_data_1 <- t_data_1[1:(2+num_pcs_post)]
      t_data_2[,-c(1,2)] <- predict(obs_data_pca_post, newdata=t_data_2[,-c(1,2)])
      t_data_2 <- t_data_2[1:(2+num_pcs_post)]
      
      t_full_data <- merge(t_data_1, t_data_2, by=c(1, 2))
    } else {
      tmp_pca <- predict(obs_data_pca_post, newdata=full_data[,-c(1,2)])
      t_tmp_pca <- predict(obs_data_pca_post, newdata=t_full_data[,-c(1,2)])
      full_data <- cbind(full_data[,c(1,2)], tmp_pca[,1:num_pcs_post])
      t_full_data <- cbind(t_full_data[,c(1,2)], t_tmp_pca[,1:num_pcs_post])
      names(full_data)[-c(1,2)] <- colnames(tmp_pca)[1:num_pcs_post]
      names(t_full_data) <- names(full_data)
    }
  }
  
  # Begin the Classification Methods
  fmla = as.formula(paste("resp~", paste(names(full_data)[-c(1,2)], collapse="+")))
  
  if (use_lda) {
    # Use the LDA Model
    model.lda <- lda(fmla, data=full_data)
    model.lda.preds <- predict(model.lda, newdata=t_full_data)$class
    model.lda.ae <- apparent_error(model.lda.preds, t_full_data[,2])
    
    print(paste("LDA Error: ", model.lda.ae$ae))
  }
  
  if (use_qda) {
    # Use the QDA Model
    model.qda <- qda(fmla, data=full_data)
    
    model.qda.preds <- predict(model.qda, newdata=t_full_data)$class
    model.qda.ae <- apparent_error(model.qda.preds, t_full_data[,2])
    
    print(paste("QDA Error: ", model.qda.ae$ae))
  }
  
  if (use_svm) {
    # Use the QDA Model
    model.svm <- svm(fmla, data=full_data)
    
    model.svm.preds <- predict(model.svm, newdata=t_full_data)
    model.svm.ae <- apparent_error(round(model.svm.preds), t_full_data[,2])
    
    print(paste("SVM Error: ", model.svm.ae$ae))
  }
  
  if (use_lr) {
    # Use the LR Model
    model.lr <- glm(fmla, family=binomial(link='logit'), data=full_data)
    
    model.lr.preds <- predict(model.lr, newdata=t_full_data)
    model.lr.ae <- apparent_error(round(model.lr.preds), t_full_data[,2])
    
    print(paste("LR Error: ", model.lr.ae$ae))
  }
  
  if (use_knn) {
   # Use the KNN Model
   if (diagnostic) {
     
     knn.cv.err <- NULL
     knn.cv.sd <- NULL
     
     for (i in 1:try){
       print(paste("Checking for N=", i))
       print("")
       temp <- NULL
       
       pb <- txtProgressBar(min = 0, max = total, style = 3)
        
       for (j in 1:total){
         temp <- c(temp,mean(knn.cv(full_data[,-c(1,2)], cl=full_data[,2], k=i) != full_data[,2]))
         setTxtProgressBar(pb, j)
       }
       knn.cv.err <- c(knn.cv.err, mean(temp))
       knn.cv.sd <- c(knn.cv.sd,sd(temp))
     }
     
     plot(y=knn.cv.err, x=seq(1,try,by=1), xlim=c(1,25), type="n", main=paste("CV Results for", file_pre))
     lines(y=knn.cv.err, x=seq(1,try,by=1), col="red")
   }
    
   model.knn.preds <- knn(train = data.frame(full_data[,-c(1,2)]), test = data.frame(t_full_data[,-c(1,2)]),
                          cl = full_data[,2], k=n_knn)
   model.knn.ae <- apparent_error(model.knn.preds, t_full_data[,2])
   
   print(paste("KNN Error: ", model.knn.ae$ae))
  }
}

























































data <- read.table("C:\\Users\\sjsty\\Desktop\\Masters\\Final Programs\\wine.data", sep = ",", header = FALSE)

data$wine1 <- as.numeric(data$V1 == 1)
data$wine2 <- as.numeric(data$V1 == 2)
data$wine3 <- as.numeric(data$V1 == 3)
data <- data[, -1]

softmax <- function(x) {
  exp_x <- exp(x-max(x))  # Subtract max(x) for numerical stability
  probabilities <- exp_x / sum(exp_x)
  return(probabilities)
}


#Activation function between layers
activation <- function(x){
  # tmp = NULL
  # eps = 0.0001
  # if(x<0){
  #   tmp = eps*x
  # } else{
  #   tmp=x
  # }
  # return(x)
  value = tanh(x)
  return(value)
}

#Function to create probability distribution
score <- function(x,epoch){
  #Uniform
  #rank = 1
  #Inverse
  #rank = min(1/x*epoch,1000000)
  #exponential
  #rank = exp(-x)
  rank= x^2
  #rank = 1/cosh(x^2)
  return(rank)
}

#Note that when using a single variable case nrow(y) needs to be changed to length(y)
nearestneighbours <- function(x,y,n){
  distances = NULL
  for(i in 1:nrow(y)){
    distances[i] = sum(abs(x-y[i,]))
  }
  samples = NULL
  for (i in 1:n){
    samples[i] = which.min(distances)
    distances[samples[i]] = 10000000000000
  }
  return(samples)
  #lst = sort(distances, index.return = TRUE, decreasing = FALSE)
  #return(lst$ix[1:n])
}


main <- function(epochs){
  #Architechture of neural network:
  number_of_inputs <<- 13
  number_of_hiddennodes1 <<- 100
  number_of_outputs <<- 3
  
  #Random start points for the weights
  w1 <<- matrix(rnorm((number_of_inputs+1)*number_of_hiddennodes1,0,0.01), ncol = (number_of_inputs+1))
  w2 <<- matrix(rnorm((number_of_hiddennodes1+1)*number_of_outputs,0,0.01), ncol = (number_of_hiddennodes1+1))
  
  w1check <<- w1
  w2check <<- w2
  
  
  
  # Split the dataset into 80% train and 20% test
  train_indices <- sample(1:nrow(data),floor(0.8*nrow(data)))
  
  # Create train and test datasets
  traindata <<- data[train_indices, ]
  testdata <<- data[-train_indices, ]
  
  traindata$wine1 <<- ifelse(traindata$wine1 == 1, log(0.9999999), log(0.0000001)-log(2))
  traindata$wine2 <<- ifelse(traindata$wine2 == 1, log(0.9999999), log(0.0000001)-log(2))
  traindata$wine3 <<- ifelse(traindata$wine3 == 1, log(0.9999999), log(0.0000001)-log(2))
  
  #Normalizing the inputs for the x variable training set
  for(i in 1:number_of_inputs){
    traindata[,i] <<- -1 + 2*(traindata[,i]-min(traindata[,i]))/(max(traindata[,i])-min(traindata[,i]))
  }
  
  
  #Normalizing the inputs for the x variable training set
  for(i in 1:number_of_inputs){
    testdata[,i] <<- -1 + 2*(testdata[,i]-min(testdata[,i]))/(max(testdata[,i])-min(testdata[,i]))
  }
  
  testdata$y1 <<- ifelse(testdata$wine1 == 1, log(0.9999999), log(0.0000001)-log(2))
  testdata$y2 <<- ifelse(testdata$wine2 == 1, log(0.9999999), log(0.0000001)-log(2))
  testdata$y3 <<- ifelse(testdata$wine3 == 1, log(0.9999999), log(0.0000001)-log(2))
  
  
}

splitsLog <- function(epochs){
  #Number of splits in the dataset
  #splits <<- c(10,5,4,3,2,rep(1,epochs-5))
  splits <<- c(rep(1,epochs))
  
}

#########################################################################################################
# Run algorithm
#########################################################################################################

#Number of epochs:
epochs = 5
#Create log to store all MSE values
accuracy_log <<- matrix(NA, ncol = epochs, nrow =0)
mse_log <<- matrix(NA, ncol = epochs, nrow=0)

for(v in 1:200){
  par(mfrow=c(1,2))
  
  main(epochs)
  splitsLog(epochs)
  meansquarederror = NULL
  accuracy = NULL
  for(epoch in 1:epochs){
    
    #Randomly sample the sampling set
    traindata = traindata[sample(1:nrow(traindata)),]
    
    #Size of sampling set:
    batchsize=floor(nrow(traindata)/splits[epoch])
    delta = floor((batchsize/number_of_outputs*0.25))
    for(k in 1:splits[epoch]){
      #These values move along the data sets so that we see new observations throughout the process
      a = 1 + batchsize*(k-1)
      b = batchsize*k
      data1 = traindata[a:b,]
      
      #Tracking the observations and all their hidden layer values during the feed forward process.
      #This just saves us from having to calculate any inverses
      samplevalues = NULL
      
      for(i in 1:batchsize){
        singlesample = NULL
        x1 = c(1,as.numeric(data1[i,1:number_of_inputs]))
        y1 = w1check %*% x1
        x2 = c(1,sapply(y1,activation))
        y2 = w2check %*% x2
        truey2 = as.numeric(data1[i,(number_of_inputs+1):(number_of_inputs+number_of_outputs)])
        singlesample = c(x1,y1,y2,truey2)
        samplevalues = rbind(samplevalues,singlesample)
      }
      
      print("fed forward")
      #Set all the initial matrices
      A1=0
      A2=0
      B1=0
      B2=0
      #Updating the samples one at a time at this point using the sampling set to bootstrap the hiddenlayer
      for(i in 1:batchsize){
        #Reading in the data for the update sample
        updatex1 = c(1,as.numeric(data1[i,1:number_of_inputs]))
        updatey2 = as.numeric(data1[i,-c(1:number_of_inputs)])
        #Finding the N nearest neighbours with respect to the L1 norm for the inputs
        # distances = rowSums(abs(updatex1[2:(number_of_inputs+1)]-samplevalues[,2:(number_of_inputs+1)]))
        # samples = sort(distances, index.return=TRUE, decreasing=FALSE)
        # value = max(delta+1-epoch,8)
        # nn = samples$ix[1:value]
        
        correctclass = NULL
        
        for(i in 1:batchsize){
          
          if (identical(samplevalues[i,(ncol(samplevalues)-number_of_outputs+1):ncol(samplevalues)], updatey2)) {
            correctclass <- rbind(correctclass,samplevalues[i,])
          }
        }
        
        
        nn = nearestneighbours(updatex1[2:(number_of_inputs+1)], correctclass[,2:(number_of_inputs+1)],max(delta-epoch,5))
        
        tmpvalues = correctclass[nn,]
        
        #Finding the errors of that point with respect to the L1 norm for the outputs
        errors = NULL
        for(y in 1:length(nn)){
          max_index <- which.max(updatey2)
          errors[y] = epoch*sum(updatey2*log(softmax(tmpvalues[y,(ncol(tmpvalues)-5):(ncol(tmpvalues)-3)])))
        }
        tmpvalues = cbind(tmpvalues, errors)
        
        #Subtracting off the min error so we don't get floating point errors at a later step
        regularized = tmpvalues[,ncol(tmpvalues)]-min(tmpvalues[,ncol(tmpvalues)])
        #Applying these errors to the scoring function and creating a probability distribution out of them
        probs = sapply(regularized,score,epoch=epoch)
        probs = probs/sum(probs)
        #Sampling a point from the N nearest neighbours using that probability distribution
        point = sample(1:length(nn),1,prob = probs)
        
        #Plugging the hidden layer values of that sample point into the middle of the update sample
        updatey1 = as.numeric(tmpvalues[point,(2+number_of_inputs):(number_of_inputs+number_of_hiddennodes1+1)]) 
        updatex2 = c(1,sapply(updatey1,activation))
        
        A1 = A1 + updatex1 %*% t(updatex1)
        A2 = A2 + updatex2 %*% t(updatex2)
        B1 = B1 + updatey1 %*% t(updatex1)
        B2 = B2 + updatey2 %*% t(updatex2)
      }
      if(epoch == 1){
        if(k==1){
          A1check = A1/batchsize
          A2check = A2/batchsize
          B1check = B1/batchsize
          B2check = B2/batchsize
        } else{
          A1check = A1check/2 + A1/(2*batchsize)
          A2check = A2check/2 + A2/(2*batchsize)
          B1check = B1check/2 + B1/(2*batchsize)
          B2check = B2check/2 + B2/(2*batchsize)
        }
      } else{
        A1check = A1check/2 + A1/(2*batchsize)
        A2check = A2check/2 + A2/(2*batchsize)
        B1check = B1check/2 + B1/(2*batchsize)
        B2check = B2check/2 + B2/(2*batchsize)
      }
      
      w1=w1check
      w2=w2check
      
      
      gain1 = 2/(max(eigen(A1check)$values)+min(eigen(A1check)$values))
      gain2 = 2/(max(eigen(A2check)$values)+min(eigen(A2check)$values))
      
      for(s in 1:100000){
        w1 = w1 + gain1*(B1check-w1%*%A1check)
        w2 = w2 + gain2*(B2check-w2%*%A2check)
      }
      
      #w1check = w1check/2 +w1/(2)
      #w2check = w2check/2 +w2/(2)
      w1check = w1
      w2check = w2
      
      
      
    }
    #Graph results
    correct <- 0
    mse <- 0
    for(s in 1:nrow(testdata)){
      x1 = as.numeric(c(1,testdata[s,1:number_of_inputs]))
      X1tmp = w1check %*% x1
      x2 = c(1,sapply(X1tmp,activation))
      y2 = w2check %*% x2
      output= softmax(y2)
      max_index <- which.max(output)
      result_vector <- c(rep(0,number_of_outputs))
      result_vector[max_index] <- 1
      
      if (identical(result_vector, as.numeric(testdata[s,(number_of_inputs+1):(number_of_inputs+number_of_outputs)]))){
        correct <- correct + 1
      }
      mse = mse + sum((y2-testdata[s,(ncol(testdata)-number_of_outputs):ncol(testdata)])^2)
    }
    accuracy = c(accuracy, correct/nrow(testdata)*100)
    meansquarederror = c(meansquarederror, mse/nrow(testdata))
    cat(correct, "out of", nrow(testdata), ":", correct/nrow(testdata)*100, "\n\n")
    plot(accuracy, type = "l", main = "Accuracy for Dataset")
    plot(meansquarederror, type = "l", main = "Mean Squared Error for Dataset")
  }
  
  cat("Iteration: ", v, " complete")
  accuracy_log = rbind(accuracy_log,accuracy)
  mse_log = rbind(mse_log,meansquarederror)
  print(colMeans(accuracy_log))
  print(colMeans(mse_log))
  
  write.csv(accuracy_log, "C:\\Users\\sjsty\\Desktop\\Masters\\Final Programs\\Wine_3step_CE2.csv", row.names=FALSE)
}
print(rowMeans(accuracy_log))
print(rowMeans(mse_log))
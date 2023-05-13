library(matrixStats)

#Activation function between layers
activation <- function(x){
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
  rank = exp(-x^2*epoch)
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

#Data generating function to estimate
data_gen <- function(x){
  return(2*x[,1]^2*x[,2]-6*x[,1]*x[,3])
}


main <- function(epochs){
  #Architechture of neural network:
  number_of_inputs <<- 3
  number_of_hiddennodes1 <<- 100
  number_of_outputs <<- 1
  
  #Random start points for the weights
  w1 <<- matrix(rnorm((number_of_inputs+1)*number_of_hiddennodes1,0,0.5), ncol = (number_of_inputs+1))
  w2 <<- matrix(rnorm((number_of_hiddennodes1+1)*number_of_outputs,0,0.5), ncol = (number_of_hiddennodes1+1))
  
  w1check <<- w1
  w2check <<- w2
  
  #Sample Size
  smp_size <<- 6000
  
  valid_size <<- 1000
  
  x1 = runif(smp_size,-5,5)
  x2 = runif(smp_size,-2,2)
  x3 = runif(smp_size,0,4)
  x = cbind(x1,x2,x3)
  
  train_data <<- data.frame(x,data_gen(x))
  
  x1 = runif(valid_size,-5,5)
  x2 = runif(valid_size,-2,2)
  x3 = runif(valid_size,0,4)
  x = cbind(x1,x2,x3)
  
  test_data <<- data.frame(x,data_gen(x))
  
  #Splitting the train data up
  xtrain <<- as.matrix(train_data[,1:number_of_inputs],nrow = nrow(train_data))
  ytrain <<- as.matrix(train_data[,(number_of_inputs+1)],nrow = nrow(train_data))
  
  
  #Normalizing the inputs for the x variable training set
  for(i in 1:ncol(xtrain)){
    xtrain[,i] <<- -1 + 2*(xtrain[,i]-min(xtrain[,i]))/(max(xtrain[,i])-min(xtrain[,i]))
  }
  
  traindata <<- cbind(xtrain,ytrain)
  
  #Splitting the test data up
  xtest <<- as.matrix(test_data[,1:number_of_inputs], nrow = nrow(test_data))
  ytest <<- as.matrix(test_data[,(number_of_inputs+1)], nrow = nrow(test_data))
  
  #Normalizing the inputs for the x variable validation set
  for(i in 1:ncol(xtest)){
    xtest[,i] <<- -1 + 2*(xtest[,i]-min(xtest[,i]))/(max(xtest[,i])-min(xtest[,i]))
  }
  
  testdata <<- cbind(xtest,ytest)
}

splitsLog <- function(epochs){
  #Number of splits in the dataset
  splits <<- c(10,5,4,3,2,rep(1,epochs-5))
  #splits <<- c(rep(1,epochs))
  
  #Create log to store all MSE values
  mse_log <<- matrix(NA, nrow = epochs, ncol=0)
}

#########################################################################################################
# Run algorithm
#########################################################################################################

for(v in 1:100){
  #Number of epochs:
  epochs = 50
  delta = 40
  
  main(epochs)
  splitsLog(epochs)
  meansquarederror = NULL
  for(epoch in 1:epochs){
    
    #Randomly sample the sampling set
    traindata = traindata[sample(1:nrow(traindata)),]
    
    #Size of sampling set:
    batchsize=floor(smp_size/splits[epoch])
    
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
        truey2 = as.numeric(data1[i,(number_of_inputs+1)])
        singlesample = c(x1,y1,y2,truey2)
        samplevalues = rbind(samplevalues,singlesample)
      }
      
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
        #distances = rowSums(abs(updatex1[2:(number_of_inputs+1)]-samplevalues[,2:(number_of_inputs+1)]))
        #samples = sort(distances, index.return=TRUE, decreasing=FALSE)
        #value = max(delta+1-epoch,8)
        #nn = samples$ix[1:value]
        
        nn = nearestneighbours(updatex1[2:(number_of_inputs+1)], samplevalues[,2:(number_of_inputs+1)],max(40-epoch,8))
        
        tmpvalues = samplevalues[nn,]
        
        #Finding the errors of that point with respect to the L1 norm for the outputs
        errors = NULL
        for(y in 1:length(nn)){
          errors[y] = sum(abs(updatex1[2:(number_of_inputs+1)] - tmpvalues[y, 2:(number_of_inputs+1)])) + abs(updatey2-tmpvalues[y,(ncol(tmpvalues)-1)])
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
      
      #w1check = w1check +(w1 -w1check)/2
      #w2check = w2check +(w2 -w2check)/2
      w1check = w1
      w2check = w2
    }

    
 
    #Graph results
    sqerror = 0
    for(s in 1:nrow(xtest)){
      x1 = as.numeric(c(1,xtest[s,]))
      X1tmp = w1check %*% x1
      x2 = c(1,sapply(X1tmp,activation))
      output = w2check %*% x2
      sqerror = sqerror + (output-ytest[s])^2
    }
    meansquarederror = c(meansquarederror, sqerror/valid_size)
    print(sqerror/valid_size)
    plot(meansquarederror, type = "l", main = "Mean Squared Error for Dataset")
  }
  
  cat("Iteration: ", v, " complete")
  mse_log = cbind(mse_log,meansquarederror)
  print(rowMeans(mse_log))
}
print(rowMeans(mse_log))


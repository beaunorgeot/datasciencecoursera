
library(caret)
library(randomForest)

setwd("/Users/bnorgeot/datasciencecoursera/titanic")

inTraining <- read.table("train.csv", sep= ",", header=TRUE)
testSet <- read.table("test.csv", sep= ",", header=TRUE) #these are unlabeled and used to upload to kaggle.


# CHECK THE PROPORTIONS of survived died.If 1 outcome is far more common than the other you need to change your approach
#This is an important step because if the proportion was smaller than 15%, 
# it would be considered a rare event and would be more challenging to model.
prop.table(table(inTraining$Survived)) #survived is outcome 1 = .38

set.seed(42)
#split the training set into a train/test set (which I'm calling validate), so that testSet is a true hold out set
inTrain <- createDataPartition(inTraining$Survived, p = 0.8, list = F)
trainSet <- inTraining[inTrain,]
validSet <- inTraining[-inTrain,]

#remove features that I don't want to use for prediction. I only do this on trainSet, I want validSet to look exactly like testSet, except that outcome is labeled
library(dplyr)
#trainSet <- trainSet %>% select(-c(PassengerId, Name,Ticket,Cabin))

#glmnet can't deal w/factors, it can only handle numerics. Solve this problem by turning factor variables into dummyVars
#make dummies
#titanicDummy <- dummyVars("~.", data=trainSet, fullRank=F)
#trainSet <- as.data.frame(predict(titanicDummy,trainSet))
#trainSet$Survived <- as.factor(trainSet$Survived)

#validSet <- as.data.frame(predict(titanicDummy,validSet))

# TRAIN SOME MODELs
myControl <- trainControl(method = "repeatedcv", #use cross-validation
                        number = 10, repeats = 5 )

#glmnetFit <- train(Survived ~., data = trainSet, method = "glmnet", preProcess = "knnImpute", trControl = control, metric ="Accuracy") #.80
#formula <- c(Survived ~ Pclass + Sex + Fare + SibSp + Embarked + Parch)
#form. <- list("Pclass"+"Sex"+"Fare"+"SibSp"+"Embarked"+"Parch")
#modRF1 <- train(form = formula, method = "rf", data = trainSet, trControl = myControl)
trainSet$Survived <- as.factor(trainSet$Survived) # outcome should be factor for classification
modRF <- train(Survived ~Pclass + Sex + Fare + SibSp + Embarked + Parch, method = "rf", data = trainSet, trControl = myControl)
modGBM <- train(Survived ~Pclass + Sex + Fare + SibSp + Embarked + Parch, method = "gbm", data = trainSet, trControl = myControl, verbose = F) 
modLDA <- train(Survived ~Pclass + Sex + Fare + SibSp + Embarked + Parch, method = "lda", data = trainSet, preProcess=c("center","scale"), trControl = myControl) 

modRF #.815
modGBM #.81
modLDA #.78

# STACK THE PREDICTIONS
# make predictions
predRF <- predict(modRF,validSet)
predGBM <- predict(modGBM, validSet)
predLDA <- predict(modLDA, validSet)

# Fit a model that combines all (both of the predictors)
predDF <- data.frame(predRF,predGBM,predLDA,Survived=validSet$Survived)
predDF$Survived <- as.factor(predDF$Survived)
#train a new model on the predictions
validSet$Survived <- as.factor(validSet$Survived)
combModFit <- train(Survived ~.,method="rf",data=predDF, trControl = myControl)
predComb <- predict(combModFit,validSet)

# Get/compare the accuracies for the 3 singular models and the 1 combined model (4 models)
c1 <- confusionMatrix(predRF, validSet$Survived)$overall[1]
c2 <- confusionMatrix(predGBM, validSet$Survived)$overall[1]
c3 <- confusionMatrix(predLDA, validSet$Survived)$overall[1]
c4 <- confusionMatrix(predComb, validSet$Survived)$overall[1]
print(paste(c1, c2, c3, c4)) #"0.786516853932584 0.786516853932584 0.775280898876405 0.820224719101124"
# Combined did 4-5% better than any individual model

#Compare the predictions from each model to eachother, and color by the true answer to compare how close they were
qplot(predRF,predGBM,colour=Survived,data=validSet) #this just produced 2 data points wtf? duh, there's only 2 outcomes. 
# This diagnostic would be useful for regression, but isn't all that useful for classification

#check correlation modelCor 
predDF1 <- predDF
predDF1$predRF <- as.numeric(predDF1$predRF)
predDF1$predGBM <- as.numeric(predDF1$predGBM)
predDF1$predLDA <- as.numeric(predDF1$predLDA)
predDF1$Survived <- as.numeric(predDF1$Survived)
cor(predDF1) # this works, and it checks the correlation of the predictions LDA and GBM are highly correlated, the others are not 

#Here is the caret method for checking the correlation of the models
modCor <- modelCor(resamples(list(RF = modRF, GBM = modGBM, LIN = modLDA))) # we see no correlation between models

#Try some other models for poops
modGLMNET <- train(Survived ~Pclass + Sex + Fare + SibSp + Embarked + Parch, method = "glmnet", data = trainSet, preProcess=c("center","scale"), trControl = myControl) 
# see what models are available
names(getModelInfo())
modSVM <- train(Survived ~Pclass + Sex + Fare + SibSp + Embarked + Parch, method = "svmRadial", data = trainSet, preProcess=c("center","scale"), trControl = myControl) 
modGLMNET #.79
modSVM #.79
modCor2 <- modelCor(resamples(list(GLMNET = modGLMNET, SVM = modSVM))) #no correlation b/tween models

predGLMNET <- predict(modGLMNET, validSet)
predSVM <- predict(modSVM, validSet)
#join these predictions to the old combined set to create a super set built w/5 models, all w/similar accuracy
predDF2 <- data.frame(predDF,predSVM,predGLMNET)
#check correlation across all models
modCor3 <- modelCor(resamples(list(RF = modRF, GBM = modGBM, LIN = modLDA,GLMNET = modGLMNET, SVM = modSVM))) #No High Correlations
#New Stack, Use GBM to bring all pieces together
combModFit2 <- train(Survived ~.,method="gbm",data=predDF2, trControl = myControl)
predComb2 <- predict(combModFit2,validSet)
# Check accuracy of everything
c1 <- confusionMatrix(predRF, validSet$Survived)$overall[1]
c2 <- confusionMatrix(predGBM, validSet$Survived)$overall[1]
c3 <- confusionMatrix(predLDA, validSet$Survived)$overall[1]
c4 <- confusionMatrix(predComb, validSet$Survived)$overall[1]
c5 <- confusionMatrix(predGLMNET, validSet$Survived)$overall[1]
c6 <- confusionMatrix(predSVM, validSet$Survived)$overall[1]
c7 <- confusionMatrix(predComb2, validSet$Survived)$overall[1]
print(paste(c1, c2, c3, c4, c5, c6, c7))
# Accuracy for comb2 was LOWER than comb1, overfitting?

# Does combining in an RF change anything? YES
combModFit3 <- train(Survived ~.,method="rf",data=predDF2, trControl = myControl)
predComb3 <- predict(combModFit3,validSet)
c8 <- confusionMatrix(predComb3, validSet$Survived)$overall[1] #.831

# Change testSet$Survived to factor!!
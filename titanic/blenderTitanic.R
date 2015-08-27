#this assumes/depends that you have already run preProcessTitanic!!!!
#this ignores good practices about train, test, validate

library(caret)
library(doMC) #doMC package to take advantage of parallel processing with multiple cores
registerDoMC(cores=4)
library(plyr) #many caret functions use this. Loading plyr after dplyr causes all kinds of problems
library(dplyr)

# CHECK THE PROPORTIONS of survived died.If 1 outcome is far more common than the other you need to change your approach
#This is an important step because if the proportion was smaller than 15%, 
# it would be considered a rare event and would be more challenging to model.
prop.table(table(trainSet$Survived)) #survived is outcome 1 = .38

#Training
myControl <- trainControl(method = "repeatedcv", #use cross-validation
                          number = 10, repeats = 5 )

# The general approach is to split the data into 3 pieces ensembleData, blenderData, testData
splitData <- createDataPartition(trainSet$Survived, p=0.5, list = F)
ensembleData <- trainSet[splitData,]
blenderData <- trainSet[-splitData,]

modRF <- train(Survived ~., data = trainSet, method = "rf", trControl = myControl)
modGBM <- train(Survived ~., data = trainSet, method = "gbm", trControl = myControl, verbose = F) 
modLDA <- train(Survived ~., data = trainSet,method = "lda", preProcess=c("center","scale"), trControl = myControl) 
modGLMNET <- train(Survived ~., data = trainSet, method = "glmnet", preProcess=c("center","scale"), trControl = myControl) 
modSVM <- train(Survived ~., data = trainSet, method = "svmRadial", preProcess=c("center","scale"), trControl = myControl)
#perhaps it's time to drop lda and replace it w/LogitBoost

#after models are trained, predict on both blenderData and testingData
# Do this to harvest the predictions from both data sets and add those predictions as new features to the same data sets
# 5 models means 5 new columns to blender and testing

blenderData$rf_prob <- predict(modRF, blenderData)
blenderData$gbm_prob <- predict(modGBM, blenderData)
blenderData$lda_prob <- predict(modLDA, blenderData)
blenderData$glmnet_prob <- predict(modGLMNET, blenderData)
blenderData$svm_prob <- predict(modSVM, blenderData)

testSet$rf_prob <- predict(modRF, testSet)
testSet$gbm_prob <- predict(modGBM, testSet)
testSet$lda_prob <- predict(modLDA, testSet)
testSet$glmnet_prob <- predict(modGLMNET, testSet)
testSet$svm_prob <- predict(modSVM, testSet)

# Training the final blending model on the old data and new predictions
blendFit <- train(Survived ~., data=blenderData, method="gbm", trControl=myControl) #this basically did the same thing as the stacked model. RF got all the importance

testSet$Survived <- predict(blendFit, testSet)

#Generate SUBMISSION
blendedSubmission <- testSet %>% select(PassengerId,Survived) #blendedSubmission <- subset(testSet, select= c(PassengerId,Survived))
# kaggle requires that the outcome is just (0,1)
blendedSubmission1 <- stackedSubmission %>% mutate(Survived = ifelse(Survived == "Survived", 1,0)) 
blendedSubmission %>% group_by(Survived) %>% summarise(n()) #Died 265, Survived 153.  #Same titanic result as rf, and stacked. .7703                                                                                                                                                                          

#write resulting predictions w/only the two columns to csv
write.table(blendedSubmission1,file = "blendedSubmission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
#Same titanic result as rf, and stacked. .7703
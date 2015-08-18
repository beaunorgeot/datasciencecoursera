
library(caret)
library(randomForest)

trainSet <- read.table("train.csv", sep= ",", header=TRUE)
testSet <- read.table("test.csv", sep= ",", header=TRUE)

trainSet$Survived<-factor(trainSet$Survived, levels=c(0,1), labels=c("Died", "Survived"))
trainSet$Pclass<-factor(trainSet$Pclass, levels=c(1,2,3), labels=c("First class", "Second class", "Third class"))

# Train the model using the RF algorithm
model <- train(Survived ~ Pclass + Sex + Fare + SibSp + Embarked + Parch, 
               # Survived is a function of the variables chosen
               data = trainSet,
               method = "rf", #use the RF algorithm
               trControl = trainControl(method = "cv", #use cross-validation
                                        number = 5))
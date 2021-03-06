
#setwd("/Users/bnorgeot/datasciencecoursera/titanic")

library(shiny)
library(RcppEigen)
library(e1071)
library(caret)
library(randomForest)
library(shinyapps)
source("simTitanic.R") #this is the script used to build the model/generate the predictions

# #Features: Survived ~ Pclass + Sex + Fare + SibSp + Embarked + Parch
shinyServer(function(input, output) {
  
  output$didHeSurvive <- renderText({
    
    toTest <- data.frame(Sex=input$Sex, Pclass=input$Pclass, SibSp=input$SibSp, Fare=input$Fare, Embarked=input$Embarked, Parch=input$Parch)
    toTest$Pclass<-factor(toTest$Pclass, levels=c(1,2,3), labels=c("First class", "Second class", "Third class"))
    #
    Prediction <- predict(model, toTest)
    write.csv(Prediction, "prediction.csv")
    return(as.character(Prediction))
  })
})

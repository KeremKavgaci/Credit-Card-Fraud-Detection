# Gerekli K??t??phanelerin Y??klenmesi
library(ranger)         # Random Forest algoritmas??
library(caret)          # Model de??erlendirme ve makine ????renmesi ara??lar??
library(data.table)     # Veri okuma ve veri i??leme i??in h??zl?? bir k??t??phane
library(caTools)        # Veri setini e??itim/test olarak ay??rmak i??in
library(pROC)           # ROC e??risi ve AUC hesaplamalar??
library(rpart)          # Karar a??ac?? olu??turmak i??in
library(rpart.plot)     # Karar a??ac??n?? g??rselle??tirmek i??in
library(neuralnet)      # Yapay sinir a???? modeli olu??turmak i??in
library(gbm)            # Gradient Boosting modeli i??in

# Veri Setini Okuma ve ??nceleme
creditcard_data <- fread("C:/Users/Asus/Desktop/Projeler/Credit Card Fraud Detection/creditcard.csv")
dim(creditcard_data)               # Veri setinin boyutlar??
head(creditcard_data, 6)           # ??lk 6 sat??r?? yazd??r
tail(creditcard_data, 6)           # Son 6 sat??r?? yazd??r
table(creditcard_data$Class)       # S??n??f da????l??m?? (0: normal, 1: sahte i??lem)
summary(creditcard_data$Amount)    # 'Amount' s??tununun ??zet istatistikleri

# 'Amount' S??tununun Normalize Edilmesi ve 'Time' S??tununun Kald??r??lmas??
creditcard_data$Amount <- scale(creditcard_data$Amount)    # Miktar verisinin standartla??t??r??lmas??
NewData = creditcard_data[,-c(1)]                          # 'Time' s??tununun kald??r??lmas??

# E??itim ve Test Verisinin Olu??turulmas?? (%80 E??itim, %20 Test)
set.seed(123)
data_sample <- sample.split(NewData$Class, SplitRatio = 0.80)
train_data <- subset(NewData, data_sample == TRUE)
test_data <- subset(NewData, data_sample == FALSE)

# Model Performans??n?? De??erlendirmek i??in Fonksiyon
evaluate_model <- function(predictions, true_labels, model_name) {
  predictions <- as.factor(predictions)
  true_labels <- as.factor(true_labels)
  cm <- confusionMatrix(predictions, true_labels, positive = "1")  # Kar??????kl??k matrisi
  cat("\n======================", model_name, "======================\n")
  print(cm)
  cat("Accuracy :", round(cm$overall["Accuracy"], 4), "\n")
  cat("Precision:", round(cm$byClass["Precision"], 4), "\n")
  cat("Recall   :", round(cm$byClass["Recall"], 4), "\n")
  cat("F1 Score :", round(cm$byClass["F1"], 4), "\n")
}

# Lojistik Regresyon Modeli
Logistic_Model <- glm(Class ~ ., data = train_data, family = binomial())  # Modelin kurulmas??
lr.prob <- predict(Logistic_Model, test_data, type = "response")          # Olas??l??k tahminleri
lr.class <- ifelse(lr.prob > 0.5, 1, 0)                                   # S??n??f tahminleri
roc(test_data$Class, lr.prob, plot = TRUE, col = "blue")                  # ROC e??risinin ??izilmesi
evaluate_model(lr.class, test_data$Class, "Logistic Regression")          # De??erlendirme
plot(Logistic_Model)                                                      # Lojistik regresyon i??in grafikler

# Karar A??ac?? Modeli
decisionTree_model <- rpart(Class ~ ., data = train_data, method = 'class')   # Modelin kurulmas??
pred.dt <- predict(decisionTree_model, test_data, type = 'class')             # S??n??f tahminleri
rpart.plot(decisionTree_model)                                                # Karar a??ac??n??n g??rselle??tirilmesi
auc_dt <- auc(test_data$Class, predict(decisionTree_model, test_data)[,2])    # AUC de??eri hesaplan??r
print(auc_dt)                                                                 # AUC de??eri yazd??r??l??r
evaluate_model(pred.dt, test_data$Class, "Decision Tree")                     # De??erlendirme

# Yapay Sinir A???? Modeli
ANN_model <- neuralnet(Class ~ ., data = train_data, linear.output = FALSE)   # Sinir a???? modelinin e??itilmesi
plot(ANN_model)                                                               # Sinir a????n??n yap??s??n??n ??izilmesi
pred.ann.prob <- compute(ANN_model, test_data)$net.result                     # Olas??l??k tahminleri
pred.ann <- ifelse(pred.ann.prob > 0.5, 1, 0)                                 # S??n??f tahminleri
auc_ann <- auc(test_data$Class, pred.ann.prob)                                # AUC d??eri hesaplan??r
print(auc_ann)                                                                # AUC de??eri yazd??r??l??r
evaluate_model(pred.ann, test_data$Class, "Artificial Neural Network (ANN)")  # De??erlendirme

# Gradient Boosting Modeli (GBM)
system.time(                                                            # Modelin ??al????ma s??resinin ??l????lmesi
  model_gbm <- gbm(Class ~ ., 
                   data = rbind(train_data, test_data),                # E??itim + test verisi
                   distribution = "bernoulli", 
                   n.trees = 1500, 
                   interaction.depth = 5, 
                   shrinkage = 0.005, 
                   n.minobsinnode = 750, 
                   bag.fraction = 0.7, 
                   train.fraction = 0.75)
)
gbm.iter <- gbm.perf(model_gbm, method = "test")                             # En iyi a??a?? say??s??n??n belirlenmesi
summary(model_gbm, n.trees = gbm.iter)                                       # En iyi a??a?? say??s?? i??in de??i??ken ??nem s??ras??
gbm.pred <- predict(model_gbm, newdata = test_data, n.trees = gbm.iter)      # Tahminlerin yap??lmas??
roc(test_data$Class, gbm.pred, plot = TRUE, col = "red")                     # ROC e??risi ??izimi
gbm.class <- ifelse(gbm.pred > 0.5, 1, 0)                                    # S??n??f tahminleri
evaluate_model(gbm.class, test_data$Class, "Gradient Boosting Model (GBM)")  # De??erlendirme
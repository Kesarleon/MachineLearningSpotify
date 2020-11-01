library(spotifyr)
library(tidyverse)

Sys.setenv("SPOTIFY_CLIENT_ID" = '8dabc4c0795a4b0ea6544e0f21967c2a')
Sys.setenv("SPOTIFY_CLIENT_SECRET" = '4f8a2d3aeb8d484e82278746a442a3e8')
playlist_username <- 'Masterpiece in progress'
playlist_uris <- c('2tiye2096SXAS4uMnFsUFP')

# Credenciales de conexión para la API de Spotify
get_spotify_access_token(client_id = Sys.getenv("SPOTIFY_CLIENT_ID"),
                         client_secret = Sys.getenv("SPOTIFY_CLIENT_SECRET"))

# Obtener la playlist desde la API
masterpiece <- get_playlist_audio_features(playlist_username, playlist_uris)


num_cols = c("acousticness",
             "danceability",
             "energy",
             "instrumentalness",
             "key",
             "liveness",
             "loudness",
             "speechiness",
             "tempo",
             "time_signature",
             "track.duration_ms",
             "track.popularity",
             "valence")

Num_master <- masterpiece %>%
  dplyr::select(all_of(num_cols))

# Checando correlación
library(corrplot)
M<-cor(Num_master)
corrplot(M, method="circle")

# Preprocesamiento de variables
boxplot(Num_master)
Num_Data_Norm <- sapply(Num_master, scale)
boxplot(Num_Data_Norm)
summary(Num_Data_Norm)


# Regresión Logística ----

logist_master <- masterpiece %>%
  dplyr::select(num_cols, added_by.id)

logist_master <- logist_master %>% 
                  mutate(added_by.id = as.factor(added_by.id))

model <- glm(added_by.id ~.,family=binomial(link='logit'),data=logist_master)

summary(model)

anova(model, test="Chisq")


# Separar datos en train y test
library(caret)

set.seed(123)
training.samples <- logist_master$added_by.id %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- logist_master[training.samples, ]
test.data <- logist_master[-training.samples, ]

library(MASS)
# Ajustar el modelo
model <- glm(added_by.id ~., data = train.data, family = binomial) %>%
  stepAIC(trace = TRUE)
# Estadísticos del modelo final
summary(model)
# Predicción sobre los datos test
probabilities <- model %>% predict(test.data, type = "response")
test.data$prediction <- probabilities

# Buscar el threshold óptimo
library(pROC)
roc_obj <- roc(test.data$added_by.id, test.data$prediction)
roc_obj
plot(roc_obj)
coords(roc_obj, "best", "threshold", transpose = T)

predicted.classes <- ifelse(probabilities > 0.88, "22xma24ys4af6t67voqa263qa", "kesarleon")


# Precisión del modelo
mean(predicted.classes==test.data$added_by.id)

test.data %>% 
  ggplot(aes(x = seq(1:length(test.data$prediction)), y = sort(probabilities), color = test.data$added_by.id)) +
  geom_point() +
  theme(legend.position = 'bottom',
        legend.box = 'vertical')

summary(probabilities)

# Random Forest ----

train.data %>% dplyr::select(added_by.id) %>% table()
test.data %>% dplyr::select(added_by.id) %>% table()

library(rpart)
treeimb <- rpart(added_by.id ~ ., data = train.data)
pred.treeimb <- predict(treeimb, newdata = test.data)

library(ROSE)
accuracy.meas(test.data$added_by.id, pred.treeimb[,2])

roc.curve(test.data$added_by.id, pred.treeimb[,2])

# Oversamplig
data_balanced_over <- ovun.sample(added_by.id ~ ., 
                                  data = train.data, 
                                  method = "over",
                                  seed = 1)$data

table(data_balanced_over$added_by.id)

# Undersampling
data_balanced_under <- ovun.sample(added_by.id~ ., 
                                   data = train.data, 
                                   method = 'under', 
                                   seed = 1)$data

table(data_balanced_under$added_by.id)

# balanceo
data_balanced_both <- ovun.sample(added_by.id~ ., 
                                  data = train.data, 
                                  method = 'both', 
                                  p = 0.5)$data

table(data_balanced_both$added_by.id)

# ROSE
data.rose <- ROSE(added_by.id ~ ., 
                  data = train.data, 
                  seed = 1)$data

table(data.rose$added_by.id)

# Probando los 4 métodos de balanceo
tree.rose <- rpart(added_by.id ~ ., data = data.rose)
tree.over <- rpart(added_by.id ~ ., data = data_balanced_over)
tree.under <- rpart(added_by.id ~ ., data = data_balanced_under)
tree.both <- rpart(added_by.id ~ ., data = data_balanced_both)

# Predicción sobre test data
pred.tree.rose <- predict(tree.rose, newdata = test.data)
pred.tree.over <- predict(tree.over, newdata = test.data)
pred.tree.under <- predict(tree.under, newdata = test.data)
pred.tree.both <- predict(tree.both, newdata = test.data)

# AUC ROSE
roc.curve(test.data$added_by.id, pred.tree.rose[,2])

# AUC Oversampling
roc.curve(test.data$added_by.id, pred.tree.over[,2])

# AUC Undersampling
roc.curve(test.data$added_by.id, pred.tree.under[,2])

# AUC Both
roc.curve(test.data$added_by.id, pred.tree.both[,2])

ROSE.holdout <- ROSE.eval(added_by.id ~ ., 
                          data = train.data, 
                          learner = rpart, 
                          method.assess = "holdout", 
                          extr.pred = function(obj)obj[,2], 
                          seed = 1)
ROSE.holdout


# Gradient Boosting

library(caret) # for model-building
library(DMwR) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculations

# Parámetros del modelo

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

# Target

levels(train.data$added_by.id) <- c('l', 'k')
levels(test.data$added_by.id) <- c('l', 'k')

set.seed(1234)

orig_fit <- train(added_by.id ~ .,
                  data = train.data,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl)

#AUC
roc(test.data$added_by.id, 
    predict(orig_fit, test.data, type = "prob")[, 2]) %>% auc()


# Modelo ponderado (los pesos deben sumar 1)

model_weights <- ifelse(train.data$added_by.id == "l",
                        (1/table(train.data$added_by.id)[1]) * 0.5,
                        (1/table(train.data$added_by.id)[2]) * 0.5)

# Semilla
ctrl$seeds <- orig_fit$control$seeds

# Entrenamiento del modelo (buscando maximizar ROC)

weighted_fit <- train(added_by.id ~ .,
                      data = train.data,
                      method = "gbm",
                      verbose = FALSE,
                      weights = model_weights,
                      metric = "ROC",
                      trControl = ctrl)

# Mismo modelo con downsampling

ctrl$sampling <- "down"

down_fit <- train(added_by.id ~ .,
                  data = train.data,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl)

# Mismo modelo con oversampling

ctrl$sampling <- "up"

up_fit <- train(added_by.id ~ .,
                data = train.data,
                method = "gbm",
                verbose = FALSE,
                metric = "ROC",
                trControl = ctrl)

# Modelo con técnica SMOTE (Synthetic Minority Over-sampling Technique)

ctrl$sampling <- "smote"

smote_fit <- train(added_by.id ~ .,
                   data = train.data,
                   method = "gbm",
                   verbose = FALSE,
                   metric = "ROC",
                   trControl = ctrl)

# Evaluando modelos en test

model_list <- list(original = orig_fit,
                   weighted = weighted_fit,
                   down = down_fit,
                   up = up_fit,
                   SMOTE = smote_fit)

# Función para ROC

test_roc <- function(model, data) {
  
  roc(data$added_by.id,
      predict(model, data, type = "prob")[, 2])
  
}

model_list_roc <- model_list %>%
  map(test_roc, data = test.data)

model_list_roc %>%
  map(auc)


results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in model_list_roc){
  
  results_list_roc[[num_mod]] <- 
    tibble(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(model_list)[num_mod])
  
  num_mod <- num_mod + 1
  
}

results_df_roc <- bind_rows(results_list_roc)

# Plot ROC curve for all 5 models
library(viridis)
custom_col <- c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")

ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc) +
  geom_line(aes(color = model), size = 1) +
#  scale_color_manual(values = custom_col) +
  scale_color_viridis(discrete = T, option = 'D') +
    geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)

# Construimos nuevas features y las agregamos al modelo

library(lubridate)

lanzamiento <- ymd(masterpiece$track.album.release_date)
lanzamiento1  <- interval(min(lanzamiento, na.rm = T),lanzamiento)
lanzamiento1 <- lanzamiento1 %/% days(1)
lanzamiento1 <- ifelse(is.na(lanzamiento1), mean(lanzamiento1, na.rm = T), lanzamiento1)
mean(lanzamiento1)
masterpiece <- masterpiece %>% mutate(lanzamiento.track = lanzamiento1)


masterpiece %>% dplyr::select(track.id, lanzamiento.track, added_by.id) %>% 
  ggplot(aes(x = seq(1:length(lanzamiento.track)), y = sort(lanzamiento.track), color = added_by.id)) +
  geom_point(show.legend = FALSE) 


masterpiece <- masterpiece %>% mutate(added_at.trans = ymd_hms(added_at),
                                      added_at.fecha = as_date(added_at.trans),
                                      added_at.hora = hour(added_at.trans)
)

lanzamiento2 <- masterpiece$added_at.fecha
lanzamiento2  <- interval(min(lanzamiento2, na.rm = T),lanzamiento2)
lanzamiento2 <- lanzamiento2 %/% days(1)

masterpiece <- masterpiece %>% mutate(added_at.dias = lanzamiento2)



masterpiece %>% dplyr::select(track.id, added_at.hora, added_by.id) %>% 
  ggplot(aes(x = seq(1:length(added_at.hora)), y = sort(added_at.hora), color = added_by.id)) +
  geom_point(show.legend = FALSE)


num_cols2 = c(
              "acousticness",
              "added_at.dias", 
              "added_at.hora", 
              "danceability",
              "energy",
              "instrumentalness",
              "lanzamiento.track",
              "liveness",
              "loudness",
              "speechiness",
              "tempo",
              "track.duration_ms",
              "track.popularity",
              "valence"
              )

Num_master <- masterpiece %>%
  dplyr::select(all_of(num_cols2))

# Correlación y preprocesamiento
library(corrplot)
M<-cor(Num_master)
corrplot(M, method="circle")


boxplot(Num_master)
Num_Data_Norm <- sapply(Num_master, scale)   #Normalized Numerical Data
boxplot(Num_Data_Norm)
summary(Num_Data_Norm)

logist_master <- masterpiece %>%
  dplyr::select(num_cols2, added_by.id)

logist_master <- logist_master %>% mutate(added_by.id = as.factor(added_by.id))


library(caret)

set.seed(123)
training.samples <- logist_master$added_by.id %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- logist_master[training.samples, ]
test.data <- logist_master[-training.samples, ]

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

levels(train.data$added_by.id) <- c('l', 'k')
levels(test.data$added_by.id) <- c('l', 'k')

set.seed(1234)

orig_fit <- train(added_by.id ~ .,
                  data = train.data,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl)

ggplot(orig_fit)

model_weights <- ifelse(train.data$added_by.id == "l",
                        (1/table(train.data$added_by.id)[1]) * 0.5,
                        (1/table(train.data$added_by.id)[2]) * 0.5)

ctrl$seeds <- orig_fit$control$seeds

weighted_fit <- train(added_by.id ~ .,
                        data = train.data,
                        method = "gbm",
                        verbose = FALSE,
                        weights = model_weights,
                        metric = 'Sens',
                        trControl = ctrl)

ggplot(weighted_fit)

weighted_fit_2 <- train(added_by.id ~ .,
                        data = train.data,
                        method = "gbm",
                        verbose = FALSE,
                        weights = model_weights,
                        metric = 'Spec',
                        trControl = ctrl)
ggplot(weighted_fit_2)


prediccion <- predict(weighted_fit, newdata = test.data, type = 'prob')
prediccion_2 <- predict(weighted_fit_2, newdata = test.data, type = 'prob')
observed.classes <- test.data$added_by.id

roc.curve(test.data$added_by.id, prediccion[,2])
roc.curve(test.data$added_by.id, prediccion_2[,2])


roc_obj <- roc(test.data$added_by.id, prediccion[,2])
roc_obj
plot(roc_obj)
coords(roc_obj, "best", "threshold", transpose = T)

roc_obj <- roc(test.data$added_by.id, prediccion_2[,2])
roc_obj
plot(roc_obj)
coords(roc_obj, "best", "threshold", transpose = T)

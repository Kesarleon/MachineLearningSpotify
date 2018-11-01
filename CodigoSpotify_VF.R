library(readr)            # Importa datos
library(dplyr)            # Manipulacion de datos
library(stringr)          # Manipulación de cadenas
library(missForest)       # Imputación de valores de variables no disponibles
library(ggplot2)          # Graficación de variables
library(rpart)            # Árboles
library(rpart.plot)       # Graficación de árboles


# Leer la base (Aquí va la ruta donde está guardado el dataset)

Compa<-read.csv("C:/Users/julio.hernandezl/Documents/Cesar/Kesar/Machine Learning/Spotify/ComparadoCompleto_RF.csv")


# Establecer variables categóricas

Compa <- mutate(Compa, id = factor(id)
                , key = factor(key)
                , mode = factor(mode)
                , time_signature = factor(time_signature)
                , List = factor(List))

# Mostrar resumen de estadísticos para cada variable del dataset
summary(Compa)


# Graficar distribución de acousticness (probabilidad de que la canción use instrumentos acústicos).

variable1<-Compa$acousticness

boxplot(variable1~Compa$List, sub="Acousticness")

ggplot(Compa, aes(variable1, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Acousticness")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")




# Graficar distribución de duration_ms.

variable2<-Compa$duration_ms

boxplot(variable2~Compa$List, sub="Duration")

ggplot(Compa, aes(variable2, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Duration")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")




# Graficar distribución de danceability (fuerza y regularidad del ritmo).

variable3<-Compa$danceability

boxplot(variable3~Compa$List, sub="Danceability")

ggplot(Compa, aes(variable3, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Danceability")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")




# Graficar distribución de energy (rapidez y "ruidosidad" de la canción).

variable4<-Compa$energy

boxplot(variable4~Compa$List, sub="Energy")

ggplot(Compa, aes(variable4, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Energy")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")



# Graficar distribución de instrumentalness (Canciones mas instrumentales que vocales).

variable5<-Compa$instrumentalness

boxplot(variable5~Compa$List, sub="Instrumentalness")

ggplot(Compa, aes(variable5, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Instrumentalness")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")



# Graficar distribución de liveness (Probabilidad de que la canción se haya grabado en vivo).

variable6<-Compa$liveness

boxplot(variable6~Compa$List, sub="Liveness")

ggplot(Compa, aes(variable6, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Liveness")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")



# Graficar distribución de loudness (Volumen promedio de la canción).

variable7<-Compa$loudness

boxplot(variable7~Compa$List, sub="Loudness")

ggplot(Compa, aes(variable7, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Loudeness")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")



# Graficar distribución de speechiness (qué tanto se habla en la canción).

variable8<-Compa$speechiness

boxplot(variable8~Compa$List, sub="Speechiness")

ggplot(Compa, aes(variable8, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Speechiness")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")



# Graficar distribución de tempo (duración promedio del ritmo).

variable9<-Compa$tempo

boxplot(variable9~Compa$List, sub="Tempo")

ggplot(Compa, aes(variable9, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Tempo")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")



# Graficar distribución de valence ("positividad" de la canción).

variable10<-Compa$valence

boxplot(variable10~Compa$List, sub="Valence")

ggplot(Compa, aes(variable10, fill = Compa$List, colour = Compa$List, show.legend=TRUE)) +
  geom_density(alpha = 0.1) + 
  xlab("Valence")+ 
  theme(legend.title=element_blank()) + 
  labs(fill="Lista")


# Filtrar las variables que se van a usar para el modelo

Compa2<-Compa[,-c(1,3,7,9,15,16,17,18)]

names(Compa2)


# Correr un solo árbol

set.seed(200)

arbol <- rpart(List ~ tempo+valence+energy, method = "class",Compa2)
t_pred <- predict(arbol,Compa2,type="class")

# Mostrar Matriz de Confusión del árbol

(MatrizConfusion <- table(Compa2$List,t_pred))

# Precisión del árbol

(Precision <- sum(diag(MatrizConfusion))/sum(MatrizConfusion))


# Correr Random Forest

rf <- randomForest(x = select(Compa2,-c(List)), 
                   y = Compa2$List, 
                   importance = TRUE, ntree = 100)


# Mostrar Importancia de variables

varImpPlot(rf)


# Precision del random forest

(Precision_rf<-sum(diag(rf$confusion[,-3]))/sum(rf$confusion[,-3]))

fourfoldplot(rf$confusion[,-3])

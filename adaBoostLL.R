################################################################################
#                           ADABOOST simple training                           #
################################################################################

## leer el archivo https://www.kaggle.com/uciml/student-alcohol-consumption
df <- read.table('./data/ensembles/student-por.csv',header = TRUE,sep = ';' )
library(dplyr)
str(df)

# la variable absences indica el número de veces que se ha faltado a clase
# (int: de 0 a 93) está variable la agrupamos en 5 factores 
df$absences <- ntile(df$absences, 5)  



# Las columnas G1 y G2 se refieren a las notas de los periodos 1 y 2, nos
# vamos a limitar a predecir los aprobados de las notas finales G3
df$G1 <- NULL
df$G2 <- NULL

# En el algoritmo que vamos a emplar es necesario que las variables sean
# categóricas. Esto no es así necesariamente, pero, por simplicidad hemos
# empleado esta generalidad.
#
# Convertimos el resto de las variables numéricas a factores. No tenemos
# variables numéricas continuas sino categóricas ordinales, con lo que en esta
# generalización la pérdida de información es asumible.
#
nums <- sapply(df, is.numeric)
columns <- colnames(df)
numericColumns <- as.list(columns[nums]) # lista de las columnas numéricas
print(numericColumns)
numericColumns[numericColumns == 'G3'] <- NULL # target
numericColumns <- unlist(numericColumns)
df[,numericColumns] <-   lapply(df[,numericColumns] , factor)  
str(df)


df[df$G3 < 10,'G3'] <- -1 ## Suspenso, las notas son sobre 20
df[df$G3 >= 10,'G3'] <- 1 ## Aprobado
df$G3 <-   sapply(df$G3 , factor)  

# Veamos el balance de clases
aprobados <- df[df$G3 == 1,]
suspensos <- df[df$G3 == -1,]
c(dim (aprobados)[1], dim (suspensos)[1])
# Las clases están muy desbalanceadas, vamos a balancearlas
set.seed(1)
aprobados <- aprobados[sample(nrow(aprobados), nrow(suspensos)), ]
# ahora tenemos las clases balanceadas:
df <- rbind (aprobados, suspensos)
dim (df[df$G3 == 1,])[1]
dim (df[df$G3 == -1,])[1]

# Generar las variables dummies: Nuestro árbol es binario, con lo que vamos a
# necesitar un one-hot encoding en las variables para que funcione. Para ello, 
# tiramos de caret
library (caret)
dummies <- dummyVars(G3 ~ ., data = df)
dummyfied <- predict(dummies, newdata = df)
head (dummyfied)

# Columna con las variables a predecir
targets <- df$G3




dat <- data.frame (dummyfied)
dat$G3 <- targets
varnames <- colnames (dummyfied)




# Tenemos un tamaño reducido del dataset pero es suficiente para poder jugar con
# él y mostrar el funcionamiento del algoritmo


################################################################################
################################################################################
#                        Árboles binarios con pesos                            #
################################################################################
################################################################################

# A continuación comenzamos con las funciones para la construcción del árbol de
# decisión



####### IMPORTANTE: Paso de AdaBoost #######
# La selección se hace teniendo en cuenta los pesos omega_i de las samples.
# En las dos funciones siguientes es donde se tiene en cuenta este hecho: se mide
# en la siguiente y se invoca en variableToSplit

################################################################################
#                   Decisiones dentro del algoritmo                            #
################################################################################
# Devuelve el peso de los errores que minimiza cada decisión:
# A la hora de tener un nodo en un arbol, el algoritmo se va a quedar con 
# la predicción que haga menor el peso de los errores.



erroresPesados <- function (labels, weights) {
  # Ojo, el peso de los errores al clasificar todos como positivos es la suma 
  # de los pesos de etiqueta negativa y viceversa
  pesosPositivos <- sum (weights[labels == -1]) # suma pesos erróneos si elegimos clase = +1
  pesosNegativos <- sum (weights[labels == 1])  # suma pesos erróneos si elegimos clase = -1
  if (pesosNegativos < pesosPositivos)
    return (c(pesosNegativos, -1))
  else
    return (c(pesosPositivos, +1))
}

# Testing
# example_labels = c (-1, -1, 1, 1, 1)
# example_data_weights = c(1., 2., .5, 1., 1.)
# erroresPesados(example_labels, example_data_weights) == as.vector (c(2.5, -1))




################################################################################
#                   Split en variable                                          #
################################################################################
# A la hora de que el árbol vaya generándose, ha de elegir en cada nodo la 
# variable por la que hacer el split. Para ello tiene que evaluar los pesos
# de cada instancia de los datos. Quitando la función anterior y la necesidad
# de los parámetros "peso", esta función no tiene especial interés dentro del
# boosting
# 
variableToSplit <- function (df, features, target, weights){
  
  # Calculamos los errores al hacer split con cada variable:
  errores <- sapply (1: length (features), function(x) {
    # en el lado izquierdo, los valores para cada feature en los que
    # la variable toma el valor 0 y 1 en la derecha
    leftSplit <- df[df[,x] == 0,]
    rightSplit <- df[df[,x] == 1,]
    
    # Los pesos de cada rama
    leftWeights <- weights[df[,x] == 0]
    rightWeights <- weights[df[,x] == 1]
    
    # Computamos los errores de cada rama: cuánto nos equivocamos 
    # (de manera pesada) si clasificamos así haciendo split por esta feature
    # ñapa de longitud de errores
    if (length (leftWeights) > 0) {
      # Errores en la hoja izquierda
      leftPesoErrores <-
        erroresPesados(leftSplit[target], leftWeights)[1]
    } else {leftPesoErrores <-0 }
    
    if (length (rightWeights) > 0) {
      # Errores en la hoja derecha
      rightPesoErrores <-
        erroresPesados(rightSplit[target],rightWeights)[1]
    } else {rightPesoErrores <- 0}
    # Total de errores para la variable
    error <-  (leftPesoErrores + rightPesoErrores) / sum (weights)
    return(error)
  })
  
  # tomamos la feature que menor error ha obtenido:
  return (features[sort.int(errores, index.return = TRUE)$ix[1]])
}


# 
# frase <- c("Bucle","for","gratuito", "para","que","a","Pablo","le","sangren","los","ojos")
# 
# for (i in 1:length (frase)){
#   print(frase[i])
#   Sys.sleep(0.5)
# }


# Testing
# variableToSplit(dat, varnames, "G3", rep (1.0, dim (dummyfied)[1]))
# "failures.0"


################################################################################
#                Estructura de Hojas                                           #
################################################################################
# Una hoja va a ser una lista nombrada con valores en la variable en la que divide,
# si es o no hoja (obviamente, sí), y lo que predice. Esto se heredará para 
# todo el árbol
# 
createLeaf <- function (targets, weights){
  leaf<- list (splitting_feature = "None", is_leaf = TRUE)
  aux <- erroresPesados (targets, weights)
  leaf$prediction <- aux[2]
  return (leaf)
}



################################################################################
#                  Generado del árbol                                          #
################################################################################
# Se crea el árbol teniendo en cuenta los pesos 
# (sólo para la función de erroresPesados)
# 
createWeightedDecissionTree <- function (data, features, target, weights, 
                                         current_depth = 1, max_depth = 10){
  remainingFeatures <- features
  target_values <-  data[target]
  # Detendremos el algoritmo si:
  # - El error es 0
  # - Ya no hay más variables
  # - Se supera el límite de profundidad por árbol.
  # En cualquier caso, se genera un nodo hoja.
  #
  if (erroresPesados(target_values, weights)[1] <= 1e-15){
    return (createLeaf(target_values, weights))}
  
  if (length (remainingFeatures) == 0){
    return (createLeaf(target_values, weights))}
  
  if (current_depth > max_depth){
    return (createLeaf(target_values, weights))
  }
  
  # Selección de la variable por la que hacer split dependiendo de los pesos
  splitting_feature <- variableToSplit(data, features, target, weights)
  # Sólo se puede hacer un split por la variable elegida (el árbol es binario)
  remainingFeatures <- remainingFeatures[remainingFeatures!= splitting_feature]
  
  # Truco con las features y los espacios.
  if (!(splitting_feature %in% colnames (data)) ){
    splitting_feature <- gsub (" ", ".", splitting_feature)
  }
  
  # Siempre vamos a considerar la rama izquierda como la que tiene valor 0 en 
  # cada variable y la derecha, la que tiene 1.
  left_split <- data[data[splitting_feature] == 0,]
  right_split <- data[data[splitting_feature] == 1,]
  
  # Separamos los pesos
  left_data_weights <- weights[data[splitting_feature] == 0]
  right_data_weights <- weights[data[splitting_feature] == 1]
  
  # En caso de un corte perfecto, generamos una hoja
  if (dim(left_split)[1] == dim(data)[1]){
    return (createLeaf(left_split[target], weights))}
  if (dim(right_split)[1] == dim(data)[1]){
    return (createLeaf(right_split[target], weights))}
  
  
  # Y de manera recursiva, generamos árboles en ambas ramas
  left_tree = createWeightedDecissionTree(
    left_split, remainingFeatures, target, left_data_weights,
    current_depth + 1, max_depth)
  right_tree = createWeightedDecissionTree(
    right_split, remainingFeatures, target, right_data_weights,
    current_depth + 1, max_depth)
  
  # Vamos a devolver una lista nombrada, cuyo contenido, al margen de lo que 
  # ya contienen los nodos hoja son otras lista nombradas (cada rama)
  return (list (is_leaf= FALSE,
                prediction = "None",
                splitting_feature = splitting_feature,
                left = left_tree,
                right = right_tree))
  
}

# Auxiliar: cuenta los nodos del árbol
countNodes <- function (tree){
  if (tree$is_leaf) return (1)
  else {
    return (1 + countNodes(tree$left) + countNodes(tree$right)) 
  }
}



# Testing
# 
# example_data_weights = rep (1, dim(dat)[1])
# small_data_decision_tree =
#   createWeightedDecissionTree(dat, colnames(dat)[colnames(dat)!= "G3"], "G3",
#                               example_data_weights, max_depth=2)
# 
# countNodes(small_data_decision_tree) == 7



# Nos permite clasificar con nuestor árbol de pesos

classify <- function (tree, sample, annotate = FALSE){
  # si el árbol es hoja, tenemos la clasificación por su predicción
  if (tree$is_leaf) {
    if (annotate){
      print(paste0("Predicción: " , tree$prediction))
    }
    return (tree$prediction)
  } else{
    # en otro caso, continuamos por las ramas acorde con los valores de nuestra
    # sample
    split_feature_value = sample[tree$splitting_feature]
    if (annotate){ 
      print(paste0("Tomamos el split en la variable ", 
                   tree$splitting_feature, " con el valor ", split_feature_value))
      
    }
    # Y de manera recursiva, recorremos el árbol
    if (split_feature_value == 0){
      return (classify(tree$left, sample, annotate))
    } else {
      return (classify(tree$right, sample, annotate))
    }
  }
  
} 

# Testing
# 
# classify (small_data_decision_tree, dat[1,], TRUE)




# Hasta aquí nos hemos limitado a introducir los pesos en el algoritmo de árbol,
# pero no hay ningún recálculo de pesos. En la siguiente función computamos
# ambos pesos



################################################################################
#         Adaboost: calibrado de pesos para modelos e instancias               #
################################################################################
# Se crea el árbol teniendo en cuenta los pesos 
# (sólo para la función de erroresPesados)
# 
adaBoostonTrees <- function (data, features, target, num, depth=1){
  # Estos van a ser los pesos de las instancias. Comienzan siendo normalizados
  omega <- rep (1/dim(data)[1], dim(data)[1])
  targets <- data[target]
  # Entrenamos un árbol para cada iteración (número de modelos)
  trees <- sapply (1:num, function (x){
    print ('=====================================================')
    print (paste0 ('Iteración número ', x))
    print ('=====================================================' )
    
    tree <- createWeightedDecissionTree  (data, features, 
                                          target, weights=omega, 
                                          max_depth=depth)
    
    
    # Evaluación del árbol: generado de los pesos de las instancias
    predictions <- sapply ((1:dim(data)[1]), 
                           function (x){classify (tree, data[x,])})
    
    
    # Booleanos útiles
    is_correct <- predictions == targets
    is_wrong   <- predictions != targets
    
    
    # El error pesado es el resultado de la suma de los pesos de error 
    # entre el total
    weighted_error <- sum (omega[is_wrong])/sum (omega)
    
    
    ####### IMPORTANTE: Paso de AdaBoost #######
    # coeficiente del modelo: peso
    # Ojo, que aquí el peso no se normaliza: Estamos clasificando en -1, 1. 
    # El máximo valor que puede dar antes de tomar log es la suma de los pesos
    # y el mínimo, lo mismo, con signo negativo (1 - x). Puesto que clasificamos 
    # dependiendo del signo final, la longitud del intervalo no es importante
    alpha <- (log ((1- weighted_error)/weighted_error))/2
    
    
    ####### IMPORTANTE: Paso de AdaBoost #######
    # Ajustado de los pesos de las samples de acuerdo con la fórmula
    adjustment <- sapply (1:length (is_correct), function(x) {
      if (is_correct[x]) exp (-alpha)
      else exp (alpha)
    })
    
    aux <- omega * adjustment
    # Nota:
    # <<- actualiza el valor de forma global (fuera del apply)
    omega <<- aux/sum (aux)
    # Lista nombrada con el resultado
    resultado <- list (tree=tree, weights = alpha)
    
    return (resultado)
    
  })
  
  return (trees)
  
}

# Testing: todo funciona?
# res <- adaBoostonTrees (dat, colnames(dat)[colnames(dat)!= "G3"], "G3",
#                         num=2, depth=2)
# Ponemos un epsilon
# Ojo, que esto sólo va con la semilla
# (unlist (res["weights",]) - (c(0.6041556,0.4123396))) < 0.00001

# Y la predicción del modelo
predictAdaboost <- function (weights, trees, data, type = "class") {
  scores <- rep (0, dim(data)[1])
  
  # Obtenemos la predicción de cada modelo
  results <- sapply (1:length (trees), function (x) {
    predictions <-sapply ((1:dim(data)[1]), 
                          function (y){
                            classify (trees[[x]], data[y,])})
    
    # que va a venir multiplicada por su respectivo peso
    partialres <- weights[[x]] * predictions
    
    return (partialres)
  })
  # El total de "probabilidades" como su suma.
  probs <- rowSums (results)
  
  # Y el resultado a modo de clase
  res <- rep (-1, length (probs))
  res[probs>0] <- 1
  return(res)
}

# Testing
# predictions = predictAdaboost(res["weights",], res["tree",], dat)
# Ojo, que esto sólo va con la semilla
# predictions [c(1, 15, 35, 140)] == c(1, 1, 1, -1)


# Ahora vamos a comprobar qué tal funciona nuestro algoritmo:


################################################################################
#                                 Testing                                      #
################################################################################

# Train test split
set.seed(1) # misma semilla de antes
TrainIndex <- createDataPartition(dat$G3, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- dat[ TrainIndex,]
test <- dat[ -TrainIndex,]

dim(train)
sum(train$G3 == -1)

dim(test)
sum(test$G3 == -1)



# Iniciamos los vectores de modelos y aciertos
aciertos <- NULL
modelos <- NULL


# Veamos qué ocurre con un único árbol, de profundidad 2. Al tomar sólo un árbol
# en nuestro algoritmo, no hay pesos en juego, puesto que no se actualizan
# y todos los samples tienen el mismo valor al inicio.
res <- adaBoostonTrees (train, varnames, 'G3', num=1, depth=2)
predictions = predictAdaboost(res["weights",], res["tree",], test)

aciertoTotal = sum(predictions == test$G3) / nrow(test)
print('Predicción con 1 árbol de profundidad 2')
print(aciertoTotal) #  0.675

aciertos <- c (aciertos, aciertoTotal)
modelos <- c (modelos, "1 árbol, profundidad = 2")




## Predicción con 40 árboles de profundidad 1
res <- adaBoostonTrees (train, varnames, 'G3', num=40, depth=1)
predictions = predictAdaboost(res["weights",], res["tree",], test)


aciertoTotal = sum(predictions == test$G3) / nrow(test)
print('Predicción con 40 árboles de profundidad 1')
print(aciertoTotal) # 0.725


aciertos <- c (aciertos, aciertoTotal)
modelos <- c (modelos, "40 árboles, profundidad = 1")



## Predicción con 20 árboles de profundidad 2
res <- adaBoostonTrees (train, varnames, 'G3', num=20, depth=2)
predictions = predictAdaboost(res["weights",], res["tree",], test)


aciertoTotal = sum(predictions == test$G3) / nrow(test)
print('Predicción con 20 árboles de profundidad 2')
print(aciertoTotal) #  0.75

aciertos <- c (aciertos, aciertoTotal)
modelos <- c (modelos, "20 árboles, profundidad = 2")



## Predicción con 10 árboles de profundidad 3
res <- adaBoostonTrees (train, varnames, 'G3', num=10, depth=3)
predictions = predictAdaboost(res["weights",], res["tree",], test)


aciertoTotal = sum(predictions == test$G3) / nrow(test)
print('Predicción con 10 árboles de profundidad 3')
print(aciertoTotal) #  0.8

aciertos <- c (aciertos, aciertoTotal)
modelos <- c (modelos, "10 árboles, profundidad = 3")










################################################################################
#                         Pruebas con otros modelos                          #
################################################################################

# Ahora vamos a comparar con otros modelos de clasificación:


########################### Logistic regression ################################

folds=5
repeats=5
method <-  "repeatedcv"
myControl<- trainControl(method=method, number=folds, repeats=repeats)


mod_fit <- caret::train(G3 ~.,  data=train, method="glm",
                 family="binomial",trControl = myControl)

predictions <- predict(mod_fit, newdata=test, type="raw")


aciertoTotal = sum(predictions == test$G3) / nrow(test)
print('Predicción con Logistic Regression')
print(aciertoTotal) #  0.7


aciertos <- c (aciertos, aciertoTotal)
modelos <- c (modelos, "reg.logística")



##################################### SVM ######################################
library(kernlab)       # support vector machine 
library(pROC)	

folds=5
repeats=5
method <-  "repeatedcv"
myControl<- trainControl(method=method, number=folds, repeats=repeats)



grid <- expand.grid(sigma = c(.01, .015, 0.2),
                    C = c(0.75, 0.9, 1, 1.1, 1.25)
)
#Train and Tune the SVM
svm.tune <- caret::train(G3 ~.,data=train,
                         method = "svmRadial",
                         metric="AUC",
                         tuneGrid = grid,
                         trControl=myControl)



predictions <- predict(svm.tune, newdata=test, type="raw")

aciertoTotal = sum(predictions == test$G3) / nrow(test)
print('Predicción con SVM')
print(aciertoTotal) #  0.7

aciertos <- c (aciertos, aciertoTotal)
modelos <- c (modelos, "SVM radial")




# Podemos ver aquí el resumen de los resultados:
resultado <- data.frame (aciertos = aciertos, modelos = modelos)


resultado [with(resultado, order(-aciertos)), ]




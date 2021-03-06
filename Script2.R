housing <- read.csv("~/aprendizagem de maquina/housing.csv")
View(housing)
dados <- housing
rm(housing)
head(dados)

str(dados)

#Variaveis

# longitude       
# latitude 
# housing_median_age
# total_rooms     
# total_bedrooms    
# population        
# households        
# median_income     
# median_house_value
# ocean_proximity   

attach(dados)


# ANALISE EXPLORATORIA ------------------------------------------------------

#medias e medianas 
summary(dados)
#devios padroes
round(sd(longitude),2)
round(sd(latitude),2)
round(sd(housing_median_age),2)
round(sd(total_rooms),2)
round(sd(total_bedrooms),2)
round(sd(population),2)
round(sd(households),2)
round(sd(median_income),2)
round(sd(median_house_value),2)
#normalidade
shapiro.test(longitude)
shapiro.test(latitude)
shapiro.test(housing_median_age)
shapiro.test(total_rooms)
shapiro.test(total_bedrooms)
shapiro.test(population)
shapiro.test(households)
shapiro.test(median_income)
shapiro.test(median_house_value)
#histogramas
par(mfrow=c(2,2))
hist(longitude, xlab = "Longitude", ylab = "Frequ�ncia", main="")
hist(latitude, xlab = "Latitude", ylab = "Frequ�ncia", main="")
hist(housing_median_age,xlab = "Idade m�dia das casas", ylab = "Frequ�ncia", main="")
hist(total_rooms,xlab = "Total de c�modos", ylab = "Frequ�ncia", main="")
par(mfrow=c(2,2))
hist(total_bedrooms,xlab = "Total de quartos", ylab = "Frequ�ncia", main="")
hist(population,xlab = "Popula��o", ylab = "Frequ�ncia", main="")
hist(households,xlab = "Fam�lias", ylab = "Frequ�ncia", main="")
hist(median_income, xlab = "Renda m�dia", ylab = "Frequ�ncia", main="")
x11()
options(scipen = 999)
hist(median_house_value, xlab = "Custo m�dio", ylab = "Frequ�ncia", main="")
#correlacoes  
cor(median_house_value, longitude)
cor(median_house_value, latitude)
cor(median_house_value, housing_median_age)
cor(median_house_value, total_rooms)
cor(median_house_value, total_bedrooms) 
cor(median_house_value, population)
cor(median_house_value, households)
cor(median_house_value, median_income) #maior correlacao
#variavel qualitativa
table(ocean_proximity)

# PRE-PROCESSAMENTO -------------------------------------------------------

summary(dados) #observamos que apenas a vari�vel total_bedrooms possui NA

# Preenchendo todos os valores NA com o valor m�dio

library(dplyr)

dados <- dados %>% 
  mutate_all(~ ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))
summary(dados)
attach(dados)

# Deixando todas na mesma escala

dados$longitude <- scale(dados$longitude)
dados$latitude <- scale(dados$latitude)
dados$housing_median_age <- scale(dados$housing_median_age)
dados$total_rooms <- scale(dados$total_rooms)
dados$total_bedrooms <- scale(dados$total_bedrooms)
dados$population <- scale(dados$population)
dados$households <- scale(dados$households)
dados$median_income <- scale(dados$median_income)
attach(dados)

# SEPARANDO OS CONJUNTO - TESTE E TREINAMENTO -----------------------------

library(caret)
set.seed(2104)
trainIndex <- createDataPartition(dados$ocean_proximity,
                                  p = .8, list = FALSE, times = 1) #-- balanceando entre proximidade do oceano
head(trainIndex)
dadosTrain <- dados[ trainIndex,] #--- amostra de treinamento
dadosTest <- dados[-trainIndex,] #--- amostra usada para testar a previs�o
table(dadosTrain$ocean_proximity)
table(dadosTest$ocean_proximity)


# APLICANDO OS MODELOS ----------------------------------------------------

#---------------------------
# REGRESSAO LINEAR SIMPLES
#---------------------------

# VARI�VEL RESPOSTA: median_house_value
# VARI�VEL EXPLICATIVA: median_income


# GRAFICO DE DISPERSAO
plot(median_house_value~median_income)


# CALCULO DA CORRELACAO
cor(median_house_value,median_income)
# Podemos observar por meio do calculo da correla��o das vari�veis median_income
# e median_house_value, que existe uma correlacao proxima de 0.7, ou seja, nao 
# podemos concluir que a correla�ao entre as duas � forte.


# TESTE DA CORRELACAO
cor.test(median_house_value,median_income, method = "pearson", 
         alternative="great")
# Outra forma de saber se h� correla��o entre as vari�veis � fazendo o teste de 
# correlacao. Assim feito, podemos observar por meio dessa que a correla��o das 
# vari�veis independente e explicativa, que existe uma correlacao proxima de 0.7.
# Com isso, � inferido que a correla�ao entre as duas nao � forte.


# AJUSTE DO MODELO
mod1 <- lm(median_house_value ~ median_income, data = dadosTrain); mod1


# RESUMO DO AJUSTE
summary(mod1)
names(mod1)
names(summary(mod1))
# Ao analizarmos os testes  T e F (teste de adequa��o global), na qual em ambos 
# n�o rejeitamos a hipotese nula, provamos que existe rela��o linear entre as 
# v�riaveis.


plot(median_house_value ~ median_income, data = dadosTrain, main = "Grafico de Dispersao")
abline(mod1, col = "red")
# Pelo gr�fico � poss�vel afirmar que o aumento do valor m�dio das casas 
# faz com que a renda m�dia tamb�m aumente.


anova(mod1)
# Atraves da tabela da anova podemos observar que, o parametro (beta 1) da 
# vari�vel median_income esta bem estimado para a nossa reta, pois atraves 
# do p valor nos rejeitamos a hipotese nula. Dessa forma, podemos confirmar 
# por meio da anova uma dependencia linear entre o valor m�dio das casas e
# a renda m�dia.


# Predicao
fitted(mod1)
predict(mod1)
mod1$fitted.values
predict(mod1, newdata = data.frame(x = 3000), type = "response")
points(3000, y_est, col = "red", pch=20, cex=2)

#---------------------------
# REGRESSAO LINEAR M�LTIPLA
#---------------------------

names(dados)
mod2 <- lm(median_house_value ~ longitude + latitude  + housing_median_age  + 
             total_rooms + total_bedrooms + population +households + 
             median_income,data = dadosTrain)
mod2
summary.lm(mod2)
# Analisando esse  modelo, percebemos que nos Testes T, rejeitamos a hipotese
# de que todos os coeficiente   � diferente de zero. Al�m disso, segundo
# o teste F, temos evid�ncias para afirmar que existe pelo menos uma variavel 
# que mant�m rela��o linear com a vari�vel resposta. Ademias, veja que o R 
# quadrado ajustdo apresentou um bom resultado, no qual apresentou valor de
# aproximadamente 0,6356, o que mostra que cerca de 63,56 % da varabilidade 
# total est� sendo explicada pelo modelo de regress�o.


residuo <- residuals(mod2)
fit <- fitted.values(mod2)
ard <- ls.diag(mod2)
respadron <- ard$std.res
hi <- ard$hat 
cook <- ard$cooks

#--------------------
# ANALISE RESIDUAL
#--------------------

# NORMALIDADE
library(nortest)
lillie.test(respadron)
qqnorm(respadron); qqline(respadron, col=2)
# Ao realizar o teste de lilliefors e considerando o n�vel de signific�ncia de 
# 5% rejeitamos a hipotese nula, ou seja, nao temos evidencias suficientes para 
# afirmar que os dados seguem normalidade. Porem, atraves do grafico podemos 
# constatar que os dados seguem certa normalidade mas estes est�o sendo afetados
# por possiveis outliers. Com  base nessas duas afirma��es, iremos considerar a 
# n�o normalidade dos residuos.



# HOMOCEDASTICIDADE
library(lmtest)
plot(fit, respadron)
bptest(mod2)
# Considerando um n�vel de confian�a de 5%, ao realizar os teste de 
# Breusch-Pagan, rejeitamos a hipotese nula, ou seja, temos evid�ncias 
# suficientes para afirmar que os erros s�o nao homocedasticos. Pelo gr�fico � 
# possivel observar que os dados n�o se apresentam distribudos de forma 
# aleatoria, comprovando assim a afirma��o do teste.



# LINEARIDADE
plot(dadosTrain$median_house_value~fit)
resettest(mod2)
# Ao analisar o teste RESET, percebemos que a hipotese de linearidade � 
# rejeitada a um n�vel de signific�ncia de 5%. Al�m disso, ao ver o gr�fico, 
# podemos considerar que a hipotese de linearidade est� sendo violada, visto 
# que os pontos distoam de um reta .



# AUTOCORRELA��O
library(car)
durbinWatsonTest(mod2,max.lag=1)
acf(respadron)
# Podemos observar que todas as correla�oes dao muito baixas, proximas a 0, mostrando
# que a altocorrela��o dos residuos eh muito baixa.  
# De acordo com o grafico, podemos ver que todos os lags estao fora dos limites,
# ou seja, a um n�vel de 5% de signific�ncia rejeitamos a hipotese nula, o que 
# implica dizer que temos evid�ncias suficientes para afirmar que a suposi��o 
# de indenpencia dos erros est� sendo violada.


#---------------------------
# ANALISE DE DIAGNOSTICO
#---------------------------

n <- nrow(dadosTrain)
p <- mod2$rank


# IDENTIFICANDO PONTOS ABERRANTES
plot(respadron)
abline(h=-2, col = 2); abline(h=2, col=2)
aberrantes <- which(abs(respadron)>2)



# IDENTIFICANDO PONTOS DE ALAVANCA
plot(hi)
abline(h=2*(p/n), col=2) 
alavanca <- which(hi>2*(p/n))



# IDENTIFICANDO PONTOS DE INFLUENCIA
plot(cook)
abline(h=4/n, col = 2);
influencia <- which(cook>4/n)



# VERIFICANDO OS PONTOS ABERRANTES QUE NAO SAO DE ALAVANCA
i <- NULL
vetor1 <- numeric()
for (i in 1:length(aberrantes)){
  if (aberrantes[i] %in% alavanca)
    NULL
  else
    vetor1[i] <- print(aberrantes[i])
}
vetor_aberr_nao_alav <- as.vector(na.omit(vetor1))
vetor_aberr_nao_alav



# VERIFICANDO OS PONTOS ABERRANTES QUE NAO SAO DE INFLUENCIA NEM DE ALAVANCA
i <- NULL
vetor2 <- numeric()
for (i in 1:length(vetor_aberr_nao_alav)){
  if (vetor_aberr_nao_alav[i] %in% influencia)
    NULL
  else
    vetor2[i] <-print(vetor_aberr_nao_alav[i])
}
vetor_aberrantes <- as.vector(na.omit(vetor2))
vetor_aberrantes
# Se desejarmos fazer uma analise mais avan�ada, podemos retirar os pontos
# que sao aberrantes que nao sao nem de alavanca nem de influencia. Assim feito, 
# foi observado que h� 338 observa��es que poder�o ser retiradas. Dessa forma,
# foi construido o codigo abaixo afim de retirar esses pontos. Sendo assim, ire-
# mos reconstruir nosso modelo sem essas observa��es.



# RETIRANDO OS PONTOS QUE SAO ABERRANTES POREM NAO SAO DE ALAVANCA NEM DE 
# INFLUENCIA
dadosTrain2 <- dadosTrain[-c(vetor_aberrantes),]



# FAZENDO A SELE��O DE VARIAVEIS.
names(dadosTrain)
mod3 <- lm(median_house_value ~ longitude + latitude  + housing_median_age  + 
             total_rooms + total_bedrooms + population +households + 
             median_income,dadosTrain2)
mod3
summary.lm(mod3)
# Sendo assim, Analisando esse modelo, percebemos que nos Testes T, continuamos  
# rejeitando a hipotese de que todos os coeficiente s�o diferente de zero. Al�m 
# disso, segundo o teste F, temos evid�ncias para afirmar que existe pelo menos
# uma variavel que mant�m rela��o linear com a vari�vel resposta. Ademias, veja
# que o R quadrado ajustado apresentou um bom resultado e foi maior que nosso
# modelo anterior, no qual apresentou valor de aproximadamente 0,6479, o que 
# mostra que cerca de 64,79 % da varabilidade total est� sendo explicada pelo 
# modelo de regress�o. Mostrando uma pequena melhora do nosso r2, e tambem uma 
# diminui��o do nosso sigma chapeu, podemos concluir que esse novo modelo 
# apresnta uma pequena melhora do ajuste do nosso modelo.

#--------------------------------------------------
#  MEDIDAS PARA CORRIGIR AS VIOLA��ES DAS HIPOTESES
#--------------------------------------------------

# Fazendo uma nova sele��o de vari�veis atrav�s da multicolinearidade 

#Podemos usar a multicolieanirade antes de aplicar o modelo, para fazer como uma analise explonatoria 
matriz.X = cbind(longitude, latitude, housing_median_age, total_bedrooms, total_rooms, population, households, median_income) #variaveis independentes do meu modelo final. mas poderiamos usar todas as variaveis e fazer um pre-processamento.
cor(matriz.X)

library("car")
vif(mod3) 

#logo aquelas que obtiveram vif(fatores de infla��o de variancia )>2 devem ser retiradas do nosso banco, sobrando as seguintes;

mod4 <- lm(median_house_value ~  + housing_median_age  +  median_income,dadosTrain2)
mod4
summary.lm(mod4)

#Transformacao de Box~cox 

#Necessario denotar a formula do modelo: y ~ X1 + X2 + ... + Xp
#Caso o modelo final nao tenha o intercepto: y ~ 0 + X1 + X2 + ... + Xp

library("MASS")
boxcox(mod4, lambda = seq(-0.2,0.5,0.1), plotit=TRUE, data=dadosTrain2)


library("car")

ylambda = bcPower(dadosTrain2$median_house_value,lambda = 0.3)#transformando a variavel

mod5 <- lm(ylambda ~ housing_median_age + median_income,dadosTrain2)
summary(mod5)

#no qual vemos uma diminuicao  do nosso r2. Porem por outro lado, vemos uma diminui��o relativamente grande do nosso sigma chapeu. Fazendo com o que o modelo fique
#melhor ajustado a priori apesar do nosso problema com o r2 no qual foi de apenas 48%.

residuo <- residuals(mod5)
fit <- fitted.values(mod5)
ard <- ls.diag(mod5)
respadron <- ard$std.res
hi <- ard$hat 
cook <- ard$cooks

#--------------------
# ANALISE RESIDUAL
#--------------------

# NORMALIDADE

lillie.test(respadron)
qqnorm(respadron); qqline(respadron, col=2)
# Ao realizar o teste de lilliefors e considerando o n�vel de signific�ncia de 
# 5% rejeitamos a hipotese nula, ou seja, nao temos evidencias suficientes para 
# afirmar que os dados seguem normalidade. Porem, atraves do grafico podemos 
# constatar que os dados seguem certa normalidade mas estes podem ta sendo afetados
# por possiveis outliers. Com  base nessas duas afirma��es, iremos considerar a 
# normalidade dos residuos.



# HOMOCEDASTICIDADE

plot(fit, respadron)
bptest(mod5)
# Considerando um n�vel de confian�a de 5%, ao realizar os teste de 
# Breusch-Pagan, rejeitamos a hipotese nula, ou seja, temos evid�ncias 
# suficientes para afirmar que os erros s�o nao homocedasticos. Pelo gr�fico � 
# possivel observar que os dados  se apresentam distribudos de forma 
# aleatoria, comprovando assim certa homocedasticidade dado que os testes sao sensiveis a outliers.



# LINEARIDADE
plot(ylambda~fit)
resettest(mod5)
# Ao analisar o teste RESET, percebemos que a hipotese de linearidade � 
# rejeitada a um n�vel de signific�ncia de 5%. Al�m disso, ao ver o gr�fico, 
# podemos considerar que a hipotese de linearidade est� sendo violada, visto 
# que os pontos distoam de um reta .



# AUTOCORRELA��O
durbinWatsonTest(mod5,max.lag=1)
acf(respadron)
# Podemos observar que todas as correla�oes dao muito baixas, proximas a 0, mostrando
# que a altocorrela��o dos residuos eh muito baixa.  
# De acordo com o grafico, podemos ver que todos os lags estao fora dos limites,
# ou seja, a um n�vel de 5% de signific�ncia rejeitamos a hipotese nula, o que 
# implica dizer que temos evid�ncias suficientes para afirmar que a suposi��o 
# de indenpencia dos erros est� sendo violada.
install.packages("forecast")
library("forecast")
InvBoxCox(ylambda,0.3)#retornando a variavel original.


#pre�o_da_casa = 123.6245 + 3.6262*(idade media da casa)  + 14.8908*(renda mediana )


# VALIDA�AO CRUZADA -------------------------------------------------------

mod1.pred <- predict(mod1, newdata = dadosTest, se.fit = T) #reg linear simples
mod2.pred <- predict(mod2, newdata = dadosTest, se.fit = T) #reg linear multipla
mod3.pred <- predict(mod3, newdata = dadosTest, se.fit = T) #reg multipla com retirada de alguns outliers
mod4.pred <- predict(mod4, newdata = dadosTest, se.fit = T) #multicolinariedade
mod5.pred <- predict(mod5, newdata = dadosTest, se.fit = T) #boxcox

mod1.pred.error <- mod1.pred$fit - dadosTest$median_house_value
mod2.pred.error <- mod2.pred$fit - dadosTest$median_house_value
mod3.pred.error <- mod3.pred$fit - dadosTest$median_house_value
mod4.pred.error <- mod4.pred$fit - dadosTest$median_house_value
mod5.pred.error <- mod5.pred$fit - dadosTest$median_house_value

mod1.mspe <- mean(mod1.pred.error^2)
mod2.mspe <- mean(mod2.pred.error^2)
mod3.mspe <- mean(mod3.pred.error^2)
mod4.mspe <- mean(mod4.pred.error^2)
mod5.mspe <- mean(mod5.pred.error^2)

mod1.mspe
mod2.mspe #o segundo deu menor
mod3.mspe
mod4.mspe
mod5.mspe

#M�TODO SIMPLES DE AVALIA��O - Abordagem do conjunto de valida��o (ou divis�o de dados)

AIC(mod1)
AIC(mod2) 
AIC(mod3)
AIC(mod4) 
AIC(mod5) #menor

BIC(mod1)
BIC(mod2) 
BIC(mod3)
BIC(mod4)
BIC(mod5) #menor

prediction1 <- predict(mod1, newdata = dadosTest)
prediction2 <- predict(mod2, newdata = dadosTest)
prediction3 <- predict(mod3, newdata = dadosTest)
prediction4 <- predict(mod4, newdata = dadosTest)
prediction5 <- predict(mod5, newdata = dadosTest)


data.frame( R2 = R2(prediction1, dadosTest$median_house_value ), 
            RMSE = RMSE(prediction1, dadosTest$median_house_value ), 
            MAE = MAE(prediction1, dadosTest$median_house_value )) 

data.frame( R2 = R2(prediction2, dadosTest$median_house_value ), 
            RMSE = RMSE(prediction2, dadosTest$median_house_value ), 
            MAE = MAE(prediction2, dadosTest$median_house_value )) 

data.frame( R2 = R2(prediction3, dadosTest$median_house_value ), 
            RMSE = RMSE(prediction3, dadosTest$median_house_value ), 
            MAE = MAE(prediction3, dadosTest$median_house_value )) 

data.frame( R2 = R2(prediction4, dadosTest$median_house_value ), 
            RMSE = RMSE(prediction4, dadosTest$median_house_value ), 
            MAE = MAE(prediction4, dadosTest$median_house_value )) 

data.frame( R2 = R2(prediction5, dadosTest$median_house_value ), 
            RMSE = RMSE(prediction5, dadosTest$median_house_value ), 
            MAE = MAE(prediction5, dadosTest$median_house_value )) 

#No geral, o modelo 5 foi o melhor

#K-FOLD
set.seed(2104)
treinamento_k_fold <- trainControl(method = "cv", number = 10, 
                                   verboseIter = TRUE)
modelo_k_fold1 <- train(median_house_value ~ median_income, dadosTrain,
                       method = "lm",trControl = treinamento_k_fold)
modelo_k_fold2 <- train(median_house_value ~ longitude + latitude +  
                         housing_median_age + total_rooms + total_bedrooms 
                       + population + households + median_income, dadosTrain,
                       method = "lm",trControl = treinamento_k_fold)

modelo_k_fold3 <- train(median_house_value ~  + housing_median_age  +  median_income,dadosTrain2,
                        method = "lm",trControl = treinamento_k_fold)

dadosTrain2 <- dadosTrain2 %>% mutate(ylambda)
modelo_k_fold4 <- train(ylambda ~ housing_median_age + median_income,dadosTrain2,
                        method = "lm",trControl = treinamento_k_fold)


print(modelo_k_fold1)
print(modelo_k_fold2)
print(modelo_k_fold3)
print(modelo_k_fold4)


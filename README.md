# UNIVERSIDAD DE COSTA RICA
## ESCUELA DE INGENIERÍA ELÉCTRICA

## IE0405 - MODELOS PROBABILÍSTICOS DE SEÑALES Y SISTEMAS 

# TAREA 3

## EDGAR MADRIGAL VÍQUEZ
## CARNÉ: B64047
## Profesor: Fabián Abarca


##  PREGUNTA 1

A partir de los datos presentes en el archivo xy se determinó que la mejor curva de ajuste es la gaussiana de acuerdo a la forma de campana observada en la curva de datos graficada.  Se procedió a graficar la curva de densidad marginal para X y la curva de densidad marginal para Y. Para obtener estos respectivos vectores se realizó la suma de los valores de probabilidad de cada una de las variables. Para en el caso de las X obtener un vector de longitud 11 y en el caso de las Y obtener un vector de longitud 21. Los valores respectivos de X están entre 5 y 10 y los valores de Y entre 5 y 25.

Ecuación para función de densidad marginal de X:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_x(x) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}}">  
</p>

Ecuación para función de densidad marginal de Y:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_x(x) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(y-\mu)^2}{2\sigma^2}}">  
</p>


Curva de ajuste gaussiana para la densidad marginal de X:
![AjusteX](/fitX.png)

Curva de ajuste gaussiana para la densidad marginal de Y:
![AjusteY](/fitY.png)



## PREGUNTA 2

Recordando, teníamos que las funciones marginales de X y Y son, respectivamente:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_x(x) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}}">  
</p>

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_x(x) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(y-\mu)^2}{2\sigma^2}}">  
</p>


Como se asume la independecia de X y de Y, la función de densidad conjunta es la multiplicación de la función de densidad marginal de X y la función de densidad marginal de Y.
Entonces la función de densidad conjunta es:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_{x,y}(x,y) = (\frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}} ) \cdot (\frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(y-\mu)^2}{2\sigma^2}})">  
</p>

Sustiyendo los valores respectivos de la media y desviación estándar, tenemos que:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_{x,y}(x,y) = (\frac{1}{(3.1622) \sqrt{2\pi}} \cdot e^{-\frac{(x-10)^2}{2(3.1622)^2}} ) \cdot (\frac{1}{(3.0553) \sqrt{2\pi}} \cdot e^{-\frac{(y-15)^2}{2(3.0553)^2}})">  
</p>




## PREGUNTA 3 
Hallar los valores de correlación, covarianza y coeficiente de correlación (Pearson) para los datos y explicar su significado.

Correlación: 

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\sum_{y=5}^{25}\sum_{x=5}^{15}xy f_{x,y}(x,y)">  
</p>

Covarianza:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\sum_{y=5}^{25}\sum_{x=5}^{15}(x-\bar{X})(y-\bar{Y}) f_{x,y}(x,y)">  
</p>

Coeficiente de correlación:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\sum_{y=5}^{25}\sum_{x=5}^{15}\frac{(x-\bar{X})}{\sigma_x}\frac{(y-\bar{Y})}{\sigma_y} f_{x,y}(x,y)">  
</p>


## PREGUNTA 4
Graficar las funciones de densidad marginales (2D), la función de densidad conjunta (3D).

La función de densidad conjunta se calcula con la siguiente ecuación:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_{x,y}(x,y) = f_x(x)\cdot f_y(y)">  
</p>

Para graficar en 3D la densidad conjunta, el vector de las z corresponde a la multiplicación de los vectores de densidad marginal de X y de Y, para obtener un vector de longitud 231 que me permita graficar junto con los vectores de las X y las Y del archivo xyp.

Función de densidad conjunta de 3D:
![conjunta](/d_conjunta.png)


Función de densidad marginal de X:
![X](/paraX.png)


Función de densidad marginal de Y:
![Y](/paraY.png)








 

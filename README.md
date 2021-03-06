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

Para el vector de densidad marginal de X se obtuvo una media de: 10 y una desviación estándar de 3.1622.

Ecuación para función de densidad marginal de Y:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_x(y) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(y-\mu)^2}{2\sigma^2}}">  
</p>

Para el vector de densidad marginal de Y se obtuvo una media de: 15 y una desviación estándar de 6.02693775. 

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
  <img src="https://render.githubusercontent.com/render/math?math=f_y(y) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(y-\mu)^2}{2\sigma^2}}">  
</p>


Como se asume la independencia de X y de Y, la función de densidad conjunta es la multiplicación de la función de densidad marginal de X y la función de densidad marginal de Y.
Entonces la función de densidad conjunta es:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_{x,y}(x,y) = (\frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}} ) \cdot (\frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(y-\mu)^2}{2\sigma^2}})">  
</p>

Sustiyendo los valores respectivos de la media y desviación estándar, tenemos que la expresión de la función de densidad conjunta que modela los datos es:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_{x,y}(x,y) = (\frac{1}{(3.1622) \sqrt{2\pi}} \cdot e^{-\frac{(x-10)^2}{2(3.1622)^2}} ) \cdot (\frac{1}{(6.03) \sqrt{2\pi}} \cdot e^{-\frac{(y-15)^2}{2(6.03)^2}})">  
</p>




## PREGUNTA 3 

Correlación: 

Por medio de la siguiente ecuación se obtiene el valor de correlación.

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\sum_{y=5}^{25}\sum_{x=5}^{15}xy f_{x,y}(x,y)">  
</p>

 El valor de correlación obtenido fue de: 149.54281.
La correlación me indica si 2 variables están relacionadas. Como la correlación se puede escribir de la forma Rxy = E[X]* E[Y] entonces podemos decir que no hay correlación entre X y Y.


Covarianza:

La covarianza es un valor que indica la variación que tienen dos variables aleatorias con respecto a su respectiva media.
Se puede obtener este valor con la siguiente ecuación:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\sum_{y=5}^{25}\sum_{x=5}^{15}(x-\bar{X})(y-\bar{Y}) f_{x,y}(x,y)">  
</p>

El valor de covarianza obtenido fue de: 0.06481
Podemos observar que el valor obtenido de covarianza es aproximadamente 0, por lo que se puede interpretar que no hay covarianza ya que no hay correlación. Se obtuvo el valor esperado.

Coeficiente de correlación:

Este coeficiente de correlación o coeficiente de Pearson indica también de cierta forma qué tanto están relacionadas dos variables.
Este valor se puede obtener con la siguiente ecuación:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\sum_{y=5}^{25}\sum_{x=5}^{15}\frac{(x-\bar{X})}{\sigma_x}\frac{(y-\bar{Y})}{\sigma_y} f_{x,y}(x,y)">  
</p>

El valor del coeficiente de correlación obtenido fue de: 0.00338459.
Como era de esperarse, el valor fue nuevamente aproximadamente 0 ya que no hay relación entre X y Y.



## PREGUNTA 4
Graficar las funciones de densidad marginales (2D), la función de densidad conjunta (3D).

La función de densidad conjunta se calcula con la siguiente ecuación:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f_{x,y}(x,y) = f_x(x)\cdot f_y(y)">  
</p>

Para graficar en 3D la densidad conjunta, el vector de las z corresponde a la multiplicación de los vectores de densidad marginal de X y de Y, para obtener un vector de longitud 231 que me permita graficar junto con los vectores de las X y las Y del archivo xyp. Se observa de igual forma el comportamiento en forma gaussiana de la curva.

Función de densidad conjunta en 3D:
![conjunta](/d_conjunta.png)


Función de densidad marginal de X:
![X](/paraX.png)


Función de densidad marginal de Y:
![Y](/paraY.png)








 

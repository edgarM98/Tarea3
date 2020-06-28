# Se importa las librerías necesarias

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

# Abrimos el archivo .txt y leemos
prueba = open("dm.txt", "r")
lineas = prueba.read().splitlines()

# Esto es para leer como float cada elemento de la lista.
datos = [linea.split(",") for linea in lineas]
datos = [[float(dato) for dato in linea] for linea in datos]
#print(datos)

# El total de muestras va a ser el número de muestras que tenga el archivo .txt
total = len(datos)
matriz = np.array(datos)
#print(matriz)
#print("\n")

####### MEDIA, VARIANZA, DESVIACION ESTANDAR    #########
# Calculamos la media, varianza y desviacion estandar

a = np.array(datos)
mu = a.mean()
sigma = a.std()
var = a.var()

print("La media es:", mu)
print("La varianza es:", var)
print("La desviación estandar es:", sigma)


# MEDIA = 0.004329
# DESVIACION = 0.0000076121

#### FUNCION DE DENSIDAD MARGINAL DE X ##########

# Crea el vector para la densidad marginal de X
lx = np.sum(matriz, axis=1)
#print(fi)

# Vector creado para graficar con respecto a la función
# marginal de X
zx = [5,6,7,8,9,10,11,12,13,14,15]

# Grafico ambos vectores 

plt.plot(zx,lx)
plt.title("Función de densidad marginal de X")
plt.ylabel('Vector de densidad marginal de x')
plt.xlabel('Vector de x')
plt.savefig("paraX.png")
plt.show()

a = np.array(lx)
mu = a.mean()
sigma = a.std()
var = a.var()

#print("La media es:", mu)
#print("La varianza es:", var)
#print("La desviación estandar es:", sigma)

# La media es de: 10
# La desviación estándar es de: 3.1622


#### CURVA DE AJUSTE FUNCION DE DENSIDAD MARGINAL DE X ###

x = [5,6,7,8,9,10,11,12,13,14,15]

# Parametros
n = len(x)                        
mu = 10              
sigma = 3.1622   

# Defino la ecuacion de la función de densidad gaussiana
def gaus(x,a,mu,sigma):
  return 1 / (np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))

# Realiza el ajuste con la curva
popt,pcov = curve_fit(gaus,x,lx,p0=[1,mu,sigma])

# Grafica la curva de ajuste gaussiana
plt.plot(x,lx,'b+:',label='data')
plt.plot(x,gaus(x,*popt),'ro:',label='fitCurve')
plt.legend()
plt.ylabel('Vector de densidad marginal de x')
plt.xlabel('Vector de x')
plt.title("Curva de ajuste gaussiana para densidad marginal de X") 
plt.savefig("fitX.png")
plt.show()


#### FUNCION DE DENSIDAD MARGINAL DE Y ##########

# Crea el vector para la densidad marginal de Y
ly = np.sum(matriz, axis=0)
#print(fi)

# Vector creado para graficar con respecto a la función
# marginal de Y
zy = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# Grafico
plt.plot(zy,ly)
plt.title("Función de densidad marginal de Y")
plt.ylabel('Vector de densidad marginal de y')
plt.xlabel('Vector de y')
plt.savefig("paraY.png")
plt.show()

at = np.array(zy)
mu = at.mean()
sigma = at.std()
var = at.var()

print("La media es:", mu)
print("La varianza es:", var)
print("La desviación estandar es:", sigma)


# La media es de: 15
# La desviación estándar es de: 6.02693775

#### CURVA DE AJUSTE FUNCION DE DENSIDAD MARGINAL DE Y ###

# Mismo procedimiento, pero ahora en base al array de densidad marginal de Y

x = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

n = len(x)                        
mu = 15               
sigma = 3.05530     

def gaus(x,a,mu,sigma):
  return 1 / (np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,ly,p0=[1,mu,sigma])

# Grafico la curva para array densidad marginal de Y

plt.plot(x,ly,'b+:',label='data')
plt.plot(x,gaus(x,*popt),'ro:',label='fitCurve')
plt.legend()
plt.ylabel('Vector de densidad marginal de y')
plt.xlabel('Vector de y')
plt.title("Curva de ajuste gaussiana para densidad marginal de Y") 
plt.savefig("fitY.png")
plt.show()





#######################################
#      AHORA USAMOS EL ARCHIVO XYP
#######################################


# Abrimos el archivo .txt y leemos
prueba1 = open("dm2.txt", "r")
lineas1 = prueba1.read().splitlines()

# Esto es para leer como float cada elemento de la lista.
datos1 = [linea1.split(",") for linea1 in lineas1]
datos1 = [[float(dato1) for dato1 in linea1] for linea1 in datos1]
#print(datos)

# El total de muestras va a ser el número de muestras que tenga el archivo .txt
total = len(datos1)
matriz = np.array(datos1)


###########   CORRELACIÓN   ##########

# Calcula la correlacion en base a esta ecuacion según los datos del archivo xyp

# Correlación = x*y*p
# Multiplico cada fila, forma vector con esos valores y sumo todos los valores para obtener la correlación

Fy = np.prod(datos1,axis=1)
#print(Fy)

def sumalista(listaNumeros):
  laSuma = 0
  for i in listaNumeros:
    laSuma = laSuma + i
  return laSuma

th = sumalista(Fy)
print("La correlacion es de:", th)
# SE OBTUVO UN VALOR DE APROXIMADAMENTE 149.54281


###########   COVARIANZA    ##########

# Utilizo la siguiente ecuacion
# (x - X´)*(y - Y´)* p


# Obtengo la media de la primera columna del archivo xyp
sumX = np.mean(datos1,axis=0)
#print("La media del array x es:", sumX)
#print(sumX[0])

# Creo el vector con el elemento 10 de longitud 231
aX = [10]*231
#print (aX)

#Obtengo la primera columna del archivo xyp 
columX = np.array(datos1)
tX = columX[:,0]
#print(tX)

np_list1x = np.array(tX)
np_list2x = np.array(aX)

# Hago la resta de ambos vectores
np_X = np_list1x - np_list2x
#print(np_X)

################# PARA Y #################
# Mismo procedimiento, pero ahora para columna de Y

sumY = np.mean(datos1,axis=0)
#print("La media del array x es:", sumY)
#print(sumY[1])

# Creo el vector con el elemento 15 de longitud 231
aY = [15]*231
#print (ay)

columY = np.array(datos1)
tY = columY[:,1]
#print(tY)

np_list1y = np.array(tY)
np_list2y = np.array(aY)

np_Y = np_list1y - np_list2y
#print(np_Y)

col = np.array(datos1)
fxy = col[:,2]
#print(fxy)

cY = np.array(np_Y)
cX = np.array(np_X)

# Multipico ambos vectores  (x - X´)*(y - Y´)

producto1 = cX * cY

fdx = np.array(producto1)
fdy = np.array(fxy)

producto2 = fdx * fdy

# Sumo todos los elementos de ese array (eso es la covarianza)
covarian = np.sum(producto2)
print("La covarianza es de:", covarian)

## LA COVARIANZA ES: 0.06481


###########   PEARSON   ##########

#  Uso la siguiente ecuacion
# coef_pearson =  covarianza / sigmaX * sigmaY

# Vector de la columna de las x
dX = np.array(datos1)
dXp = dX[:,0]
desviacion_x = dXp.std()

# Vector de la columna de las y
dY = np.array(datos1)
dYp = dY[:,1]
desviacion_y = dYp.std()

# Aplico la formula
sigma = 0.06481 / (desviacion_x * desviacion_y )
print("El coeficiente de correlacion es de:", sigma)

# Coeficiente de pearson = 0.00338459

############# DENSIDAD CONJUNTA    ###########

# Creo los vectores de longitud 21 para cada elemento del vector de densidad marginal de x

ax = [lx[0]]*21
bx = [lx[1]]*21
cx = [lx[2]]*21
dx = [lx[3]]*21
ex = [lx[4]]*21
fx = [lx[5]]*21
gx = [lx[6]]*21
hx = [lx[7]]*21
ix = [lx[8]]*21
jx = [lx[9]]*21
kx = [lx[10]]*21

# Los hago arrays de numpy
n_ax = np.array(ax)
n_bx = np.array(bx)
n_cx = np.array(cx)
n_dx = np.array(dx)
n_ex = np.array(ex)
n_fx = np.array(fx)
n_gx = np.array(gx)
n_hx = np.array(hx)
n_ix = np.array(ix)
n_jx = np.array(jx)
n_kx = np.array(kx)

# Array de numpy para densidad marginal de y
dcly = np.array(ly)

# Realizo las multiplicaciones del vector de densidad marginal de X y el Y
dca = n_ax * dcly
dcb = n_bx * dcly
dcc = n_cx * dcly
dcd = n_dx * dcly
dce = n_ex * dcly
dcf = n_fx * dcly
dcg = n_gx * dcly
dch = n_hx * dcly
dci = n_ix * dcly
dcj = n_jx * dcly
dck = n_kx * dcly 

# Sumo todos estos vectores para tener un array final de longitud 
# de 231 elementos
d_conjunta = np.concatenate((dca, dcb, dcc, dcd, dce, dcf, dcg, dch, dci, dcj, dck))
#print(d_conjunta)


###########   GRAFICAR DENSIDAD CONJUNTA EN 3D ##########

# Procedo a graficar en 3D
# El vector de las X es el de la columna X del archivo xyp
# El vector de las Y es el de la columna X del archivo xyp
# El vector de las Z es el array de densidad conjunta 

dX = np.array(datos1)
dXp = dX[:,0]

dY = np.array(datos1)
dYp = dY[:,1]

#dZ = np.array(datos1)
#dZp = dY[:,2]


# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')

# Definimos los datos de prueba
x = np.array([dXp])
y = np.array([dYp])
z = np.array([d_conjunta])

# Agregamos los puntos en el plano 3D
ax1.scatter(x, y, z, c='r', marker='o')

ax1.set_xlabel('Vector de las x')
ax1.set_ylabel('Vector de las y')
ax1.set_zlabel('Densidad conjunta')

# Mostramos el gráfico
plt.savefig("d4.png")
plt.show()



'''

'''
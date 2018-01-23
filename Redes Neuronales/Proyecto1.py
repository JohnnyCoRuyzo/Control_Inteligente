import numpy as np
import tensorflow as tf

''' Numero de iteraciones a la tabla '''
iteraciones = 6

''' Numero de neuronas de la red '''
neuronas = 20

''' Numero de entradas totales '''
entradas = 3

''' Numero de entradas mostradas '''
entM = 3

''' Numero de salidas totales '''
salidas = 1

''' Tasa de aprendizaje '''
ta = 0.01

''' Numero de datos de la tabla de entrenamiento '''
ndatosE = 14 

''' Numero de datos de la tabla de generacion '''
ndatosG = 13


W = tf.Variable(0.1*tf.ones([neuronas,1+entradas]),'float')
C = tf.Variable(0.1*tf.ones([neuronas,salidas]),'float')
eT = tf.Variable(tf.zeros([ndatosG]),'float')
eTT = tf.Variable(tf.zeros([ndatosE*iteraciones]),'float')
yF = tf.Variable(tf.zeros([ndatosG]),'float')

'''Entrenamiento'''
x = tf.placeholder(tf.float32,[ndatosE,1+entradas])
yd = tf.placeholder(tf.float32,[ndatosE,salidas])

'''Generacion'''
xg = tf.placeholder(tf.float32,[ndatosG,1+entradas])
ydg = tf.placeholder(tf.float32,[ndatosG,salidas])

for i in range(iteraciones):
    for m in range(ndatosE):
        y = 0
        for n in range(neuronas):
            s = 0
            for p in range(entM+1):
                s = s+W[n][p]*x[m][p]
            h = tf.tanh(s)
            y = y + h*C[n]
            e = yd[m]-y
            ad = 2*ta*C[n]*e*(1-y*y)
    
    eTT = eTT + tf.one_hot(ndatosE*i+m,ndatosE*iteraciones,e[0]*e[0])
    
    W = W + ad*x[m]
    C = C + 2*ta*e*y

for m in range(ndatosG):
    y = 0
    for n in range(neuronas):
        s = 0
        for p in range(entM+1):
            s = s+W[n][p]*xg[m][p]
        h = tf.tanh(s)
        y = y + h*C[n]
        e = ydg[m]-y
        ad = 2*ta*C[n]*e*(1-y*y)
    eT = eT + tf.one_hot(m,ndatosG,e[0]*e[0])
    yF = yF + tf.one_hot(m,ndatosG,y[0])
    
    
xt = [[1,      0.7533,     0.7567,    0.7333],[1,     0.7500,   0.7533,     0.7567],
      [1,      0.6333,     0.7500,    0.7533],[1,     0.4967,   0.6333,     0.7500],
      [1,      0.5267,     0.4967,    0.6333],[1,     0.7467,   0.5267,     0.4967],
      [1,      0.7533,     0.7467,    0.5267],[1,     0.7367,   0.7533,     0.7467],
      [1,      0.7267,     0.7367,    0.7533],[1,     0.6333,   0.7267,     0.7367],
      [1,      0.5067,     0.6333,    0.7267],[1,     0.5367,   0.5067,     0.6333],
      [1,      0.7667,     0.5367,    0.5067],[1,     0.7300,   0.7667,     0.5367]]

ydt = [[     0.7500],[     0.6333],[    0.4967],[     0.5267],[   0.7467],[     0.7533],
      [      0.7367],[     0.7267],[    0.6333],[     0.5067],[   0.5367],[     0.7667],
      [      0.7300],[     0.7367]]

xgg = [[1,     0.7367,     0.7300,    0.7667],[1,     0.7067,   0.7367,     0.7300],
      [1,      0.6300,     0.7067,    0.7367],[1,     0.4967,   0.6300,     0.7067],
      [1,      0.7333,     0.4967,    0.6300],[1,     0.7400,   0.7333,     0.4967],
      [1,      0.6533,     0.7400,    0.7333],[1,     0.8333,   0.6533,     0.7400],
      [1,      0.7067,     0.8333,    0.6533],[1,     0.6067,   0.7067,     0.8333],
      [1,      0.4900,     0.6067,    0.7067],[1,     0.7000,   0.4900,     0.6067],
      [1,      0.7200,     0.7000,    0.4900]]

ygg = [[     0.6067],[     0.4900],[    0.7000],[     0.7200],[   0.7467],[     0.7067],
      [      0.6300],[     0.4967],[    0.7333],[     0.7400],[   0.6533],[     0.8333],
      [      0.7067]]

eT = tf.expand_dims(eT,0)
eTT = tf.expand_dims(eTT,0)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

cW, cC, ce, ceTT, cyF = sess.run([W,C,eT,eTT,yF],{x:xt, yd:ydt, ydg:ygg, xg:xgg})
print("\nW: %s\nC: %s\ne: %s\ny: %s\neTT: %s"%(cW, cC, ce, cyF, ceTT))
       
       
       
       
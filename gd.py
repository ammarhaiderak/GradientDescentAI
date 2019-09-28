import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

np.random.seed(101)
tf.set_random_seed(101)


learning_rate=0.02
epochs=1000
n_samples=200

train_x=np.linspace(0,50,n_samples*2).reshape(n_samples,2)
train_y=np.linspace(0,50,n_samples)

train_x[:,0]+=np.random.uniform(-4,4,n_samples)
train_x[:,1]+=np.random.uniform(-4,4,n_samples)

ones=np.ones(shape=[n_samples,1],dtype='float32')

train_x=np.append(train_x,ones,1)


train_y+=np.random.uniform(-4,4,n_samples)





B0=tf.Variable(np.random.randn())
B1=tf.Variable(np.random.randn())
B2=tf.Variable(np.random.randn())

X1=tf.placeholder(tf.float32)
X2=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

pred=B0+B1*X1+B2*X2

cost=tf.reduce_sum((Y-pred)**2)/(2*n_samples)

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.global_variables_initializer()

#print(train_x)

input('x')

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(B0))
    print(sess.run(B1))
    print(sess.run(B2))

    for epoch in range(epochs):
        print(epoch)
        for x1,x2,y in zip(train_x[:,0],train_x[:,1],train_y):
            sess.run(optimizer,feed_dict={ X1 : x1, X2 : x2 , Y : y })


    cb0=sess.run(B0)
    cb1=sess.run(B1)
    cb2=sess.run(B2)

    print(cb0)
    print(cb1)
    print(cb2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(train_x[:, 0], train_x[:, 1], train_y,'gray')
    ax.scatter3D(train_x[:, 0], train_x[:, 1], train_x[:,0]*cb1+train_x[:,1]*cb2+cb0,'red')

    plt.show()










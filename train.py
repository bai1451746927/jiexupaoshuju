import  numpy as np
import tensorflow as tf
x=np.random.rand(100).astype(np.float32)
y=x*0.1+0.3
w=tf.Variable(tf.random_uniform([1],-1.0,1.0),name='w')
b=tf.Variable(tf.zeros([1]),name='b')
m=w*x+b
mm=y-m
uu=tf.reduce_mean(tf.square(mm))
pip=tf.train.GradientDescentOptimizer(0.5)
pp=pip.minimize(uu)
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    saver=tf.train.Saver()
    for step in range(10):
        sess.run(pp)
        gra = saver.save(sess, './data',global_step=step)
        print(step)
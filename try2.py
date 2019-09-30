import numpy as np
import  tensorflow as tf
meta=tf.train.import_meta_graph(r'./data-14.meta')
sess=tf.Session()
meta.restore(sess,tf.train.latest_checkpoint(r'./'))
# canshu=meta.set_last_checkpoints('./')
# print(canshu)
graph=tf.get_default_graph()
print(sess.run("w:0"),sess.run("b:0"))
w=graph.get_tensor_by_name('w:0')
b=graph.get_tensor_by_name('b:0')
x=np.random.rand(100).astype(np.float32)
y=x*0.1+0.3
re=w*x+b
ff=re-y
uu=tf.reduce_mean(tf.square(ff))
kk=tf.train.GradientDescentOptimizer(0.1)
pp=kk.minimize(uu)

saver=tf.train.Saver()
for step in range(15):
    sess.run(pp)
    saver.save(sess,'./data',global_step=step)
    print(step)
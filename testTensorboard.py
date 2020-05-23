import tensorflow as tf

input1=tf.constant([1.0,2.0,3.0],name='input1')
input2=tf.Variable(tf.random.uniform([3]),name='input2')

output=tf.add_n([input1,input2],name='add')

writer = tf.summary.FileWriter("./testLog",tf.get_default_graph())
writer.close()
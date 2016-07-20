'''
Created on Jul 18, 2016

@author: zlfzeng
'''

import tensorflow as tf
import numpy as np
from simple.myrnn2 import seqlength

#hyper:
data=open("/Users/liup/rnn.test").read()
voc = sorted(list(set(data)))
char2int = {ch: i for i,ch in enumerate(voc)}
int2char = {i:ch for i,ch in enumerate(voc)}
vocsize= len(voc)
hidden_size = 100
seqsize = 25

x = tf.placeholder(tf.float32, [None, vocsize])
y = tf.placeholder(tf.float32, [None, vocsize])
h = tf.placeholder(tf.float32, [1, hidden_size])

init = tf.random_normal_initializer( stddev=0.01)

#compute y_predict
with tf.variable_scope("rnn", reuse=True) as scope: #does this affect teh control flow like other identation?
    wxh = tf.get_variable("wxh", [vocsize, hidden_size], tf.float32, initializer=init) #once for all
    whh = tf.get_variable("whh", [hidden_size, hidden_size], tf.float32, initializer=init)
    why = tf.get_variable("why", [hidden_size, vocsize], tf.float32, initializer=init)
    bh= tf.get_variable("bh", [hidden_size], tf.float32, initializer=init)
    by =tf.get_variable("why", [ vocsize], tf.float32, initializer=init)
    
    htmp = h
    yslices = []
    for xslice in tf.split(0, seqsize, x):
        htmp = tf.tanh(tf.matmul(xslice, wxh) + tf.matmul(htmp, whh) + bh)
        yslice = tf.matmul(htmp, why) + by
        yslices.append(yslice)
        
    y_predict = tf.concat(0, yslices)
    y_last = tf.nn.softmax(yslices[-1])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    init = tf.initialize_all_variables()

ss = tf.Session()
ss.run(init)

p=0
count = 0
htmp2 = np.zeros([1, hidden_size])
while True:
    count = count + 1
    if(p+1+seqlength>=len(data)):
        p=0
       
    xseq = data[p:p+seqlength]
    yseq= data[p+1, p+1+seqlength] 
    xseqvector = [char2int[ch] for ch in xseq]
    yseqvector=[char2int[ch] for ch in yseq]
   
    xmatrix = np.eye(vocsize)[xseqvector]
    ymatrix=np.eye(vocsize)[yseqvector]
    _,htmp2=ss.run([optimizer, htmp], feed_dict={x:xmatrix, y:ymatrix, h:htmp2})
   
    if count % 500 == 0:
        htmp3 = np.copy(htmp2)
        zseq = data[5:5+seqlength]
        zseqvector = [char2int[ch] for ch in zseq]
        
        yarray = []
        for t in range(200):
            zmatrix = np.eye(vocsize)[zseqvector]
            ydis,htmp3=ss.run([ y_last,htmp], feed_dict={x:xmatrix, y:ymatrix, h:htmp3})
            nexty = np.random.choice(range(vocsize), p=ydis.ravel())
            yarray.append(nexty)
            zseqvector=zseqvector[1:] + [nexty]
        
        ychararray = [int2char[i] for i in yarray]
        print ychararray
        
   
    p=p+seqlength
        
    

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.contrib import distributions

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def _weight_prior(name, shape):
    w = distributions.Normal(mu = tf.zeros(shape),sigma = tf.ones(shape),name = name+'_w_prior')
    tf.add_to_collection('weights_prior',w)
    return w


def _bias_prior(name, shape):
    b = distributions.Normal(mu = tf.zeros(shape),sigma = tf.ones(shape),name = name + '_b_prior')
    tf.add_to_collection('bias_prior',b)
    return b
    
    
def _weight_inference(name,shape):
    #create the mean to learn
    w_mu = tf.get_variable(name+'mu', shape, tf.float32, tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('w_mu',w_mu)
    
    #create the sigma to learn
    w_sigma = tf.nn.softplus(tf.get_variable(name+'sigma', shape, tf.float32, tf.contrib.layers.xavier_initializer()))
    tf.add_to_collection('w_sigma',w_sigma)
    
    #create the distribution for the pdf function
    w_dist = distributions.Normal(mu = w_mu, sigma = w_sigma,name = name)
    tf.add_to_collection('w',w_dist)
    
    #also make a prior distribution for that weight
    _weight_prior(name,shape)
    return w_mu,w_sigma,w_dist

def _bias_inference(name,shape):
    #create the mean to learn
    b_mu = tf.get_variable(name+'mu', shape, tf.float32, tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('b_mu',b_mu)
    
    #create the sigma to learn
    b_sigma = tf.nn.softplus(tf.get_variable(name+'sigma', shape, tf.float32, tf.contrib.layers.xavier_initializer()))
    tf.add_to_collection('b_sigma',b_sigma)
    
    #create the distribution for the pdf function
    b_dist = distributions.Normal(mu = b_mu, sigma = b_sigma,name = name)
    tf.add_to_collection('b',b_dist)
    
    #also make a prior distribution for that bias
    _bias_prior(name,shape)
    return b_mu,b_sigma,b_dist
    
def inference_network(x):
    #layer 1 for input x
    with tf.variable_scope('infer_layer1') as scope:
        w_mu,w_sigma,w_dist = _weight_inference('w',[x_dim,500])
        b_mu,b_sigma,b_dst = _bias_inference('b',[500])
        
        #we take a sample of epsilon for EVERY data point. Hence we are getting a new w and b for each point
        epsilon = tf.random_normal([batch_size,x_dim,500],0,1,dtype = tf.float32)
        w = tf.add(w_mu,tf.multiply(epsilon,w_sigma))
        tf.add_to_collection('w_sample',w)
        
        epsilon = tf.random_normal([batch_size,1,500],0,1,dtype = tf.float32)
        b = tf.add(b_mu,tf.multiply(epsilon,b_sigma))
        tf.add_to_collection('b_sample',b)
        
        layer1 = tf.nn.tanh(tf.matmul(x,w) + b)
        
    #layer 2 for input x
    with tf.variable_scope('infer_layer2') as scope:
        w_mu,w_sigma,w_dist = _weight_inference('w',[500,500])
        b_mu,b_sigma,b_dst = _bias_inference('b',[500])
        
        epsilon = tf.random_normal([batch_size,500,500],0,1,dtype = tf.float32)
        w = tf.add(w_mu,tf.multiply(epsilon,w_sigma))
        tf.add_to_collection('w_sample',w)
        
        epsilon = tf.random_normal([batch_size,1,500],0,1,dtype = tf.float32)
        b = tf.add(b_mu,tf.multiply(epsilon,b_sigma))
        tf.add_to_collection('b_sample',b)
        
        layer2 = tf.nn.tanh(tf.matmul(layer1,w) + b)
        
    #layer 3 for input x
    with tf.variable_scope('infer_layer3') as scope:
        w_mu,w_sigma,w_dist = _weight_inference('w',[500,500])
        b_mu,b_sigma,b_dst = _bias_inference('b',[500])
        
        epsilon = tf.random_normal([batch_size,500,500],0,1,dtype = tf.float32)
        w = tf.add(w_mu,tf.multiply(epsilon,w_sigma))
        tf.add_to_collection('w_sample',w)
        
        epsilon = tf.random_normal([batch_size,1,500],0,1,dtype = tf.float32)
        b = tf.add(b_mu,tf.multiply(epsilon,b_sigma))
        tf.add_to_collection('b_sample',b)
        
        layer3 = tf.nn.tanh(tf.matmul(layer2,w) + b)
        
    #output
    with tf.variable_scope('infer_output') as scope:
        w_mu,w_sigma,w_dist = _weight_inference('w',[500,latent_dim*2])
        b_mu,b_sigma,b_dst = _bias_inference('b',[latent_dim*2])
        
        epsilon = tf.random_normal([batch_size,500,latent_dim*2],0,1,dtype = tf.float32)
        w = tf.add(w_mu,tf.multiply(epsilon,w_sigma))
        tf.add_to_collection('w_sample',w)
        
        epsilon = tf.random_normal([batch_size,1,latent_dim*2],0,1,dtype = tf.float32)
        b = tf.add(b_mu,tf.multiply(epsilon,b_sigma))
        tf.add_to_collection('b_sample',b)
        
        output = tf.matmul(layer2,w) + b
        
    z_mu = output[:,:,:latent_dim]
    z_sigma = tf.nn.softplus(output[:,:,latent_dim:])
    
    return z_mu,z_sigma
    
    
def generator_network(z):
    with tf.variable_scope('gen_layer1') as scope:
        w_mu,w_sigma,w_dist = _weight_inference('w',[latent_dim,500])
        b_mu,b_sigma,b_dst = _bias_inference('b',[500])
        
        epsilon = tf.random_normal([batch_size,latent_dim,500],0,1,dtype = tf.float32)
        w = tf.add(w_mu,tf.multiply(epsilon,w_sigma))
        tf.add_to_collection('w_sample',w)
        
        epsilon = tf.random_normal([batch_size,1,500],0,1,dtype = tf.float32)
        b = tf.add(b_mu,tf.multiply(epsilon,b_sigma))
        tf.add_to_collection('b_sample',b)
        
        layer1 = tf.nn.tanh(tf.matmul(z,w) + b)
        
    #layer 2 for input x
    with tf.variable_scope('gen_layer2') as scope:
        w_mu,w_sigma,w_dist = _weight_inference('w',[500,500])
        b_mu,b_sigma,b_dst = _bias_inference('b',[500])
        
        epsilon = tf.random_normal([batch_size,500,500],0,1,dtype = tf.float32)
        w = tf.add(w_mu,tf.multiply(epsilon,w_sigma))
        tf.add_to_collection('w_sample',w)
        
        epsilon = tf.random_normal([batch_size,1,500],0,1,dtype = tf.float32)
        b = tf.add(b_mu,tf.multiply(epsilon,b_sigma))
        tf.add_to_collection('b_sample',b)
        
        layer2 = tf.nn.tanh(tf.matmul(layer1,w) + b)
        
    #output
    with tf.variable_scope('gen_output') as scope:
        w_mu,w_sigma,w_dist = _weight_inference('w',[500,x_dim])
        b_mu,b_sigma,b_dst = _bias_inference('b',[x_dim])
        
        epsilon = tf.random_normal([batch_size,500,x_dim],0,1,dtype = tf.float32)
        w = tf.add(w_mu,tf.multiply(epsilon,w_sigma))
        tf.add_to_collection('w_sample',w)
        
        epsilon = tf.random_normal([batch_size,1,x_dim],0,1,dtype = tf.float32)
        b = tf.add(b_mu,tf.multiply(epsilon,b_sigma))
        tf.add_to_collection('b_sample',b)
        
        output = tf.matmul(layer2,w) + b
        
    x_mu = tf.nn.sigmoid(output)
    
    return x_mu
    
latent_dim = 2
batch_size = 128
x_dim = 28*28
learning_rate_initial = 0.4
learning_decay = 0.8
global_step = tf.Variable(0,trainable=False)
x_ph = tf.placeholder(tf.float32,[batch_size,x_dim])

x_reshaped = tf.reshape(x_ph,[batch_size,1,x_dim])
z_mu,z_sigma = inference_network(x_reshaped)
#z_dist = distributions.Normal(mu = z_mu,sigma = z_sigma)
#z_prior = distributions.Normal(mu=tf.zeros((batch_size,latent_dim)),sigma = tf.ones((batch_size,latent_dim)))
#the reparameterisation trick
epsilon = tf.random_normal([batch_size,1,latent_dim],0,1)
z = tf.add(z_mu,tf.multiply(z_sigma,epsilon))

x_mu = generator_network(z)
x_mu = tf.reshape(x_mu,[batch_size,x_dim])
z_mu = tf.reshape(z_mu,[batch_size,latent_dim])
z_sigma = tf.reshape(z_sigma,[batch_size,latent_dim])

log_joint_prob = tf.reduce_sum(x_ph*tf.log(x_mu +1e-8) + \
                                    (1-x_ph)*tf.log(1-x_mu+1e-8),axis = 1)
#KL(q(z|x,y) || p(z))  
z_latent_loss = 0.5*tf.reduce_sum(1+tf.log(tf.square(z_sigma)+1e-10)
                               - tf.square(z_mu)
                                -tf.square(z_sigma),axis=1)

#z_log_prior = tf.reduce_sum(z_prior.log_pdf(z+1e-8),axis = 1)
#qz_log_prob = tf.reduce_sum(z_dist.log_pdf(z+1e-8),axis = 1)

#get the collection of weight sammples
w_samples = tf.get_collection('w_sample')
w_log_prior = 0
i = 0
for ele in tf.get_collection('weights_prior'):
    w_log_prior += tf.reduce_sum(tf.reduce_sum(ele.log_pdf(w_samples[i]),axis=1),axis=1)
    i+=1
    
qw_log_prob = 0
i = 0
for ele in tf.get_collection('w'):
    qw_log_prob += tf.reduce_sum(tf.reduce_sum(ele.log_pdf(w_samples[i]),axis=1),axis=1)
    i+=1

b_samples = tf.get_collection('b_sample')
b_log_prior = 0
i = 0
for ele in tf.get_collection('bias_prior'):
    b_log_prior += tf.reduce_sum(tf.reduce_sum(ele.log_pdf(b_samples[i]),axis=1),axis=1)
    i+=1
    
qb_log_prob = 0
i = 0
for ele in tf.get_collection('b'):
    qb_log_prob += tf.reduce_sum(tf.reduce_sum(ele.log_pdf(b_samples[i]),axis=1),axis=1)
    i+=1
    
ELBO = tf.reduce_mean(log_joint_prob + z_latent_loss + w_log_prior + b_log_prior - qw_log_prob -  qb_log_prob)

learning_rate = tf.train.exponential_decay(learning_rate_initial,global_step,10,learning_decay)
optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(-ELBO,global_step = global_step) 

sess = tf.Session()
sess.run(tf.global_variables_initializer())
n_epoch = 100
avg_loss_array = np.zeros(n_epoch)
for epoch in range(n_epoch):
    n_iter_per_epoch = int(1000/batch_size)
    avg_loss = 0.0
    for i in range(n_iter_per_epoch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        cost_val, _ ,temp= sess.run([ELBO,optimiser,z], feed_dict={x_ph: batch_xs})
        avg_loss +=cost_val
    avg_loss = avg_loss /n_iter_per_epoch
    avg_loss_array[epoch] = avg_loss
    
    if epoch %10 == 0:
        print("log p(x) >= {:0.3f}".format(avg_loss))
        print(temp)
        
plt.plot(range(0,n_epoch-1),avg_loss_array[1:])
plt.savefig('training_schedule.png')

batch_xs, _ = mnist.test.next_batch(batch_size)
reconstruct_x = sess.run(x_mu, feed_dict={x_ph: batch_xs})

print(reconstruct_x[0])

plt.figure(figsize=(8, 12))
for i in range(5):

    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(batch_xs[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(reconstruct_x[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.savefig('test_cases.png')

x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = sess.run(z_mu,feed_dict={x_ph:x_sample})
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1),cmap='rainbow')
plt.colorbar()
plt.grid()
plt.savefig('latent_space.png')

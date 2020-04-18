import torch
import tensorflow as tf
import numpy as np
import statistics 
from torch.nn import functional as F
import torch.distributions as tdist


import visual_visdom
import visual_plt
import utils
import matplotlib.pyplot as plt

#########################################################
## maindnc xsm code                                    ##
#########################################################

def maindnc(self, size, batch_index,z0,task,tasks,t_label):
 
    '''
    if list(z0.size())[0]!=0:
        #estimation of the mean and variance
        zx=z0
        mean=(zx.mean(dim=1)).mean(dim=0)
        var=(zx.std(dim=1)).mean(dim=0)
        #print('xsm mean',mean)
        #print('xsm xsm var',var)

    else:

        #estimate in begining
        mean=0
        var=1.6
    '''
    
    mean=0
    var=1.6
    n = tdist.Normal(mean, var)
    z1 =n.sample((size, self.z_dim)).to(self._device())

    t_label =n.sample((size, self.z_dim)).to(t_label)

   
    if (task<=round((tasks+1)/2)):
        z2=torch.cat((z0,z1,z1), 0) 
    else:
        z2=torch.cat((z0,z1), 0)  
    

    

    dl=64
    m=int(list(z1.size())[0]/dl)
    n=int(list(z0.size())[0]/dl)

 
    if list(z0.size())[0]!=0:

        for i in range(m):
            rows1 =z1[i*dl:i*dl+dl,:]

            tensor_similarity=0
            for j in range(n):
                    rows2 = z0[j*dl:j*dl+dl,:]
                    x = rows1
                    y = rows2
                    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    tensor_similarity+=torch.sum(cos(x, y))


            if (tensor_similarity<0):
                z2=torch.cat((z2,torch.reshape(rows1, (dl, 100))), 0)  


    image_tensor=z1


    print('xsm xsm xsm xsm z2',z2[:,:(-1)])

    plt.imsave('./plots/save.png', image_tensor.numpy() , cmap='gray')


    if  batch_index==2000:

        torch.save(z2, 'dnc.pt')


    return z2,t_label




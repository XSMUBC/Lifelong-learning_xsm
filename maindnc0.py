import torch
import tensorflow as tf
import numpy as np
import statistics 
from torch.nn import functional as F
import torch.distributions as tdist


#########################################################
## maindnc xsm code                                    ##
#########################################################

def maindnc(self, size, batch_index,z0):


    #random.sample([1, 2, 3, 4, 5],  3)
    #zx = torch.rand(size*50, self.z_dim*50).to(self._device())
    #zx=torch.distributions.Uniform(self._device()).sample()
    #print('xsm xsm zx',zx)

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
    #var=(size/self.z_dim)*(size/self.z_dim)
    #print('xsm xsm ',size/self.z_dim)
    n = tdist.Normal(mean, var)
    z1 =n.sample((size, self.z_dim)).to(self._device())

    #z1 = torch.randn(size, self.z_dim).to(self._device())
    #torch.save(z1, 'file.pt')
    #if batch_index==1:
    #read operation
        #z0=torch.load('file.pt')
    z2=torch.cat((z0,z1), 0)  

 


    dl=32
    m=int(list(z1.size())[0]/dl)
    n=int(list(z0.size())[0]/dl)
    #print('xsm m',m)
    #print('xsm n',n)



    if list(z0.size())[0]!=0:

        for i in range(m):
            rows1 =z1[i*dl:i*dl+dl,:]
            #print('rows',rows1)
            for j in range(n):


                    rows2 = z0[j*dl:j*dl+dl,:]
                    #print('rows',rows2)

                    #print('z0',z0.size())
                    #print('z1',z1.size())


                    x = rows1
                    y = rows2
                    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    tensor_similarity=torch.sum(cos(x, y))
                    #print('tensor_similarity',tensor_similarity)
                    if (tensor_similarity<0):
                        z2=torch.cat((z2,torch.reshape(rows1, (dl, 100))), 0) 














    #z2=z1
    #print('z0',z0)
    #print('z1',z1)


    # operation on the dnc memory
    #z2=torch.unique(z2, dim=0)   

    if batch_index==2000:
        # write operation

        #z2=memope(self, size,z2,z1,z0)


        torch.save(z2, 'dnc.pt')
    #z=torch.load('file.pt')


    #print('xsm z1 size',z1.size())
    #print('xsm z size',z2.size())
    #print('xsm z0 ',z0)
    #print('xsm z1 ',z1)
    #print('xsm z ',z2)


    return z2







import torch
import tensorflow as tf
import numpy as np
import math
import statistics 
from torch.nn import functional as F
import torch.distributions as tdist

import argparse
import os
import numpy as np
import time
import torch
from torch import optim
import visual_plt
import utils
from param_stamp import get_param_stamp, get_param_stamp_from_args
import evaluate
from data import get_multitask_experiment
from encoder import Classifier
#from vae_models import AutoEncoder
import callbacks as cb
from train import train_cl
from continual_learner import ContinualLearner
from exemplars import ExemplarHandler
from replayer import Replayer

import argparse

from param_stamp import get_param_stamp, get_param_stamp_from_args

import visual_visdom

import pandas as pd 
import visual_plt
import utils
import matplotlib.pyplot as plt

#########################################################
## maindnc xsm code                                    ##
#########################################################

def maindnc(self, size, batch_index,z0,task,tasks,t_label,impor,pi,model,task_id):
 
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
        z2=torch.cat((z0,z1,z1), 0)   



    

    dl=1
    tap=0.5  # determined by L-Cureve method 
    #impor=[0] * list(z0.size())[0]
    #pi=0.5
    mn=0
    m=int(list(z1.size())[0]/dl)
    n=int(list(z0.size())[0]/dl)
    zz=torch.tensor([])
    z3=z2
    n_samples=900

    #image_tensor=z1
    #print('xsm xsm xsm xsm impor',impor)
    #plt.imsave('./plots/save.png', image_tensor.numpy() , cmap='gray')  

    #torch.save(z2, 'dnc.pt')
    if  batch_index==2000:

        #z0=torch.load('dnc.pt')

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


                if (tensor_similarity<-tap):#*list(z0.size())[0]
                    z2=torch.cat((z2,torch.reshape(rows1, (dl, 100))), 0)


                if (tensor_similarity>tap):#*list(z0.size())[0]
                    impor[i]+=math.exp(-pi)
              
            impor=impor+[1/list(z0.size())[0]]*(list(z2.size())[0]-list(z0.size())[0])
            sum_impor=sum(impor)


            impor2 = pd.read_csv('impor.csv', delimiter=',', header=0)  # xsm xsm write






            #print('xsm impor size',np.prod(impor2.shape))
            #print('xsm impor dd',impor2)
            #impor2=np.zeros((list(z2.size())[0], 2))

            #impor=impor
            for j in range(list(z2.size())[0]):
                #impor2[j,0]=impor[j]/sum_impor
                impor[j]=impor[j]/sum_impor
                #impor2[j,1]=task_id
                arr2 = np.array([[impor[j]/sum_impor, task_id]])
                #print('xsm arr2 dd',arr2)
                impor2=np.append(impor2,arr2, axis=0)

           
            #for j in range(list(z2.size())[0]):
                #if (impor[j]<sum(impor)/list(z2.size())[0]):
                    #z2=torch.cat((z2,z2[j*dl:j*dl+dl,:]), 0)
            #print('xsm impor',impor)



            #print('xsm impor task task',task)            







            '''
      
            if (task_id==3):

                impormn1=np.array([[1/128, 1]])
                impormn2=np.array([[1/128, 2]])           

                

                for mn in range(size):

                    #impor2[j,1]=task_id
                    arrmn = np.array([[1/size, 1]])
                    #print('xsm arr2 dd',arr2)
                    impormn1=np.append(impormn1,arrmn, axis=0)
                    impormn2=np.append(impormn2,arrmn, axis=0)



                impormn=np.append(impormn1,impormn2, axis=0)

                impor3=np.append(impormn,impor2, axis=0)
            else:
                impor3=impor2
          
            '''


            np.savetxt('impor.csv', impor2, delimiter=',')

            for mn in range(n_samples):
                kkk=np.random.choice(np.arange(0,list(z2.size())[0]),p=impor)
            #kkk=_impor_choice(z2,impor,600)

            #for j in range(kkk):                    

                # sampling the vector with impor numpy.arange(1, 7)
                zz=torch.cat((zz,z2[kkk*dl:kkk*dl+dl,:]), 0)

            z3=zz
    
              

        
        parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
        parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
        parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
        parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
        parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
        parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
        parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

        # expirimental task parameters
        task_params = parser.add_argument_group('Task Parameters')
        task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST'])
        task_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
        task_params.add_argument('--tasks', type=int, default=5, help='number of tasks')

        # specify loss functions to be used
        loss_params = parser.add_argument_group('Loss Parameters')
        loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")
        loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'
                                                                            ' examples (only if --bce & --scenario="class")')

        # model architecture parameters
        model_params = parser.add_argument_group('Model Parameters')
        model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
        model_params.add_argument('--fc-units', type=int, default=400, metavar="N", help="# of units in first fc-layers")
        model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
        model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
        model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
        model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                           " (instead of a 'multi-headed' one)")

        # training hyperparameters / initialization
        train_params = parser.add_argument_group('Training Parameters')
        train_params.add_argument('--iters', type=int, default=2000, help="# batches to optimize solver")

        #train_params.add_argument('--iters', type=int, default=20, help="# batches to optimize solver")
        train_params.add_argument('--lr', type=float, default=0.001, help="learning rate")
        train_params.add_argument('--batch', type=int, default=128, help="batch-size")
        train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

        # "memory replay" parameters
        replay_params = parser.add_argument_group('Replay Parameters')
        replay_params.add_argument('--feedback', action="store_true", help="equip model with feedback connections")
        replay_params.add_argument('--z-dim', type=int, default=100, help='size of latent representation (default: 100)')
        replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
        replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
        replay_params.add_argument('--distill', action='store_true', help="use distillation for replay?")
        replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
        # -generative model parameters (if separate model)
        genmodel_params = parser.add_argument_group('Generative Model Parameters')
        genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='size of latent representation (default: 100)')
        genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
        genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
        # - hyper-parameters for generative model (if separate model)
        gen_params = parser.add_argument_group('Generator Hyper Parameters')
        gen_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")
        gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: lr)")

        # "memory allocation" parameters
        cl_params = parser.add_argument_group('Memory Allocation Parameters')
        cl_params.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
        cl_params.add_argument('--lambda', type=float, default=5000.,dest="ewc_lambda", help="--> EWC: regularisation strength")
        cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
        cl_params.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
        cl_params.add_argument('--gamma', type=float, default=1., help="--> EWC: forgetting coefficient (for 'online EWC')")
        cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
        cl_params.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
        cl_params.add_argument('--c', type=float, default=0.1, dest="si_c", help="--> SI: regularisation strength")
        cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
        cl_params.add_argument('--xdg', type=float, default=0., dest="gating_prop",help="XdG: prop neurons per layer to gate")

        # exemplar parameters
        icarl_params = parser.add_argument_group('Exemplar Parameters')
        icarl_params.add_argument('--icarl', action='store_true', help="bce-distill, use-exemplars & add-exemplars")
        icarl_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")
        icarl_params.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task dataset")
        icarl_params.add_argument('--budget', type=int, default=2000, dest="budget", help="how many exemplars can be stored?")
        icarl_params.add_argument('--herding', action='store_true', help="use herding to select exemplars (instead of random)")
        icarl_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

        # evaluation parameters
        eval_params = parser.add_argument_group('Evaluation Parameters')
        eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
        eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
        eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
        eval_params.add_argument('--loss-log', type=int, default=200, metavar="N", help="# iters after which to plot loss")
        eval_params.add_argument('--prec-log', type=int, default=200, metavar="N", help="# iters after which to plot precision")
        eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
        eval_params.add_argument('--sample-log', type=int, default=500, metavar="N", help="# iters after which to plot samples")
        eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")



        args = parser.parse_args()
        scenario = args.scenario

        # Prepare data for chosen experiment
        (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
            name=args.experiment, scenario=scenario, tasks=task, data_dir=args.d_dir,
            verbose=True, exception=True if args.seed==0 else False,
        )

        args.tasks=task
        #----------------------#
        #----- EVALUATION -----#
        #----------------------#
        print("\n\n--> Evaluation ({}-incremental learning scenario):".format(args.scenario))  #xsm xsm 
        # Evaluate precision of final model on full test-set  # xsm test in here
        precs = [evaluate.validate(
            model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=False,
            allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
        ) for i in range(args.tasks)]
        print("\n Precision on test-set (softmax classification):")
        for i in range(args.tasks):
            print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
        average_precs = sum(precs) / args.tasks
        print('=> average precision over all {} tasks: {:.4f}'.format(args.tasks, average_precs))
        # -with exemplars
        if args.use_exemplars:
            precs = [evaluate.validate(
                model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=True,
                allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
            ) for i in range(args.tasks)]
            print("\n Precision on test-set (classification using exemplars):")
            for i in range(args.tasks):
                print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
            average_precs_ex = sum(precs) / args.tasks
            print('=> average precision over all {} tasks: {:.4f}'.format(args.tasks, average_precs_ex))
        print("\n")

    torch.save(z3, 'dnc.pt')

    return z3,t_label,impor

'''

def _impor_choice(inputs,impor, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    #uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)
    uniform_log_prob = tf.expand_dims(impor, 0)

    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")


def tf_random_choice_no_replacement_v1(one_dim_input, num_indices_to_drop=3):

    input_length = tf.shape(one_dim_input)[0]

    # create uniform distribution over the sequence
    # for tf.__version__<1.11 use tf.random_uniform - no underscore in function name
    uniform_distribution = tf.random.uniform(
        shape=[input_length],
        minval=0,
        maxval=None,
        dtype=tf.float32,
        seed=None,
        name=None
    )

    # grab the indices of the greatest num_words_to_drop values from the distibution
    _, indices_to_keep = tf.nn.top_k(uniform_distribution, input_length - num_indices_to_drop)
    sorted_indices_to_keep = tf.contrib.framework.sort(indices_to_keep)

    # gather indices from the input array using the filtered actual array
    result = tf.gather(one_dim_input, sorted_indices_to_keep)
    return result



'''


'''

np.random.choice([1,2,3,5], 1, p=[0.1, 0, 0.3, 0.6, 0])

elems = tf.convert_to_tensor([1,2,3,5])
samples = tf.multinomial(tf.log([[1, 0, 0.3, 0.6]]), 1) # note log-prob
elems[tf.cast(samples[0][0], tf.int32)].eval()
Out: 1
elems[tf.cast(samples[0][0], tf.int32)].eval()
Out: 5

'''


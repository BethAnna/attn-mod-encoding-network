#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:14:06 2024

@author: bethannajones
Elisabeth "BethAnna" Thompson (Jones)
"""



'''

    The main script to run a simulation of the attention-modulated retention network, as 
designed and outlined in Thompson's dissertation [1]. For further reference, 
see previous work [2-3]. 
    

    The implemented network is designed to encode (randomly generated) input within its dynamcial 
activity while also retaining previous input for as long as possible. It's optimal evolution 
is determined through proximal gradient flow, shown to converge more quickly than traditional
subgradient methods [4]. The proximal gradient flow relies on the proximal operator 
        prox_g(v) = argMin_x(  g(x) - 0.5*norm(x-v)  ),    
where we have closed, proper, convex f(x) and g(x). Here f(x) is the sum of L2 "decoding 
error" and L2 "history encoding error", and g(x) is the non-differentiable L1 sparsity 
regularizer. 



BIBLIOGRAPHY:

[1]  B. Thompson, ‘Synthesis of neuronal network dynamics for optimal stimulus encoding and 
retention’. Washington University in St. Louis, 2024. https://doi.org/10.7936/96fn-0k14

[2]  B. Jones, L. Snyder, and S. Ching, ‘Heterogeneous forgetting rates and greedy allocation 
in slot-based memory networks promotes signal retention’, Neural Comput., vol. 36, no. 5, 
pp. 1022–1040, Apr. 2024. https://doi.org/10.1162/neco_a_01655

[3]  B. Jones and S. Ching, ‘Synthesizing network dynamics for short-term memory of impulsive 
inputs’, in 2022 IEEE 61st Conference on Decision and Control (CDC), Cancun, Mexico, 
2022. http://doi.org/10.1109/CDC51059.2022.9993238

[4]  N. Parikh and S. Boyd. Proximal algorithms, ser. foundations and trends (r) in 
optimization, 2013. 

[5]  M. Kafashan and S. Ching, ‘Recurrent networks with soft-thresholding nonlinearities for 
lightweight coding’, Neural Netw., vol. 94, pp. 212–219, Oct. 2017. 


'''


#=========================================================================================
#=========================================================================================
#%% IMPORT 
#=========================================================================================
#=========================================================================================


#-----------------------------
''' System '''
#-----------------------------
import os



#-----------------------------
''' Modeling & Data '''
#-----------------------------
import torch 
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


import numpy as np 
import random


import math 




#----------------------------
''' Visualization '''
#-----------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm.notebook import tqdm

import time 


#-----------------------------
''' Saving '''
#-----------------------------
import pickle 




#=========================================================================================
#=========================================================================================
#%% chg CWD
#=========================================================================================
#=========================================================================================


#=============================================================
''' Function: make the current directory the working dir. '''
#=============================================================
def changeWD(desiredDir, listFiles=True):
    ''' Make desiredDir the current working directory '''

    os.chdir(desiredDir)

    currDir = os.getcwd()
    currFiles = os.listdir(currDir)

    print('\nModified current working directory. Now:\n', desiredDir)

    if listFiles:
        print('\nWith files:')
        for i in range(len(currFiles)):
            print('\t', currFiles[i])

    return currDir







#=============================================================
''' Now do it '''
#=============================================================

print('\nCurrent Working Directory:\n', os.getcwd())

''' The directory this file is in '''
filename = os.path.basename(__file__)

fullPath = os.path.realpath(__file__)
fileDir = fullPath[0: -len(filename)-1]
# print( '\nFile directory:\n', fileDir )


currDir = changeWD(fileDir, listFiles=False)

#=============================================================




#=============================================================
''' Import my code '''
#=============================================================
from PGM_model import *
from PGM_analysis import * 

from PGM_trainAndTest import * 

# from PGM_saving import * 


# from PGM_objects import *
# from PGM_matrices import *





# from PGM_update import * 










#=========================================================================================
#=========================================================================================
#%% Settings
#=========================================================================================
#=========================================================================================


referenceFolder = 'currentInfo'




#------------------------------------------
seedNum = 0
seedNum = 10

torch.manual_seed( seedNum )
random.seed( seedNum )
np.random.seed( seedNum )
#------------------------------------------



#-----------------------------------------------------------------------------------------
''' Dictionary of simulation settings '''
#-----------------------------------------------------------------------------------------
simOptions = { 
    
        
        ##--------------------------------------------
        ## Data 
        ##--------------------------------------------
    
        'loadPrevData' : True,                          ## Stored in referenceFolder
        'loadPrevData' : False, 

        'circleStims' : True,                           ## The stims are 2D points on a (unit) circle
        # 'circleStims' : False, 

        # 'endWithZeroStim' : True,                       ## The last stim is a 0 vector
        'endWithZeroStim' : False, 
        
        
        ##--------------------------------------------
        ## Attention 
        ##--------------------------------------------
        
        
        'retentionAlpha' : 0,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.0001,                         ## lower retentionAlpha --> higher retention 
        'retentionAlpha' : 0.001,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.002,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.003,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.004,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.005,                         ## lower retentionAlpha --> higher retention 
        
        # 'retentionAlpha' : 0.05,                         ## lower retentionAlpha --> higher retention 
        'retentionAlpha' : 0.02,                         ## lower retentionAlpha --> higher retention 
        'retentionAlpha' : 0.01,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.15,                         ## lower retentionAlpha --> higher retention 

        # 'retentionAlpha' : 0.9,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.8,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.7,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.6,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.5,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.4,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.3,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.2,                         ## lower retentionAlpha --> higher retention 
        # 'retentionAlpha' : 0.1,                         ## lower retentionAlpha --> higher retention 
        
        
        'encodingAlpha' : 'context',                          ## lower encodingAlpha --> worse encoding performance
        'encodingAlpha' : 1,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.95,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.9,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.85,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.8,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.75,                          ## lower encodingAlpha --> worse encoding performance
        
        # 'encodingAlpha' : 0.7,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.6,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.5,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.4,                          ## lower encodingAlpha --> worse encoding performance
        # 'encodingAlpha' : 0.3,                          ## lower encodingAlpha --> worse encoding performance
        
        
        
        
        ##--------------------------------------------
        ## Training '''
        ##--------------------------------------------
         
        'learnSingleStimUntilConvergence' : True,       ## Train on a single stim up to maxIter times before moving on to next stim (epoch)
        # 'learnSingleStimUntilConvergence' : False, 
                         
        # 'Kafashan' : True,                            ## Use learning Algorithm 1by Kafashan, Ching (2017)
        'Kafashan' : False, 
        
        'trainMatsDirectly' : True,                     ## train D,H directly (instead of via W, Md, Ms)
        # 'trainMatsDirectly' : False,
        
        'requireStable' : True,                         ## Require initial D,H to be stable (nonnegative eigvals)
        # 'requireStable' : False,
        
        'includeDelayInTraining' : True,                ## Include the retention period in the training process 
        # 'includeDelayInTraining' : False,
        
        
        'trainBeforeDecay' : True,                      ## Do backprop training for each stimulus immediately after encoding phase
        # 'trainBeforeDecay' : False,
        
        'trainAfterDecay' : True,                       ## Do backprop training for each stimulus immediately after retention phase
        # 'trainAfterDecay' : False,
        
        
        
        ##--------------------------------------------
        ## Misc
        ##--------------------------------------------
        
        'differentProxSteps' : True,                    ## Number of (encoding) prox. steps is different between training / testing phases 
        'differentProxSteps' : False, 
         
        # 'storeAllConnMats' : True,                      ## Store mats for ALL epochs  --->  ** memory-heavy! **
        'storeAllConnMats' : False, 
        
        'takeAwayStim' : True,                          ## Set beta=0 during the retention phase 
        # 'takeAwayStim' : False, 
         
        'takeAwayRef' : True,                           ## Set refState=0 during the retention phase 
        'takeAwayRef' : False, 
        # 
        
        # 'attention' : True,                             ## total cost is the sum of encoding, retention, + sparsity terms
        # # 'attention' : False,
        
        'useQ2Also' : False,                            ## Include a duplicate of the history term, but with q=2
        # 'useQ2Also' : True,
        
        'processingTermWithD' : True,                   ## Include D in the processing term:    
        'processingTermWithD' : False,                  ##              D*(refState - H*currState)   VERSUS   refState - H*currState

        'addStimSpaceProcessTerm' : True,               ## Include D in the processing term:                 
        'addStimSpaceProcessTerm' : False,              ##              D*(refState - H*currState)   VERSUS   refState - H*currState

        'noNullDynam' : True,                           ## Only use the proximal portion of the convex combo dynamics
        'noNullDynam' : False, 

              }

    
    
    
''' No null dynamics '''
#--------------------------------------------
if simOptions[ 'noNullDynam' ]:
    rWeightList = [ 0 ]   
    nRWeights = len( rWeightList )
    
    print( 'Setting up for no null dynamics (only proximal evolution)' )
    simOptions[ 'retentionAlpha' ] = 0
    simOptions[ 'encodingAlpha' ] = 1
    
    
    

''' Processing cost term '''
#--------------------------------------------
if simOptions['processingTermWithD'] and simOptions['addStimSpaceProcessTerm']:
    raise Exception( 'Are you sure you want processingTermWithD AND addStimSpaceProcessTerm to both be true?' )






#-----------------------------------------------------------------------------------------
''' The device to use '''
#-----------------------------------------------------------------------------------------

device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu'  )
# device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'gpu'  )
print( device )

simOptions['device'] = device



# convergedTimeInds = [ (tk - 1) for tk in stimTimeInds[1::] ]   +   [ nTimes-1 ]













testingWasInterrupted = False







#=========================================================================================
#=========================================================================================
#%% WEIGHTS
#=========================================================================================
#=========================================================================================



# #-----------------------------------------------------------------------------------------
# ''' Attentional parameter ''' 
# #-----------------------------------------------------------------------------------------
# alpha = 1                           ## encoding (and sparsity) only 
# alpha = 0.9
# alpha = 0.75
# # alpha = 0.5
# # alpha = 0.1
# # alpha = 0                           ## retention (and sparsity) only 


# alphaType = 'fixed'
# alphaType = 'context'

# simOptions[ 'alphaType' ] = alphaType




#=========================================================================================
''' The cost function (loss) weights '''
#=========================================================================================

## Error term:  ||  x(t) - D*r(t)  ||_2^2
#--------------------------------------------
# eWeight = 
errWeightList = [ 0, 0.01, 1 ]   
# errWeightList = [ 2.5 ]   
errWeightList = [ 0.5 ]   
errWeightList = [ 1 ]   
errWeightList = [ 1.5 ]   
errWeightList = [ 2 ]   
errWeightList = [ 3 ]   
errWeightList = [ 4 ]   
# errWeightList = [ 5 ]   
# errWeightList = [ 10 ]   
# errWeightList = [ 0 ]   
nErrWeights = len( errWeightList )






## Efficiency term:  ||  r(t)  ||_2^2
#--------------------------------------------
# eWeight = 1
effWeightList = [ 0, 0.01, 1 ]   
effWeightList = [ 0.01 ]   
effWeightList = [ 0.02 ]   
# effWeightList = [ 0.1 ]   
# effWeightList = [ 0.15 ]   
# effWeightList = [ 1 ]  
# effWeightList = [ 5 ]    
# effWeightList = [ 0 ]   
nEffWeights = len( effWeightList )



## Sparsity term (L1):  ||  r(t)  ||_1
#--------------------------------------------
# sWeight = 0                 
# sWeightList = [ 0, 0.01, 0.02, 0.03, 0.04 ]
sWeightList = [ 0, 0.01, 0.04 ]
# sWeightList = [ 0.004 ]
sWeightList = [ 0.01 ]
# sWeightList = [ 0.02 ]
# sWeightList = [ 0.04 ]
# sWeightList = [ 1 ]
# sWeightList = [ 0 ]
nSWeights = len( sWeightList )



# ##  History term:       ||  r(t-1) - H*r(t)  ||_2^2
#--------------------------------------------
hWeightList = [ 0.25 ]       
hWeightList = [ 0.5 ]       
hWeightList = [ 1 ]       
hWeightList = [ 2 ]               
# hWeightList = [ 3 ]         
hWeightList = [ 5 ]    
hWeightList = [ 10 ]         
# hWeightList = [ 12 ]         
# hWeightList = [ 15 ]    
# hWeightList = [ 20 ]    
# hWeightList = [ 30 ]    
# hWeightList = [ 0 ]    
nHWeights = len( hWeightList )        




## Retention: alpha value during delay
#--------------------------------------------
if not simOptions[ 'noNullDynam' ]:

    rWeightList = [ 0, 0.01, 1 ]   
    rWeightList = [ 1 ]     
    
    nRWeights = len( rWeightList )



    
    
    



## Frugality term:  ||  r(t) - A*r(t-1)  ||_2^2
#--------------------------------------------
# fWeight = 0                     
fWeightList = [ 0 ]
# fWeightList = [ None ] 






#-----------------------------------------------------------------------------------------
''' Store the weights in a single reference variable '''
#-----------------------------------------------------------------------------------------
weightCombos = makeWeightComboList( errWeightList, effWeightList, sWeightList, hWeightList, rWeightList, fWeightList )
# weightCombos = makeWeightComboList( errWeightList, effWeightList, sWeightList, hWeightList, fWeightList )
nWeightCombos = len( weightCombos )



print( 'weightCombos: ', weightCombos )





#=========================================================================================
#=========================================================================================
#%% PARAMETERS
#=========================================================================================
#=========================================================================================


N = 15                  ## dimension of NETWORK 
# N = 30                  ## dimension of NETWORK 
# N = 50                  ## dimension of NETWORK 
d = 1                   ## dimension of afferent STIMULI (input)

initState = x0 = torch.zeros( [N,1] )
initState = x0 = torch.rand( [N,1] ) 



# ## From 'varsAndFigs/retrainOnStim10/eAlp0.9rAlp0.001/eSteps25rSteps500/ ...
# ##                      prox25_stable_lr0.005_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# initState = torch.tensor( [0.1535, 0.3224, 0.7908, 0.7567, 0.1129, 0.7221, 0.1385, 0.9382, 0.4488,
#                                     0.5681, 0.6448, 0.1155, 0.8566, 0.1693, 0.0540] )


# ## From 'varsAndFigs/retrainOnStim10/eAlp0.9rAlp0.07/eSteps25rSteps500/ ...
# ##                      prox25_stable_lr0.005_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000
# initState = torch.tensor([0.4251, 0.5793, 0.6092, 0.9706, 0.0054, 0.8556, 0.4518, 0.7131, 0.0349,
#         0.4928, 0.6490, 0.6910, 0.6111, 0.0122, 0.9717] )


## From 'varsAndFigs/retrainOnStim10/eAlp0.9rAlp0.002/...
initState = torch.tensor([[0.4581, 0.4829, 0.3125, 0.6150, 0.2139, 0.4118, 0.6938, 0.9693, 0.6178,
          0.3304, 0.5479, 0.4440, 0.7041, 0.5573, 0.6959]])








#=========================================================================================
''' Training '''
#=========================================================================================
nEpochs = nTrainingBetas = 50000        ## 20,000
# nEpochs = nTrainingBetas = 20000        ## 20,000
nEpochs = nTrainingBetas = 10000        ## 10,000
nEpochs = nTrainingBetas = 7000         ##  5,000
nEpochs = nTrainingBetas = 5000         ##  5,000
# nEpochs = nTrainingBetas = 4000         ##  5,000
# nEpochs = nTrainingBetas = 3000         ##  5,000
nEpochs = nTrainingBetas = 2000         ##  2,000
# nEpochs = nTrainingBetas = 1500         ##  2,000
# nEpochs = nTrainingBetas = 1300         ##  2,000
# nEpochs = nTrainingBetas = 1000         ##  1,000
# nEpochs = nTrainingBetas = 800       
# nEpochs = nTrainingBetas = 600       
# nEpochs = nTrainingBetas = 500       
# nEpochs = nTrainingBetas = 400       
# nEpochs = nTrainingBetas = 300       
# nEpochs = nTrainingBetas = 200       
# nEpochs = nTrainingBetas = 100       
# nEpochs = nTrainingBetas = 50       
# nEpochs = nTrainingBetas = 30     
# nEpochs = nTrainingBetas = 25      
# nEpochs = nTrainingBetas = 10
# nEpochs = nTrainingBetas = 5



# epochNumsToTest = getEpochNumsToTest( nEpochs )
# nEpochNumsToTest = len( epochNumsToTest )

# simOptions[ 'epochNumsToTest' ] = epochNumsToTest




''' Training iterations per single stim '''
#----------------------------------------------------------
# maxIter = 400
maxIter = 200
maxIter = 100
maxIter = 50
# maxIter = 25

# maxIter = 20
# maxIter = 15
# maxIter = 10
# maxIter = 5
# maxIter = 1



''' Time steps with access to both x(t), r(t-1)  '''
#----------------------------------------------------------
nEncodSteps_training = 1000                   
nEncodSteps_training = 750                   
nEncodSteps_training = 500                   
# nEncodSteps_training = 250                 
nEncodSteps_training = 200                   
nEncodSteps_training = 100                 
nEncodSteps_training = 50                    
# nEncodSteps_training = 25                    
# nEncodSteps_training = 20                    



''' How quickly the optimizer trains the network '''
#----------------------------------------------------------
learningRate = 0.001           ## 1e-3
learningRate = 0.005          ## 5e-3
# learningRate = 0.006          ## 5e-3
# learningRate = 0.01           ## 1e-2
# learningRate = 0.1            ## 1e-1

# simOptions[ 'learningRate' ] = learningRate 




#=========================================================================================
''' Testing '''
#=========================================================================================

nTestBetas = 1000
nTestBetas = 500
nTestBetas = 400
# nTestBetas = 300
# nTestBetas = 250
nTestBetas = 200
# nTestBetas = 100
# nTestBetas = 75
nTestBetas = 50
# nTestBetas = 25
# nTestBetas = 10
# nTestBetas = 9
# nTimes = (nTestBetas * nProxSteps) + 1 






nRetenSteps_train = nRetenSteps = 500
nRetenSteps_train = nRetenSteps = 200
# nRetenSteps_train = nRetenSteps = 150
nRetenSteps_train = nRetenSteps = 100
# nRetenSteps_train = nRetenSteps = 75
# nRetenSteps_train = nRetenSteps = 50
# nRetenSteps_train = nRetenSteps = 0

nRetenSteps_ext = 1000
nRetenSteps_ext = 600
nRetenSteps_ext = 500
nRetenSteps_ext = 300
nRetenSteps_ext = 150
nRetenSteps_ext = 75
nRetenSteps_ext = nRetenSteps_train
# nRetenSteps_ext = 2000



# nRetenSteps = 1000
# nRetenSteps = 500
# nRetenSteps = 200
# nRetenSteps = 100
# nRetenSteps = 70
# nRetenSteps = 50
# nRetenSteps = 30
# nRetenSteps = 10
# nRetenSteps = 0
# nRetenSteps = nEncodSteps_training

# if (not simOptions['takeAwayRef']) and (not simOptions['takeAwayStim']):
#     nRetenSteps = 0

# simOptions[ 'nRetenSteps' ] = nRetenSteps
    


''' Time steps with access to both x(t), r(t-1) '''
#----------------------------------------------------------
nEncodSteps_testing = nEncodSteps_training

if simOptions['differentProxSteps']:
    nEncodSteps_testing = int( nEncodSteps_training / 3 )    
    nEncodSteps_testing = int( nEncodSteps_training / 2 )   
    nEncodSteps_testing = int( nEncodSteps_training * 2 )   
    
# simOptions[ 'nEncodSteps_testing' ] = nEncodSteps_testing







#=========================================================================================
''' Kafashan Training Parameters '''
#=========================================================================================
connectionVars = { 'tau_w' : 1e-5,
                    'tau_ms' : 1e-5,
                    'tau_md' : 1e-3,
                   
                    'eps_a' : 0.1,
                    'eps_b' : 0.1,
                   
                    'alpha' : 0.2,
                    # 'alpha' : 0.8,
                    
                    'A' : torch.zeros( [d,N] ),
                    'B' : torch.zeros( [N,N] ), 
                  }




#=========================================================================================
''' Store it all '''
#=========================================================================================

parameterDict = {                   
                   'initState' : initState,

                   # 'nEncodSteps_training' : nEncodSteps_training,           ## nProxSteps when stim and refState ARE available - TRAINING
                   
                   'nEpochs' : nEpochs,                                     ## Number of training epochs -- each epoch is trained over a single stim
                   # 'learningRate' : learningRate,                           ## Learning rate for the optimizer 
                   # 'maxIter' : maxIter,                                     ## Number of times to train over a single stim
                   
                   # 'nEncodSteps_testing' : nEncodSteps_testing,             ## nProxSteps when stim and refState ARE available - TESTING
                   # 'nRetenSteps' : nRetenSteps,                             ## nProxSteps when stim and refState ARE NOT available - TESTING (only)
                   'nTotalEvolSteps' : nEncodSteps_testing + nRetenSteps,   ## total time the system evolves over a single stim
                   
                   
                   # 'alpha': alpha,                                          ## attentional parameter:  1 is encoding, 0 is retention
                   
                   # 'stimScale' : 1,                                         ## relative 
                   
              }





#=========================================================================================
#%%''' Print the settings/parameters   ''' 
#=========================================================================================


simOptions['weightCombos'] = weightCombos


simOptions[ 'parameters' ] = parameterDict



nEpochs = simOptions['parameters']['nEpochs']
epochNumsToTest = getEpochNumsToTest( nEpochs )
nEpochNumsToTest = len( epochNumsToTest )
simOptions[ 'epochNumsToTest' ] = epochNumsToTest


simOptions[ 'learningRate' ] = learningRate 

simOptions[ 'nEncodSteps' ] = nEncodSteps_training
simOptions[ 'nEncodSteps_testing' ] = nEncodSteps_testing

# simOptions[ 'nRetenSteps' ] = nRetenSteps
simOptions[ 'nRetenSteps' ] = nRetenSteps_train
simOptions[ 'nRetenSteps_ext' ] = nRetenSteps_ext


simOptions[ 'maxIter' ] = maxIter




printSimOptions2( simOptions )








#-----------------------------------------------------------------------------------------
''' Basic list/dict of rainbow colors '''
#-----------------------------------------------------------------------------------------

def discreteRainbowColors( nColors ):
    

    cmap = mpl.colormaps['hsv']
    cmap = cmap.resampled(  nColors + 2  )            ## +2 to avoid the red at the end 
    
    colors = [  cmap(i) for i in range( nColors )  ]
    
    
    return colors




# colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]
colors = discreteRainbowColors(  len( epochNumsToTest )  )



colorDict = { }

for i in range( len(colors) ):
    colorDict[ i ] = colors[ i ] 






#=========================================================================================
#=========================================================================================
#%% Training
#=========================================================================================
#=========================================================================================


nEpochs = simOptions['parameters']['nEpochs']


#-----------------------------------------------------------------------------------------
''' Create the model and the stimuli '''
#-----------------------------------------------------------------------------------------
if simOptions[ 'loadPrevData' ]:
    
    saveDir = os.path.join( currDir, referenceFolder ) 
    
    [ trainingModel, trainingInput, testingInput ] = initModelAsBefore( saveDir )
    d = trainingInput.signalDim         ## May have been modified depending on if circleStims

    
    validLoadData = checkLoadIn( locals(), trainingModel, trainingInput )
    # if not validLoadData:
    #     raise Exception( 'Loaded previous data that did not agree with current parameters. Stopping now.' )
    
    if trainingInput.nBetas != nEpochs:
        if not hasattr( trainingInput, 'stimScale' ):
            trainingInput.stimScale = 10
        trainingInput = trainingInput.addStims(  nEpochs - trainingInput.nBetas  )
    
    
    
    
else: 

    trainingInput = inputStimuli( d, nTrainingBetas, simOptions )
    d = trainingInput.signalDim         ## May have been modified depending on if circleStims
    
    trainingModel = memoryModel( d, N, weightCombos[0], 
                                    simOptions,
                                    # trainMatsDirectly=simOptions['trainMatsDirectly'], 
                                    )
    trainingModel.epochNum = None
    
trainingModel.to( device )



#-----------------------------------------------------------------------------------------
''' Define the optimizer and set the learning rate'''
#-----------------------------------------------------------------------------------------
# paramDicts = [    { 'params' : trainingModel.D, 'lr' : learningRate }, 
#                   { 'params' : trainingModel.H, 'lr' : 0.25*learningRate }
#              ]
# optimizer = torch.optim.Adam( paramDicts )


# optimizer = torch.optim.Adam( trainingModel.parameters(), lr=learningRate )
optimizer = torch.optim.Adam( trainingModel.parameters(), lr=simOptions['learningRate'] )




simOptions['optimizer'] = optimizer



print( 'D norm:', torch.linalg.matrix_norm(  trainingModel.D ) )
print( 'H norm:', torch.linalg.matrix_norm(  trainingModel.H ) )





#=========================================================================================
''' TRAIN '''
#=========================================================================================
print(  )
print(  '==================================================' )
print(  'TRAINING ' )
print(  '==================================================' )
print(  'nEpochs = {}'.format(nEpochs)  )
print(  'Cost weight combo: {}'.format(weightCombos[0])  )
print(  'Attention (e,r):  ({},{}) '.format( simOptions['encodingAlpha'], simOptions['retentionAlpha'])    )
#------------------------------------------
print(  )
print( 'Time steps: (training, testing) ' )
# print( '\t nEncodSteps: ({},{})'.format( simOptions['parameters']['nEncodSteps_training'], simOptions['parameters']['nEncodSteps_testing'])   )  
print( '\t nEncodSteps: ({},{})'.format( simOptions['nEncodSteps'], simOptions['nEncodSteps_testing'])   )  
print( '\t nRetenSteps: ({},{})'.format( simOptions['nRetenSteps'], simOptions['nRetenSteps_ext'])  )  
# print(  'nEncodSteps_train = {}'.format(nEncodSteps_training)  )
# print(  'nRetenSteps_train = {}'.format(nRetenSteps_train)  )
#------------------------------------------
print(  )
print( 'trainBeforeDecay: ', simOptions['trainBeforeDecay'] )  
print( 'trainAfterDecay: ', simOptions['trainAfterDecay'] )  
#------------------------------------------
print(  )
# print(  'learningRate = {}'.format(learningRate)  )
print(  'learnSingleStimUntilConvergence:', simOptions['learnSingleStimUntilConvergence'] ) 
if simOptions['learnSingleStimUntilConvergence']: 
    # print(  '\t maxIter = {}'.format(simOptions['parameters']['maxIter'])  )
    print(  '\t maxIter = {}'.format(simOptions['maxIter'])  )
#------------------------------------------
print(  )
print(  )



start = time.time()
[startH, startM] = printActualTime( )
print( )


[ trainingModel, trainingData ] = trainModel( trainingModel, trainingInput, simOptions,
                                             # includeDelay=simOptions['includeDelayInTraining'],  
                                             )


print( )
printTimeElapsed(  start,  time.time()  )

[endH, endM] = printActualTime( )
print( )



#% %
#=========================================================================================
''' Record data '''
#=========================================================================================

nEncodSteps_training = nEncodSteps = simOptions['nEncodSteps']



if simOptions['includeDelayInTraining']:
    # nEvolSteps_training = nEncodSteps_training + nRetenSteps_train
    # trainingData[ 'nRetenSteps' ] = nRetenSteps_train
    nEvolSteps_training = simOptions['nEncodSteps'] + simOptions['nRetenSteps']
    trainingData[ 'nRetenSteps' ] = simOptions[ 'nRetenSteps' ]
else:
    # nEvolSteps_training = nEncodSteps_training
    nEvolSteps_training = simOptions['nEncodSteps']
    
nTrainingTimes = (nEvolSteps_training * nEpochs) + 1     ## number of time steps 


    

stimTimeInds = [  (nEncodSteps_training*i)+1  for i in range(nTrainingBetas)  ]
convergedTimeIndsReten  = [  ti-1 for ti in stimTimeInds[1::]  ]  +  [ nTrainingTimes-1 ]

convergedTimeInds = [  ti-1 for ti in stimTimeInds[1::]  ]  +  [ nTrainingTimes-1 ]

trainingData[ 'stimTimeInds' ] = stimTimeInds
trainingData[ 'convergedTimeInds' ] = convergedTimeInds
# trainingData[ 'nEncodSteps' ] = nEncodSteps_training
trainingData[ 'nEncodSteps' ] = simOptions['nEncodSteps']


trainingData[ 'convergedTimeIndsReten' ] = convergedTimeIndsReten



#% %
#=========================================================================================
''' Send email to alert me when complete '''
#=========================================================================================
if nEpochs > 2000:
    sendEmailWithStatus( subject='Finished training' )



# #%%


# #=========================================================================================
# ''' Plot Connection Matrix norms '''
# #=========================================================================================
# normFig = plotConnMatNorms( trainingData, epochNumsToTest, colorDict ) 


# normFig.suptitle( 'RP Sensitivity to matrix norms' ) 




# #=========================================================================================
# ''' Plot encoding performance  '''
# #=========================================================================================
# fig = plotModelEncodingAccur( trainingModel, trainingInput, trainingData )




#=========================================================================================
#=========================================================================================
#%% Create testing models 
#=========================================================================================
#=========================================================================================


#-----------------------------------------------------------------------------------------
''' Create the testing models '''
#-----------------------------------------------------------------------------------------
epochModels = { }

WC = weightCombos[0]
WC = simOptions[ 'weightCombos' ][ 0 ]



for epochNum in epochNumsToTest:
    
    epochModel = memoryModel( d, N, WC, simOptions )

    
    #-----------------------------------------------------
    if simOptions['storeAllConnMats']:
        storageInd = epochNum
    else:
        storageInd = epochNumsToTest.index( epochNum )
    #-----------------------------------------------------
    
    #-----------------------------------------------------
    if simOptions['trainMatsDirectly']:
        D = trainingData[ 'D' ][ storageInd ] 
        H = trainingData[ 'H' ][ storageInd ] 
    
        epochModel.setInitConnections( D=D, H=H ) 
        
    else: 
        W = trainingData[ 'W' ][ storageInd ] 
        Md = trainingData[ 'Md' ][ storageInd ] 
        Ms = trainingData[ 'Ms' ][ storageInd ] 
    
        epochModel.setInitConnections( W=W, Md=Md, Ms=Ms ) 
    #-----------------------------------------------------
    
    
    epochModel.epochNum = epochNum
    epochModels[ epochNum ] = epochModel 


    # epochModel.to( device )
    
    
    

#-----------------------------------------------------------------------------------------
''' Compute the stepSize for all the models and assign the smallest to each model '''
#-----------------------------------------------------------------------------------------  

lipschConsts = [  epochModels[i].LipschitzConstant() for i in epochNumsToTest  ]
stepSizeList = [ (1/L) for L in lipschConsts ]

stepSize = min( stepSizeList )



for epochNum in epochNumsToTest:
    
    epochModel = epochModels[ epochNum ]
    epochModel.stepSize = stepSize
    
    epochModels[ epochNum ] = epochModel








#=========================================================================================
''' Structure of trained matrices  '''
#=========================================================================================
# [ Hfig, Dfig ] = compareLearnedStructures( epochModels )

# Hqfig = compareHq( epochModels, maxQ=6 )





#=========================================================================================
#=========================================================================================
#%% Testing 
#=========================================================================================
#=========================================================================================





# nTestBetas = 50
# # nRetenSteps = 100000
# # nRetenSteps = int( 1e5 )                                ## 1e5 = 100,000
# # nRetenSteps = int( 1e4 )                                ## 1e4 = 10,000
# nRetenSteps = int( 2e3 )                                ## 1e3 = 1,000
# # nRetenSteps = 100
# simOptions[ 'nRetenSteps' ] = nRetenSteps









#-----------------------------------------------------------------------------------------
''' Create the stimuli '''
#-----------------------------------------------------------------------------------------
if simOptions[ 'loadPrevData' ]:

    nNewStims = nTestBetas - testingInput.nBetas
    
    
    #-----------------------------------------------
    if nNewStims > 0:    
        newStims = torch.normal( 0, 1, size=(d, nNewStims), dtype=torch.float32 )
        testingInput.stimMat = torch.cat(  ( testingInput.stimMat, newStims ),  dim=1  )
    
    elif nNewStims < 0:
        if nNewStims >= testingInput.nBetas:
            raise Exception( '[inputStimuli.addStims] Number of stims to remove is >= nBetas! ' ) 
        else: 
            testingInput.stimMat = testingInput.stimMat[ :, 0:nNewStims ]
    
    testingInput.nBetas = testingInput.nBetas + nNewStims
    #-----------------------------------------------



    
    # if testingInput.nBetas != nTestBetas:
    #     testingInput = testingInput.addStims(  nNewStims  )
        

else: 

    testingInput = inputStimuli( d, nTestBetas, simOptions )
    




#-----------------------------------------------------------------------------------------
''' Initialize Storage '''
#-----------------------------------------------------------------------------------------
if not testingWasInterrupted:
    
    testingData = { }
    # testingData[ 'nEncodSteps_testing' ] = nEncodSteps_testing
    # testingData[ 'nEncodSteps_training' ] = nEncodSteps_training
    testingData[ 'nEncodSteps' ] = simOptions[ 'nEncodSteps_testing' ]
    
    epochNumList = epochNumsToTest
    








#% %




# nTestBetas = 10
# nRetenSteps = 2000
# simOptions[ 'nRetenSteps' ] = nRetenSteps



#=========================================================================================
''' TESTING '''
#=========================================================================================
## https://www.youtube.com/watch?v=Z_ikDlimN6A&t=22898s


print(  )
print(  '=========================' )
print(  'TESTING ' )
print(  '=========================' )
# print(  'Cost weight combo: {}'.format(weightCombos[0])  )
print(  'Cost weight combo: {}'.format( simOptions[ 'weightCombos' ][0] )  )
print(  'Attention (e,r):  ({},{}) '.format( simOptions['encodingAlpha'], simOptions['retentionAlpha'])    )
print(  'epochNumsToTest = {}'.format(epochNumsToTest)  )
print(  )
print(  'nTestBetas = {}'.format( testingInput.nBetas )  )
print(  )
# print(  'nEncodSteps = {}'.format( simOptions[ 'nEncodSteps_testing' ] )  )
print(  'nEncodSteps = {}'.format( simOptions[ 'nEncodSteps_testing' ] )  )
print(  'nRetenSteps = {}'.format( simOptions[ 'nRetenSteps_ext' ] )  )
print(  )
print(  )







toPrintInd = 0


#------------------------------------------
start = time.time()
printActualTime( )
#------------------------------------------


for epochNum in epochNumList:         # extra idx to account for init

    #---------------------------------------------
    ''' The current model '''
    #---------------------------------------------
    epochModel = epochModels[ epochNum ]
    # epochModel.to( device )
    epochModel.eval()                           # Set to evaluate 
# # 
#     print( )
    print( 'Testing epochNum: ', epochNum  )


    #---------------------------------------------
    ''' Test '''
    #---------------------------------------------
    epochModel, testingData_epoch = testModel( epochModel, testingInput, simOptions )
        
    testingData[ epochNum ] = testingData_epoch
    
    
    #---------------------------------------------
    ''' Save progress in case of memory crash'''
    #---------------------------------------------
    if nTestBetas > 250: 
        saveFolder = nameSaveDir( simOptions )
        saveDir = os.path.join( currDir, saveFolder ) 
        
        testingDataInProgress = testingData.copy() 
        filenames = saveModelInfo( locals(), saveDir, varNames=['testingDataInProgress'] )




#------------------------------------------
end = time.time()
print( )
printTimeElapsed( start, end )
printActualTime()
#------------------------------------------



#% %
#=========================================================================================
''' Record data '''
#=========================================================================================
nEncodSteps_testing = simOptions[ 'nEncodSteps_testing' ]
nRetenSteps = nRetenSteps_ext = simOptions[ 'nRetenSteps_ext' ]


nTotalEvolSteps = nEncodSteps_testing + nRetenSteps
nTestingTimes = (nTotalEvolSteps * nTestBetas)  +  1      ## number of time steps 

stimTimeInds = [  (nTotalEvolSteps*i)+1  for i in range(nTestBetas)  ]
convergedTimeIndsReten  = [  ti-1 for ti in stimTimeInds[1::]  ]  +  [ nTestingTimes-1 ]

convergedTimeInds = [  ti-nRetenSteps-1 for ti in stimTimeInds[1::]  ]  +  [ nTestingTimes-nRetenSteps-1 ]


testingData[ 'stimTimeInds' ] = stimTimeInds
testingData[ 'convergedTimeInds' ] = convergedTimeInds
testingData[ 'nEncodSteps' ] = simOptions[ 'nEncodSteps_testing' ]
testingData[ 'nRetenSteps' ] = simOptions[ 'nRetenSteps_ext' ]

testingData[ 'convergedTimeIndsReten' ] = convergedTimeIndsReten



# delayPeriodStartTimeInds = [ '' ]
# testingData[ 'delayPeriodStartInds' ] = delayPeriodStartTimeInds



#% %




# #=========================================================================================
# ''' Plot encoding performance '''
# #=========================================================================================
# for epochNum in [  epochNumsToTest[0], epochNumsToTest[-1]  ]:    
    
#     epochModel = epochModels[ epochNum ]
#     fig = plotModelEncodingAccur( epochModel, testingInput, testingData )

    



#% %
#=========================================================================================
''' Send email to alert me when complete '''
#=========================================================================================
sendEmailWithStatus( subject='Finished testing {} stimuli'.format(nTestBetas) )






#=========================================================================================
#=========================================================================================
#%% ANALYSIS
#=========================================================================================
#=========================================================================================

FH = 5
# FH = 8
# FH = 12
# FH = 15
# FH = 50
# FH = None 



RP_compareToRefState = True                 ## Compare:   D * Hq * r(t)   to   r(t-q)
# RP_compareToRefState = False                ## Compare:   D * Hq * r(t)   to   x(t-q)




analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                 compareToRefState=RP_compareToRefState )
















# testingAnalysis = { 'recoveryPerf' : { },
#                     'reconR' : { },
#                     'reconX' : { },
                    
#                     'encodingError' : { },
#                     'convergenceRate' : { },
#                     # 'reconX' : { },
#                     # 'reconX' : { },
#                   }


    
# recovPerfDict = { }
# recovPerfRetenDict = { }
# reconRDict = { }
# reconXDict = { }



# convergenceDict = { }
# proxEvolErrorDict = { }


# costTermsDict = { }


# for epochNum in epochNumsToTest:
# # for epochNum in epochNumsToTest[0:2]:
    
#     print(  )
#     print( 'epochNum:', epochNum )
    
    
#     epochModel = epochModels[ epochNum ]
    
    
#     #-------------------------------------------------------------------------------------
#     ''' Recovery Performance '''
#     #-------------------------------------------------------------------------------------
#     if simOptions[ 'parameters' ][ 'nRetenSteps' ] > 0:
#         [ recovPerf, recovPerfReten, reconR, reconX ] = recoveryPerf( epochModel, testingInput, testingData, 
#                                                              forwardHorizon=FH, retenData=True, 
#                                                              compareToRefState=RP_compareToRefState )
#         recovPerfRetenDict[ epochNum ] = recovPerfReten
        
        
#     else: 
#         [ recovPerf, reconR, reconX ] = recoveryPerf( epochModel, testingInput, testingData, 
#                                                              forwardHorizon=FH, 
#                                                              compareToRefState=RP_compareToRefState )
    
#     recovPerfDict[ epochNum ] = recovPerf
#     reconRDict[ epochNum ] = reconR
#     reconXDict[ epochNum ] = reconX
    
    
    
#     # #-------------------------------------------------------------------------------------
#     # ''' Convergence '''
#     # #-------------------------------------------------------------------------------------
#     # # convergTimes, proxEvol = convergenceRate( epochModel, testingInput, testingData )
    
#     # # convergenceDict[ epochNum ] = convergTimes
#     # # proximalEvol[ epochNum ] = proxEvol
    
#     # convergenceDict[ epochNum ] = {}
    
#     # [decodedConvergInds, stateConvergInds] = convergenceRate( epochModel, testingInput, testingData )
#     # convergenceDict[ epochNum ][ 'decoded' ] = decodedConvergInds
#     # convergenceDict[ epochNum ][ 'state' ] = stateConvergInds
    
    
#     proxEvolErrorDict[ epochNum ] = { }
#     encodingErrorNorms, encodingPercentErrors = compareProxStepError( epochModel, testingInput, testingData )
#     proxEvolErrorDict[ epochNum ][ 'absolute' ] = encodingErrorNorms 
#     proxEvolErrorDict[ epochNum ][ 'percent' ] = encodingPercentErrors 
    
    
#     # #-------------------------------------------------------------------------------------
#     # ''' Cost Terms '''
#     # #-------------------------------------------------------------------------------------
#     # epoch_costs = costTerms_model( epochModel, testingInput, testingData )
#     # costTermsDict[ epochNum ] = epoch_costs
    
    
    
#=========================================================================================
#=========================================================================================
#%% PLOTTING
#=========================================================================================
#=========================================================================================


plotOnlyRP = False
# plotOnlyRP = True


compareToRefState = True
# compareToRefState = False


nColors = len( epochNumsToTest )
minColorLimit = 0.2


figDict = { }





#=========================================================================================
''' Plot Connection Matrix norms '''
#=========================================================================================
# normFig = plotConnMatNorms( trainingData, epochNumsToTest, colorDict ) 

# normFig.suptitle( 'RP Sensitivity to matrix norms' ) 




# #=========================================================================================
# ''' Plot encoding performance  '''
# #=========================================================================================
# fig = plotModelEncodingAccur( trainingModel, trainingInput, trainingData )




#=========================================================================================
''' Plot recovery performance of past inputs '''
#=========================================================================================
cmapRP = mpl.colormaps[ 'Reds' ]
# cmapRP = mpl.colormaps[ 'gist_rainbow' ]
# colorDict = colorGradient( cmapRP, nEpochNumsToTest, minColorLimit )


rpFig = plotRP( analysisDict, nTestBetas, FH, colors=colorDict, 
                       compareToRefState=compareToRefState,
# rpFig = plotRP( analysisDict['recovPerfDict'], nTestBetas, FH, colors=colorDict, 
                                    # epochNumsToPlot=[0, 200, 600, 800, 1000, 5000, 10000],
                                    # epochNumsToPlot=[0, 200, 600, 800, 1000],
                                   )





wcStr = weightComboToReadable( weightCombos[0] )
alphaRefStr = 'rA'  +  str( simOptions['retentionAlpha'] )  +  '_'  +  'eA'  +  str( simOptions['encodingAlpha'] )
titleStr = alphaRefStr + '  -  ' + wcStr

# titleStr = r'$\gamma$={}, {}'.format(learningRate, wcStr)
rpFig.suptitle(  titleStr  )


figDict[ 'recoveryPerf' ] = rpFig



# #------------------------------------------------------

# if simOptions[ 'parameters' ][ 'nRetenSteps' ] > 0:
# #     rpRetenFig = plotRP( analysisDict['recovPerfRetenDict'], nTestBetas, FH, colors=colorDict, 
#     rpRetenFig = plotRP( analysisDict, nTestBetas, FH, colors=colorDict, 
#                                    )
    
#     rpRetenFig.suptitle(  r'Retention: $\gamma$={}, {}'.format(learningRate, wcStr)  )
#     figDict[ 'recoveryPerfReten' ] = rpRetenFig

# #------------------------------------------------------
    





#=========================================================================================
# ''' Plot cost terms (at converged times) '''
# #=========================================================================================
# costFig = plotLossTerms( costTermsDict, epochNumsToTest )

# figDict[ 'costTerms' ] = costFig





# #=========================================================================================
# ''' Proximal evolution encoding accuracy '''
# #=========================================================================================
# proxEncodingErrFig = plotProxEvolError( proxEvolErrorDict, colorDict )

# proxEncodingErrFig

# figDict[ 'proxEncodingErr' ] = proxEncodingErrFig








# #=========================================================================================
# ''' Plot training encoding performance  '''
# #=========================================================================================
# fig = plotModelEncodingAccur_training( trainingModel, trainingInput, trainingData, z=5 )

# figDict[ 'trainingEncoding' ] = fig





# #=========================================================================================
# ''' Plot testing encoding performance '''
# #=========================================================================================
# # for epochNum in [  epochNumsToTest[0], epochNumsToTest[-1]  ]:    
    
# for epochNum in [  epochNumsToTest[-1]  ]:    
    
#     epochModel = epochModels[ epochNum ]
#     fig = plotModelEncodingAccur( epochModel, testingInput, testingData, z=5 )

#     figDict[ 'encoding' + str(epochNum) ] = fig






#=========================================================================================
''' Plot reconstruction performance '''
#=========================================================================================
if not plotOnlyRP:
    
    for epochNum in [ 0, nEpochs ]:
    # for epochNum in [ nEpochs ]:
        model = epochModels[ epochNum ]
        reconFig = plotReconstruction(  model,  testingData,  testingInput  )








#%%
#=========================================================================================
''' Tuning '''
#=========================================================================================
if not plotOnlyRP:
    
    
    # cmap = mpl.colormaps['viridis']
    cmap = mpl.colormaps['seismic']
    
    
    for epochNum in [ 0, nEpochs ]:
        
        model = epochModels[ epochNum ]
    
        subnetTuningFig = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=0, cmap=cmap )
        subnetTuningFig1 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=1, cmap=cmap )
        subnetTuningFigQ2 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=2, cmap=cmap )
        # subnetTuningFigQ3 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=3, cmap=cmap )
    
    
    
    # for epochNum in [ 0, nEpochs ]:
        
    #     model = epochModels[ epochNum ]
    
    #     subnetTuningFig = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=1, cmap=cmap, converged=False )
    #     subnetTuningFigQ2 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=2, cmap=cmap, converged=False )
    #     # subnetTuningFigQ3 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=3, cmap=cmap )
    
    
#% %   
    
    
    
    
    
    if nRetenSteps > 0:
        
        
        for epochNum in [ 0, nEpochs ]:
            
            model = epochModels[ epochNum ]
        
            subnetTuningFig = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=0, cmap=cmap, retention=True )
            subnetTuningFig1 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=1, cmap=cmap, retention=True )
            subnetTuningFigQ2 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=2, cmap=cmap, retention=True )
            subnetTuningFigQ3 = subnetworkTuning( model, testingData, testingInput, trainingInput=trainingInput, q=3, cmap=cmap, retention=True )
    
        
    
    
    
    
    
    
    # fig = plt.figure(  )
    # plt.imshow(   epochModels[0].D.detach().numpy()   )
    
    
    
    
    
    
    
    
    # printSimOptions( simOptions )
    printSimOptions2( simOptions )
    




#=========================================================================================
''' Matrix structure '''
#=========================================================================================
if not plotOnlyRP:
    
    
    eigvalFig = plotLearningEigvals( epochModels )
    eigvalFig.suptitle( 'Learned Eigenvalues of H' )
    
    
    if len( epochNumsToTest ) == 7:
        epochModels_copy = epochModels.copy()
        del epochModels_copy[ 500 ]
        [ Hfig, Dfig ] = compareLearnedStructures( epochModels_copy, Hmin=-1.25, Hmax=1.25 )

    else:
        [ Hfig, Dfig ] = compareLearnedStructures( epochModels, Hmin=-1.25, Hmax=1.25 )




#=========================================================================================
''' Principal Component Analysis '''
#=========================================================================================
if not plotOnlyRP:
    
    
    k = 2
    k = 3
    
    
    #--------------------------------------------------------
    ''' State '''
    #--------------------------------------------------------
    
    
    for epochNum in epochNumsToTest:
        state = testingData[ epochNum ][ 'state' ]
        
        [ proj, fig ] = plotPCA( state.T, k=3 )
        
        titleStr = 'State (epochNum =' + str(epochNum) + ')'
        fig.suptitle( titleStr )
    



#--------------------------------------------------------
#%% convergedStates_encoding
#--------------------------------------------------------
if not plotOnlyRP:
    
    
    for epochNum in epochNumsToTest:
        state = testingData[ epochNum ][ 'convergedStates_encoding' ]
        
        # [ proj, fig ] = plotPCA( state.T, k=3 )
        # [ proj, fig ] = plotPCA( state.T, k=3, stimMat=testingInput.stimMat )
        [ proj, fig ] = plotPCA( state.T, k=3 )
        
        # fig.get_axes()[0].scatter(  x=testingInput.stimMat[0],  y=testingInput.stimMat[1], 
        #                           s=8, c='red', marker='*'  );
        
        
        titleStr = 'Encoding converged (epochNum =' + str(epochNum) + ')'
        fig.suptitle( titleStr )
        
    
    

    
    
#--------------------------------------------------------
#%% convergedStates_retention 
#--------------------------------------------------------
if not plotOnlyRP:
      
      
    
    for epochNum in epochNumsToTest:
        state = testingData[ epochNum ][ 'convergedStates_retention' ]
        
        # [ proj, fig ] = plotPCA( state.T, k=3 )
        [ proj, fig ] = plotPCA( state.T, k=3, stimMat=testingInput.stimMat )
        # fig.get_axes()[0].scatter(  x=testingInput.stimMat[0],  y=testingInput.stimMat[1], 
        #                           s=8, c='red', marker='*'  );
        
        
        titleStr = 'Retention converged (epochNum=' + str(epochNum) + ')'
        fig.suptitle( titleStr )
    
    
      
#%%
#---------------------------------------------------------
''' PCA over full state evol., then plot converged '''
#---------------------------------------------------------
for epochNum in epochNumsToTest:
    
    state = testingData[ epochNum ][ 'state' ]
    stateProj = PCA( state.T )
    
    convergedTimeInds = testingData[ 'convergedTimeInds' ]  
    nBetas = len( convergedTimeInds )
    convergedStateProj = stateProj[ convergedTimeInds, : ].detach().numpy()
    
    cmap = mpl.colormaps[ 'cool_r' ]
    colors = cmap(   [ x/nBetas for x in range(nBetas) ]   )
    
    [ fig, axs ] = plt.subplots(  1, 1,  subplot_kw={"projection": "3d"},  figsize=(10, 5)  ) 
    scatterFig = axs.scatter3D( convergedStateProj[:,0], convergedStateProj[:,1], convergedStateProj[:,2],
                                             s=10, c=colors )
    scatterFig.cmap = cmap

    axs.set_xlabel( 'Component 1' )
    axs.set_ylabel( 'Component 2' )
    axs.set_zlabel( 'Component 3' )
    
    
    cbar = plt.colorbar( scatterFig ) 
    # cbar.set_ticks(  )

    

    
    
    titleStr = 'Retention converged (epochNum=' + str(epochNum) + ')'
    fig.suptitle( titleStr )



    
    







#%% Separate encoding and retention periods

''' 
    Extract the retenion period for the first few sitmuli and and give to plotPCA(), which 
subsequently does a PCA analysis on those states and plots it. 
'''




#--------------------------------------------------------
''' Encoding period '''
#--------------------------------------------------------

plotOnlyRP = False


encoding = False
if nRetenSteps == 0:
    encoding = True

        
nBetasToPlot = 3
nBetasToPlot = nTestBetas



if 'stimTimeInds' not in locals().keys():
    stimTimeInds = testingData[ 'stimTimeInds' ]
    




# if not plotOnlyRP:
    
#     nEncodingsToPlot = 3
#     nEncodingsToPlot = nTestBetas

#     [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds[0:nEncodingsToPlot] )

#     #--------------------------------------------------------
    
    
#     for epochNum in epochNumsToTest:
#         # state = testingData[ epochNum ][ 'convergedStates_encoding' ]
#         state = testingData[ epochNum ][ 'state' ][ :, encodingTimeInds ]
    
        
#         [ proj, fig ] = plotPCA( state.T, k=3 )
#         # fig.get_axes()[0].scatter(  x=testingInput.stimMat[0],  y=testingInput.stimMat[1], 
#         #                           s=8, c='red', marker='*'  );
        
        
#         titleStr = 'Encoding periods (epochNum =' + str(epochNum) + ')'
#         fig.suptitle( titleStr )
        
        

if not plotOnlyRP:
    # 

    # [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds[0:nEncodingsToPlot] )

    #--------------------------------------------------------
        
    
    # for epochNum in [ 0, nEpochs ]:
    for epochNum in [ nEpochs ]:
        print( )
        print( epochNum )
        

        # #--------------------------------------------------------
        # ''' Figure for each retention phase '''
        # #--------------------------------------------------------
        # for k in range( nBetasToPlot ):
        #     print( k )
            
        #     fig = plt.figure( ) 
        
        #     [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, [stimTimeInds[k]] )
            
            
        #     if encoding:
        #         state = testingData[ epochNum ][ 'state' ][ :, encodingTimeInds ]
        #         titleStr = 'Encoding periods (epochNum =' + str(epochNum) + ')'
        #     else: 
        #         state = testingData[ epochNum ][ 'state' ][ :, retentionTimeInds ]
        #         titleStr = 'Retention periods (epochNum =' + str(epochNum) + ')'

                
            
            
        #     # [ proj, fig ] = plotPCA( state.T, k=3, fig=fig, cmap=mpl.colormaps['Blues'] )
        #     [ proj, fig ] = plotPCA( state.T, k=3, fig=fig )
        #     fig.suptitle( titleStr )
        #     # fig.get_axes()[0].view_init(elev=45, azim=90, roll=270)
            
            
            
            
        #--------------------------------------------------------
        '''  '''
        #--------------------------------------------------------
            
        fig = plt.figure( ) 
    
        # [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds[0:nBetasToPlot] )
        [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps_ext, stimTimeInds[0:nBetasToPlot] )
        
        print(  len(retentionTimeInds)  )
        
        if encoding:
            state = testingData[ epochNum ][ 'state' ][ :, encodingTimeInds ]
            titleStr = 'Encoding periods (epochNum =' + str(epochNum) + ')'
        else: 
            state = testingData[ epochNum ][ 'state' ][ :, retentionTimeInds ]
            titleStr = 'Retention periods (epochNum =' + str(epochNum) + ')'

            
        
        
        # [ proj, fig ] = plotPCA( state.T, k=3, fig=fig, cmap=mpl.colormaps['Blues'] )
        [ proj, fig ] = plotPCA( state.T, k=3, fig=fig )
        fig.suptitle( titleStr )
        # fig.get_axes()[0].view_init(elev=45, azim=90, roll=270)




    [ U, S, V ] = torch.pca_lowrank( state )
    
    
    fig = plt.figure()
    k = 3
    plt.plot( V[:, :k].detach().numpy() )
        
        
#--------------------------------------------------------
#--------------------------------------------------------
#%%  Phases together 
#--------------------------------------------------------
#--------------------------------------------------------
        


fig = plt.figure( ) 

# [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds )
[ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps_ext, stimTimeInds )


if encoding:
    state = testingData[ epochNum ][ 'state' ][ :, encodingTimeInds ]
    titleStr = 'Encoding periods (epochNum =' + str(epochNum) + ')'
else: 
    state = testingData[ epochNum ][ 'state' ][ :, retentionTimeInds ]
    titleStr = 'Retention periods (epochNum =' + str(epochNum) + ')'

    


[ proj, fig ] = plotPCA( state.T, k=3, fig=fig )
fig.suptitle( titleStr )
# fig.get_axes()[0].view_init(elev=45, azim=90, roll=270)

        
    







#%% Look at the phases for a set of betas


plotAll = True
plotAll = False


plotReten = True
# plotReten = False

plotEncod = True
plotEncod = False


if plotAll:
    plotEncod = True
    plotReten = True
    
    # nPts = 




nBetasToPlot = 10


nBins = 6
# nBins = 30
binnedBetaInds = binBetasByTheta( testingInput, nBins=nBins )



cmap = mpl.colormaps[ 'autumn' ]
cmap2 = mpl.colormaps[ 'winter' ]


colors = [ 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'black' ]
# cmap = mpl.colormaps[ 'gist_rainbow' ]
# colors = cmap(   [ x/nPts for x in range(nPts) ]   )





    
state = testingData[ epochNum ][ 'state' ]                          # ( N, nTimes )
stateProj = PCA( state.T ).detach().numpy()                         # ( nTimes, k )
    


fig = plt.figure( figsize=(6, 6) )
ax = fig.add_subplot( projection='3d' )
    

for binNum in range( nBins ):

    
    betaInds = binnedBetaInds[   list( binnedBetaInds.keys() )[binNum]   ][ 0 ] 
    
    if len(betaInds) == 0:
        continue

    stimTimeInds_curr = [  stimTimeInds[k] for k in betaInds  ]
    # [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds_curr )
    [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps_ext, stimTimeInds_curr )
    
    
    
    
    
    if plotReten:
        retentionProj = stateProj[ retentionTimeInds ]                      # (  (nRetenSteps * nTestBetas),  k  )
    
        # retentionProj = PCA(  state[ :,retentionTimeInds ].T  )      # (  (nRetenSteps * nTestBetas),  k  )
        # retentionProj = retentionProj.detach().numpy()
        
        # nPts = len( retentionTimeInds )
        # colors = cmap(   [ x/nPts for x in range(nPts) ]   )
        retenFig = ax.scatter( retentionProj[:,0], retentionProj[:,1], retentionProj[:,2], s=5, c=colors[binNum], marker='o' )
        # retenFig.cmap = cmap
    
        # plt.colorbar( retenFig, label='Retention t', pad=0.15 )
    
    
    if plotEncod:
        encodingProj = stateProj[ encodingTimeInds ]                      # (  (nRetenSteps * nTestBetas),  k  )
    
        # encodingProj = PCA(  state[ :,encodingTimeInds ].T  )      # (  (nRetenSteps * nTestBetas),  k  )
        # encodingProj = encodingProj.detach().numpy()
    
    
            
        # nPts = len( encodingTimeInds )
        # colors = cmap(   [ x/nPts for x in range(nPts) ]   )
        # colors = cmap2(   [ x/nPts for x in range(nPts) ]   )
        encodFig = ax.scatter( encodingProj[:,0], encodingProj[:,1], encodingProj[:,2], s=5, c=colors[binNum] )
        # encodFig.cmap = cmap2
    
        # plt.colorbar( encodFig, label='Encoding t', pad=0.15, location='left'  )
    
    
    # elif plotAll:
        
    #     retentionProj = stateProj[ retentionTimeInds ]                      # (  (nRetenSteps * nTestBetas),  k  )
    #     encodingProj = stateProj[ encodingTimeInds ]                      # (  (nRetenSteps * nTestBetas),  k  )

    #     retenFig = ax.scatter( retentionProj[:,0], retentionProj[:,1], retentionProj[:,2], s=5, c=colors[binNum] )
    #     encodFig = ax.scatter( encodingProj[:,0], encodingProj[:,1], encodingProj[:,2], s=5, c=colors[binNum] )



    
ax.set_xlabel( 'Component 1' )
ax.set_ylabel( 'Component 2' )
ax.set_zlabel( 'Component 3' )
plt.title( 'Phase PCA' )
    
    

#%%



#--------------------------------------------------------
''' Retention period '''
#--------------------------------------------------------
if not plotOnlyRP:
      
  
    
    for epochNum in epochNumsToTest:
        # state = testingData[ epochNum ][ 'convergedStates_retention' ]
        state = testingData[ epochNum ][ 'state' ][ :, retentionTimeInds ]
    
        [ proj, fig ] = plotPCA( state.T, k=3 )
        # fig.get_axes()[0].scatter(  x=testingInput.stimMat[0],  y=testingInput.stimMat[1], 
        #                           s=8, c='red', marker='*'  );
        
        
        titleStr = 'Retention periods (epochNum=' + str(epochNum) + ')'
        fig.suptitle( titleStr )









#%% Stim Lifecycle

#-----------------------------------------------------------------------------------------
''' Stim Life cycle '''
#-----------------------------------------------------------------------------------------
for epochNum in [ 0, nEpochs ]:
    
    model = epochModels[ epochNum ]    
    figList = plotPCAOfStimLifecycle( model, testingData, testingInput, k=3, nBetasToPlot=3  )



    # figList = plotPCAOfStimLifecycle( model, testingData, testingInput, k=3, nBetasToPlot=3, converged=True  )


    
    # for epochNum in epochNumsToTest:
    #     state = testingData[ epochNum ][ 'convergedStates_retention' ]
        
    #     # [ proj, fig ] = plotPCA( state.T, k=3 )
    #     [ proj, fig ] = plotPCA( state.T, k=3, stimMat=testingInput.stimMat )
    #     # fig.get_axes()[0].scatter(  x=testingInput.stimMat[0],  y=testingInput.stimMat[1], 
    #     #                           s=8, c='red', marker='*'  );
        
        
    #     titleStr = 'Retention converged (epochNum=' + str(epochNum) + ')'
    #     fig.suptitle( titleStr )






#=========================================================================================
#=========================================================================================
#%% SAVE 
#=========================================================================================
#=========================================================================================



#-----------------------------------------------------------------------------------------
''' Where to save '''
#-----------------------------------------------------------------------------------------

if simOptions[ 'loadPrevData' ]:
    saveFolder = referenceFolder
else:
    # saveFolder = nameSaveDir( simOptions, weightCombos )
    saveFolder = nameSaveDir( simOptions )


print( )
print( 'saveFolder: ', saveFolder )
print( )






#-----------------------------------------------------------------------------------------
''' Save '''
#-----------------------------------------------------------------------------------------
saveDir = os.path.join( currDir, saveFolder ) 

filenames = saveModelInfo( locals(), saveDir )



# figFilenames = saveFigureDict(  figDict,  saveDir,  imgTypes=['.svg', '.png']  )





#% %
#=========================================================================================
''' Send email to alert me when complete '''
#=========================================================================================
sendEmailWithStatus( subject='Saved successfully', body=saveDir )
print( )
print( )





#=========================================================================================
#=========================================================================================
#%% LOAD using simOptions
#=========================================================================================
#=========================================================================================

rerunTest = True
# rerunTest = False


#---------------------------------------------
''' Load '''
#---------------------------------------------
saveDir = nameSaveDir( simOptions )                     ## Where the data is saved 
print(  )
print( 'Loading...', saveDir )
modelInfoDict = getModelInfo( saveDir )                 ## Load the data 


trainingModel = modelInfoDict[ 'trainingModel' ]        ## Training
trainingInput = modelInfoDict[ 'trainingInput' ]
trainingData = modelInfoDict[ 'trainingData' ]

epochModels = modelInfoDict[ 'epochModels' ]            ## Testing
testingInput = modelInfoDict[ 'testingInput' ]
testingData = modelInfoDict[ 'testingData' ]
# stimTimeInds = testingData[ 'stimTimeInds' ]


simOptions = modelInfoDict[ 'simOptions' ]              ## simOptions
epochNumsToTest = simOptions[ 'epochNumsToTest' ]


del modelInfoDict                                       ## Clear up some memory 


# print(  'Loaded data from: \n\t{}'.format( saveDir )  )
print(  'Loaded data.'  )



#%% Re-do testing

#---------------------------------------------
''' Analyze and plot '''
#---------------------------------------------
if rerunTest: 
    nTestBetas = 250
    nTestBetas = 300
    nTestBetas = 100
    # nTestBetas = 500
    # nTestBetas = 1000
    # nTestBetas = 2000
    # nTestBetas = 5000
    
    
    # nRetenSteps = 1000
    # simOptions[ 'nRetenSteps' ] = nRetenSteps
    
    
    
    print( )
    print( 'saveDir: ', saveDir )
    
        
    ''' Interrupted Testing? '''
    #---------------------------------------------
    testingWasInterrupted = False
    
    try:
        testingDataInProgress = getModelInfo( saveDir, 'testingDataInProgress' )
    
    
        keys = list( testingDataInProgress.keys() )  
        testedEpochNums = [  x for x in keys if type(x)==int  ]
        lastEpochFullyTested = testedEpochNums[-1]
        # try: 
        #     lastEpochFullyTested = testedEpochNums[-2]
        # except: 
        #     lastEpochFullyTested = testedEpochNums[-1]
        
        
        
        if lastEpochFullyTested < nEpochs:
            testingWasInterrupted = True
            epochNumList = [ x for x in epochNumsToTest if x > lastEpochFullyTested ]
            testingData = testingDataInProgress
            runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
        
        else: 
            runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')



    except: 
        runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    
    
    

# #% %
# runcell('ANALYSIS', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
# # runcell('PLOTTING', '/Users/bethannajones/Desktop/PGM/PGM_main.py')


# #% %

# runcell('Stim Lifecycle', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
# runcell('convergedStates_retention', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    



#=========================================================================================
#%% ITERATE over a simOption parameter
#=========================================================================================


trainAndTest = True
# trainAndTest = False

    


# eAlph = simOptions[ 'encodingAlpha' ]
# simOptions[ 'encodingAlpha' ] = 0.8
# simOptions[ 'nRetenSteps' ] = 100

#----------------------------------------------------------------------------
nTestBetas = 50 
# nTestBetas = 75 
nTestBetas = 100 
# nTestBetas = 150 
# nTestBetas = 200 
# nTestBetas = 400 


referenceTestingFolder = 'currTestingInfo'
inputSaveName = 'testingInput_{}'.format(nTestBetas)
fname = os.path.join( referenceTestingFolder, inputSaveName + '.pickle' )


if not os.path.isfile( fname ):
    testingInput = inputStimuli( d, nTestBetas, simOptions )
    saveModelInfo( locals(), referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )
#----------------------------------------------------------------------------


varType = 'nEncodSteps'

varVals = [ 25, 50, 100, 200, 300 ]
# varVals = [ 5, 10, 15, 20 ]
# varVals = [ 300 ]




varType = 'maxIter'
# varValList = [ 10, 20, 50, 100, 200 ]
varValList = [ 20, 50, 100, 200 ]
# varValList = [ 5, 10, 15, 20 ]
# varValList = [ 300 ]


# varType = 'nRetenSteps_ext'
# simOptions[ 'nRetenSteps' ] = 50
# simOptions[ 'nEncodSteps' ] = simOptions[ 'nEncodSteps_testing' ] = 25
# simOptions[ 'retentionAlpha' ] = 0.02
# varValList = [ 25, 50, 100 ]
# # varValList = [ 5, 10, 15, 20 ]
# # varValList = [ 300 ]


# varType = 'nEncodSteps_testing'
# simOptions[ 'nRetenSteps' ] = simOptions[ 'nRetenSteps_ext' ] = 50
# simOptions[ 'nEncodSteps' ] = 25
# simOptions[ 'retentionAlpha' ] = 0.02
# varValList = [ 25, 50, 100, 200 ]
# varValList = [ 100 ]
# varValList = [ 50 ]
# varValList = [ 200 ]
# # varValList = [ 5, 10, 15, 20 ]
# # varValList = [ 300 ]




varType = 'retentionAlpha'

# varValList = [  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1  ]
varValList = [  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8  ] 
# varValList = [  0.9  ]


# # varValList = [  0.01,  0.03,  0.05,  0.07,  0.09  ]
varValList = [  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09  ]
# varValList = [  0.01  ]
# varValList = [  0.002,  0.004,  0.006,  0.008 ]

# varValList = [  0.01,  0.02,  0.03,  0.04,  0.05  ]
# varValList = [  0.06, 0.07, 0.08, 0.09  ]




# varVals.reverse()

for varVal in varValList:

    simOptions[ varType ] = varVal
    # if varType == 'nEncodSteps':
    #     simOptions[ 'nEncodSteps_testing' ] = varVal
    
    
    if trainAndTest:
        runcell('Training', '/Users/bethannajones/Desktop/PGM/PGM_main.py') 
        runcell('Create testing models', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    else:
        print( )
        runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
        simOptions[ varType ] = varVal

    
    
    testingInput = getModelInfo( referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )

    simOptions[ 'loadPrevData' ] = True
    runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    simOptions[ 'loadPrevData' ] = False

    
    

    #-------------------------------------------------------------------------------------
    ''' Save? '''
    #-------------------------------------------------------------------------------------
    saveInput = True
    
    
    
    nEpochs = simOptions['parameters']['nEpochs'] 
    nEpochsKeyBool = nEpochs in testingData.keys()
    # print( '\n', nEpochsKeyBool )
    
    if nEpochsKeyBool:
        runcell('SAVE, #1', '/Users/bethannajones/Desktop/PGM/PGM_main.py')   
        print(  '----------------------------------------------'  )
        continue
    
    else: 
        sendEmailWithStatus( subject='Did not save: e{}r{}'.format( simOptions['encodingAlpha'], simOptions['retentionAlpha'] ), body=saveDir )
        print(  '----------------------------------------------'  )







#%%

def iterateOverSimParam( simOptions, varValList, varType='retentionAlpha',
                                    # trainAndTest=False, autoSaveCheck=True, 
                                    testOnly=True, autoSave=True, 
                                    nTestBetas=50 ):




    #----------------------------------------------------------------------------
    referenceTestingFolder = 'currTestingInfo'
    inputSaveName = 'testingInput_{}'.format(nTestBetas)
    fname = os.path.join( referenceTestingFolder, inputSaveName + '.pickle' )
    
    
    if not os.path.isfile( fname ):
        testingInput = inputStimuli( d, nTestBetas, simOptions )
        saveModelInfo( locals(), referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )
    #----------------------------------------------------------------------------



    
    for varVal in varValList:
    
        
        simOptions[ varType ] = varVal
        
        
        #---------------------------------------------------------------------------------
        ''' Train or load training data '''
        #---------------------------------------------------------------------------------  
        if testOnly:
            print( )
            runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
            simOptions[ varType ] = varVal 
        else:
            runcell('Training', '/Users/bethannajones/Desktop/PGM/PGM_main.py') 
            runcell('Create testing models', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
        
        
        #---------------------------------------------------------------------------------
        ''' Load testingInput and run testing '''
        #---------------------------------------------------------------------------------  
        testingInput = getModelInfo( referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )
    
        simOptions[ 'loadPrevData' ] = True
        runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
        simOptions[ 'loadPrevData' ] = False
    
        
        
    
        #---------------------------------------------------------------------------------
        ''' Save? '''
        #---------------------------------------------------------------------------------        
        if autoSave:
        
            nEpochs = simOptions['parameters']['nEpochs'] 
            nEpochsKeyBool = nEpochs in testingData.keys()

            
            if nEpochsKeyBool:
                
                ## Play a sound when saving
                sys.stdout.write('\a')
                sys.stdout.flush()
                
                
                saveDir = saveBasedOnSimOptions( simOptions )
                # runcell('SAVE, #1', '/Users/bethannajones/Desktop/PGM/PGM_main.py')   
                # print(  '----------------------------------------------'  )
                # continue
            
            else: 
                sendEmailWithStatus( subject='Did not save: e{}r{}'.format( simOptions['encodingAlpha'], simOptions['retentionAlpha'] ), body=saveDir )
            
            print(  '----------------------------------------------'  )
    
    
    
    
    
        else: 
            
            ## Play a sound when input is needed
            sys.stdout.write('\a')
            sys.stdout.flush()
            
            saveInput = input( '\n---->  Save? ' )
            
            
            if (saveInput == 'y') or (saveInput == 'Y') or (saveInput == 'yes') or (saveInput == '1'):
                runcell('SAVE, #1', '/Users/bethannajones/Desktop/PGM/PGM_main.py')   
            
            elif (saveInput == 'n') or (saveInput == 'N') or (saveInput == 'no') or (saveInput == '0'):
                sendEmailWithStatus( subject='Did not save: e{}r{}'.format(eAlph, rAlph), body=saveDir )
            
            else:
                raise Exception( 'Did not understand response.' )
    
            
            print(  '----------------------------------------------'  )
            
            
    
    
      


def saveBasedOnSimOptions( simOptions, localVars, currDir=None ):
    
    
    ''' Where to save '''
    #-------------------------------------------------------------------------------------
    if simOptions[ 'loadPrevData' ]:
        saveFolder = referenceFolder
    else:
        saveFolder = nameSaveDir( simOptions )
    
    
    print( )
    print( 'saveFolder: ', saveFolder )
    print( )
    
    

    

    ''' Save '''
    #-------------------------------------------------------------------------------------
    if currDir is None:
        currDir = os.getcwd()

    saveDir = os.path.join( currDir, saveFolder ) 
    filenames = saveModelInfo( localVars, saveDir )
    
    
    
    
    
    ''' Send email to alert me when complete '''
    #-------------------------------------------------------------------------------------
    sendEmailWithStatus( subject='Saved successfully', body=saveDir )
    print( )
    print( )


    return saveDir





#%%

''' 
JUST  rerunning TESTING for a different nRetenSteps/nTestBetas
'''


saveInput = False


# simOptions[ 'nRetenSteps_ext' ] = nRetenSteps_ext = 100
# simOptions['parameters']['nTotalEvolSteps'] =  simOptions[ 'nEncodSteps' ] + simOptions[ 'nRetenSteps_ext' ]


# testingWCs = makeWeightComboList( errWeightList, effWeightList, sWeightList, [0], rWeightList, fWeightList )



#----------------------------------------------------------------------------
nTestBetas = 50 
nTestBetas = 100 
# nTestBetas = 200 


referenceTestingFolder = 'currTestingInfo'
inputSaveName = 'testingInput_{}'.format(nTestBetas)
fname = os.path.join( referenceTestingFolder, inputSaveName + '.pickle' )


if not os.path.isfile( fname ):
    testingInput = inputStimuli( d, nTestBetas, simOptions )
    saveModelInfo( locals(), referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )
#----------------------------------------------------------------------------





rententionAlphaVals = [  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7  ]
# rententionAlphaVals = [  0.2,  0.3,  0.4,  0.5,  0.6  ]



rententionAlphaVals = [  0.2  ]
# rententionAlphaVals = [  0.7  ]
# rententionAlphaVals = [  0.04  ]
# rententionAlphaVals = [  0.001  ]
# 
# rententionAlphaVals = [ 0.4,  0.5,  0.6,  0.7  ]
# rententionAlphaVals = [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6 ]
# rententionAlphaVals = [ 0.1,  0.2,  0.3  ]

rententionAlphaVals = [  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09  ]
rententionAlphaVals = [  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08 ]

# rententionAlphaVals = [  0.02,  0.04,  0.06,  0.08  ] 


rententionAlphaVals.reverse()

for rAlph in rententionAlphaVals:

    
    #-------------------------------------------------------------------------------------
    ''' Load training data '''
    #-------------------------------------------------------------------------------------
    simOptions[ 'retentionAlpha' ] = rAlph
    # simOptions[ 'nRetenSteps' ] = nRetenSteps = 2000

    print( )
    simOptions[ 'weightCombos' ] = weightCombos
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    # simOptions[ 'weightCombos' ] = testingWCs

    
    
    #-------------------------------------------------------------------------------------
    ''' Run '''
    #-------------------------------------------------------------------------------------
    # simOptions[ 'nRetenSteps' ] = nRetenSteps = 1000
    
    simOptions[ 'loadPrevData' ] = True
    testingInput = getModelInfo( referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )
    
    
    
    simOptions[ 'nRetenSteps_ext' ] = nRetenSteps = 100

    
    
    runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    simOptions[ 'loadPrevData' ] = False

    
    
    #-------------------------------------------------------------------------------------
    ''' Save? '''
    #-------------------------------------------------------------------------------------
    saveInput = True
    
    nEpochs = simOptions['parameters']['nEpochs'] 
    nEpochsKeyBool = nEpochs in testingData.keys()
    # print( '\n', nEpochsKeyBool )
    if nEpochsKeyBool:
        runcell('SAVE, #1', '/Users/bethannajones/Desktop/PGM/PGM_main.py')   
        print(  '----------------------------------------------'  )
        continue
        
    
    # print( )
    # print( 'testingData.keys(): ', testingData.keys() )
    # print( 'nEpochs in testingData.keys(): ',  nEpochsKeyBool )
    
    saveInput = input( '\n---->  Save? ' )
    
    
    
    if (saveInput == 'y') or (saveInput == 'Y') or (saveInput == 'yes') or (saveInput == '1'):
        runcell('SAVE, #1', '/Users/bethannajones/Desktop/PGM/PGM_main.py')   
    
    elif (saveInput == 'n') or (saveInput == 'N') or (saveInput == 'no') or (saveInput == '0'):
        sendEmailWithStatus( subject='Did not save: e{}r{}'.format(eAlph, rAlph), body=saveDir )
    
    else:
        raise Exception( 'Did not understand response.' )

    
    print(  '----------------------------------------------'  )








# #=========================================================================================
# #%% ITERATE -- init mat norms
# #=========================================================================================


# #----------------------------------------------------------------------------
# nTestBetas = 50 

# referenceTestingFolder = 'currTestingInfo'
# inputSaveName = 'testingInput_{}'.format(nTestBetas)
# fname = os.path.join( referenceTestingFolder, inputSaveName + '.pickle' )


# if not os.path.isfile( fname ):
#     testingInput = inputStimuli( d, nTestBetas, simOptions )
#     saveModelInfo( locals(), referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )
# #----------------------------------------------------------------------------



# varValList = varVals

# varVals.reverse()

# for varVal in varVals:

#     simOptions[ varType ] = varVal
#     # if varType == 'nEncodSteps':
#     #     simOptions[ 'nEncodSteps_testing' ] = varVal
    
    
    
#     runcell('Training', '/Users/bethannajones/Desktop/PGM/PGM_main.py') 
#     runcell('Create testing models', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    
    
#     testingInput = getModelInfo( referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )

#     simOptions[ 'loadPrevData' ] = True
#     runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
#     simOptions[ 'loadPrevData' ] = False

    
    

#     #-------------------------------------------------------------------------------------
#     ''' Save? '''
#     #-------------------------------------------------------------------------------------
#     runcell('Single sim', '/Users/bethannajones/Desktop/PGM/retentionAnalysis.py')
    
  



#%% 









    
#=========================================================================================
#%% RESUME TESTING (memory crash)
#=========================================================================================


# currMaxIter = 50
currMaxIter = 25



#-----------------------------------------------------------------------------------------
''' Load up simOptions with the parameters '''
#-----------------------------------------------------------------------------------------
runcell('IMPORT', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
runcell('chg CWD', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

runcell('Settings', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
runcell('WEIGHTS', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
runcell('PARAMETERS', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
runcell("''' Print the settings/parameters   '''", '/Users/bethannajones/Desktop/PGM/PGM_main.py')

# simOptions['parameters']['maxIter'] = currMaxIter
simOptions['maxIter'] = currMaxIter


#-----------------------------------------------------------------------------------------
''' Load the respective data based on simOptions and continue testing '''
#-----------------------------------------------------------------------------------------
runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
#%%
runcell('Re-do testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')



#-----------------------------------------------------------------------------------------
''' Save -- If no crash occured, testing is finished '''
#-----------------------------------------------------------------------------------------
# runcell('SAVE', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
runcell(18, '/Users/bethannajones/Desktop/PGM/PGM_main.py')






#%% Interrupted testing 


# saveDir = nameSaveDir( simOptions )
# testingDataInProgress = getModelInfo( saveDir, 'testingDataInProgress' )

# keys = list( trainingDataInProgress.keys() )  
# try: 
#     lastEpochFullyTested = [  x for x in keys if type(x)==int  ][-2]
# except: 
#     lastEpochFullyTested = [  x for x in keys if type(x)==int  ][-1]


# epochNumsToTest_redo = [ x for x in epochNumsToTest if x > lastEpochFullyTested ]






# testingData = testingDataInProgress




# #------------------------------------------
# start = time.time()
# printActualTime( )
# #------------------------------------------

# for epochNum in epochNumsToTest_redo:         # extra idx to account for init

#     #---------------------------------------------
#     ''' The current model '''
#     #---------------------------------------------
#     epochModel = epochModels[ epochNum ]
#     # epochModel.to( device )
#     epochModel.eval()                           # Set to evaluate 

#     print( )
#     print( 'Testing epochNum: ', epochNum  )


#     #---------------------------------------------
#     ''' Test '''
#     #---------------------------------------------
#     epochModel, testingData_epoch = testModel( epochModel, testingInput, simOptions )
        
#     testingData[ epochNum ] = testingData_epoch
    
    
#     #---------------------------------------------
#     ''' Save progress in case of memory crash'''
#     #---------------------------------------------
#     if nTestBetas > 250: 
#         saveFolder = nameSaveDir( simOptions )
#         saveDir = os.path.join( currDir, saveFolder ) 
        
#         testingDataInProgress = testingData.copy()
#         filenames = saveModelInfo( locals(), saveDir, varNames=['testingDataInProgress'] )




# #------------------------------------------
# end = time.time()
# print( )
# printTimeElapsed( start, end )
# printActualTime()
# #------------------------------------------



# #% %
# #=========================================================================================
# ''' Record data '''
# #=========================================================================================
# nTotalEvolSteps = nEncodSteps_testing + nRetenSteps
# nTestingTimes = (nTotalEvolSteps * nTestBetas)  +  1      ## number of time steps 

# stimTimeInds = [  (nTotalEvolSteps*i)+1  for i in range(nTestBetas)  ]
# convergedTimeIndsReten  = [  ti-1 for ti in stimTimeInds[1::]  ]  +  [ nTestingTimes-1 ]

# convergedTimeInds = [  ti-nRetenSteps-1 for ti in stimTimeInds[1::]  ]  +  [ nTestingTimes-nRetenSteps-1 ]


# testingData[ 'stimTimeInds' ] = stimTimeInds
# testingData[ 'convergedTimeInds' ] = convergedTimeInds
# testingData[ 'nEncodSteps' ] = nEncodSteps_testing
# testingData[ 'nRetenSteps' ] = nRetenSteps

# testingData[ 'convergedTimeIndsReten' ] = convergedTimeIndsReten



# # delayPeriodStartTimeInds = [ '' ]
# # testingData[ 'delayPeriodStartInds' ] = delayPeriodStartTimeInds



# #% %




# # #=========================================================================================
# # ''' Plot encoding performance '''
# # #=========================================================================================
# # for epochNum in [  epochNumsToTest[0], epochNumsToTest[-1]  ]:    
    
# #     epochModel = epochModels[ epochNum ]
# #     fig = plotModelEncodingAccur( epochModel, testingInput, testingData )

    



# #% %
# #=========================================================================================
# ''' Send email to alert me when complete '''
# #=========================================================================================
# sendEmailWithStatus( subject='Finished testing {} stimuli'.format(nTestBetas) )











#%% network tuning


for epochNum in [ 0, nEpochs ]:

    model = epochModels[ epochNum ]
    
    
    
    state = testingData[ epochNum ][ 'state' ]
    convergedStates = testingData[ epochNum ][ 'convergedStates_encoding' ]
    
    
    maxStateVal = torch.max(  torch.max( state )  )
    
    N = model.networkDim
    xVals = list( range(360) )
    
    
    fig, axs = plt.subplots( 3, 1, height_ratios=[3,1,1] )
    # for i in range( N ): 
    #     currNeuron = state[ i, : ]
    
    
    xVals = np.zeros( [nTestBetas, 1] )
    yVals = np.zeros( [nTestBetas, 1] )
    
    
    for k in range( nTestBetas ): 
        
        currRad = testingInput.thetas[0][ k ]
        currTheta = currRad * (180/math.pi)
    
    
        converged = convergedStates[ :, k+1 ]           ##  +1 to account for initState    
        convergedNorm = torch.linalg.norm( converged )
        
        
        xVals[k] = currTheta.detach().numpy()
        yVals[k] = convergedNorm.detach().numpy()
        
        
        
        
        # currNeuron = state[ i, : ]
        
        
        
        
    axs[0].scatter( xVals, yVals, s=8 )
    # axs[0].set_xlabel( r'$\theta$' )
    axs[0].set_ylabel(  r'$\| \mathbf{r}(t)  \|$'  )
    axs[0].get_xaxis().set_ticks([])
    
    
    
    trainingRads = trainingInput.thetas
    trainingThetas = trainingRads * (180/math.pi)
    axs[1].hist( trainingThetas, bins=90 )
    # axs[1].set_xlabel( r'$\theta$' )
    axs[1].set_ylabel(  'Training \n frequency'  )
    axs[1].get_xaxis().set_ticks([])
    
    
    
    trainingRads = testingInput.thetas
    trainingThetas = trainingRads * (180/math.pi)
    axs[2].hist( trainingThetas, bins=90 )
    axs[2].set_xlabel( r'$\theta$' )
    axs[2].set_ylabel(  'Testing \n frequency'  )
    axs[2].get_xaxis().set_ticks( list(range(0,361,60)) )
    
    
        
        
    plt.suptitle( 'Network-wide Tuning: epoch {}'.format(epochNum) )


    



#%% Subnetwork tuning





cmap = mpl.colormaps[ 'viridis' ]
norm = mpl.colors.Normalize(vmin=0, vmax=1)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


for epochNum in [ 0, nEpochs ]:

    
    model = epochModels[ epochNum ]
    
    
    
    state = testingData[ epochNum ][ 'state' ]
    convergedStates = testingData[ epochNum ][ 'convergedStates_encoding' ]
    colorInds = convergedStates.detach().numpy() 
    
    
    convergedMax = torch.max( torch.max( convergedStates ) ).detach().numpy() 
    convergedMax = torch.max( torch.max( abs(convergedStates) ) ).detach().numpy() 
    
    
    
    maxStateVal = torch.max(  torch.max( state )  )
    
    N = model.networkDim
    xVals = list( range(360) )
    
    
    # fig, axs = plt.subplots( 2, 1, height_ratios=[2,1] )
    fig, axs = plt.subplots( 3, 1, height_ratios=[4,1,1] )
    # for i in range( N ): 
    #     currNeuron = state[ i, : ]
    
    
    xVals = np.zeros( [nTestBetas, N] )
    yVals = np.zeros( [nTestBetas, N] )
    colorVals = np.zeros( [nTestBetas, 1] )
    
    
    for k in range( nTestBetas ): 
        
        currRad = testingInput.thetas[0][ k ]
        currTheta = currRad * (180/math.pi)
    
    
        converged = convergedStates[ :, k+1 ]           ##  +1 to account for initState    
        convergedNorm = torch.linalg.norm( converged )
        
        
        # xVals[k] = currTheta.detach().numpy()
        # yVals[k] = convergedNorm.detach().numpy()
        
        
        colorInd = converged.detach().numpy() 
        colorInd = colorInd / np.max( abs(colorInd) )                       ## Normalize 
        # colorInd = colorInd / convergedMax                       ## Normalize 
        color = cmap( colorInd )
        
        xVals = [ currTheta.detach().numpy() ] * N
        yVals = list( range(1,N+1) )
        # xVals[k] = [ currTheta.detach().numpy() ] * N
        # yVals[k] = list( range(1,N+1) )
        
        
        axs[0].scatter( xVals, yVals, s=5, c=color )
    
    
    
        
        
    # fig.subplots_adjust( right=0.8 )
    # # cbar_ax = fig.add_axes(  [0.85, 0.15, 0.05, 0.7]  )
    # cbar_ax = fig.add_axes(  [0.85, 0.1, 0.05, 0.8]  )
    # plt.colorbar( mappable, cax=cbar_ax, label='' )
        
    
    
        
    # # axs[0].scatter( xVals, yVals, s=8, c=colorVals )
    axs[0].set_ylabel( 'Subnetwork ind' )
    # axs[0].set_xlabel( r'$\theta$' )
    axs[0].get_xaxis().set_ticks([])
    
    # axs[0].set_ylabel(  r'$\| \mathbf{r}(t)  \|$'  )
    # plt.colorbar( cmap )
    
    
    trainingRads = trainingInput.thetas
    trainingThetas = trainingRads * (180/math.pi)
    axs[1].hist( trainingThetas, bins=90 )
    # axs[1].set_xlabel( r'$\theta$' )
    axs[1].set_ylabel(  'Training \n frequency'  )
    axs[1].get_xaxis().set_ticks([])
    
    
    trainingRads = testingInput.thetas
    trainingThetas = trainingRads * (180/math.pi)
    axs[2].hist( trainingThetas, bins=90 )
    axs[2].set_xlabel( r'$\theta$' )
    axs[2].set_ylabel(  'Testing \n frequency'  )
    axs[2].get_xaxis().set_ticks( list(range(0,361,60)) )
    
    
        
        
    plt.suptitle( 'Subnetwork Tuning: epoch {}'.format(epochNum) )
    
    
    



#%% 


cmap = mpl.colormaps[ 'viridis' ]
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


for epochNum in [ 0, nEpochs ]:

    model = epochModels[ epochNum ]
    
    
    
    state = testingData[ epochNum ][ 'state' ]
    convergedStates = testingData[ epochNum ][ 'convergedStates_encoding' ]
    colorInds = convergedStates.detach().numpy() 
    
    
    converged_normalized = convergedStates / torch.linalg.norm( convergedStates, dim=0 )
    
    
    
    maxStateVal = torch.max(  torch.max( state )  )
    
    N = model.networkDim
    xVals = list( range(360) )
    
    
    # fig, axs = plt.subplots( 2, 1, height_ratios=[2,1] )
    fig, axs = plt.subplots( 3, 1, height_ratios=[4,1,1] )
    # for i in range( N ): 
    #     currNeuron = state[ i, : ]
    
    
    xVals = np.zeros( [nTestBetas, N] )
    yVals = np.zeros( [nTestBetas, N] )
    colorVals = np.zeros( [nTestBetas, N] )
    
    
    for k in range( nTestBetas ): 
        
        currRad = testingInput.thetas[0][ k ]
        currTheta = currRad * (180/math.pi)
    
    
        # converged = convergedStates[ :, k+1 ]           ##  +1 to account for initState    
        converged = converged_normalized[ :, k+1 ]           ##  +1 to account for initState    
        convergedNorm = torch.linalg.norm( converged )
        
    
        xVals[k] = [ currTheta.detach().numpy() ] * N
        yVals[k] = list( range(1,N+1) )
        colorVals[k] = converged.detach().numpy()
        
        
    axs[0].scatter( xVals, yVals, s=5, c=colorVals, cmap=cmap )
    
        
        
    fig.subplots_adjust( right=0.8 )
    # cbar_ax = fig.add_axes(  [0.85, 0.15, 0.05, 0.7]  )
    cbar_ax = fig.add_axes(  [0.85, 0.1, 0.05, 0.8]  )
    plt.colorbar( mappable, cax=cbar_ax, label='' )
    # plt.colorbar( mappable, cax=cbar_ax )
        
        
    # # axs[0].scatter( xVals, yVals, s=8, c=colorVals )
    axs[0].set_ylabel( 'Subnetwork ind' )
    # axs[0].set_xlabel( r'$\theta$' )
    axs[0].get_xaxis().set_ticks([])
    
    # axs[0].set_ylabel(  r'$\| \mathbf{r}(t)  \|$'  )
    # plt.colorbar( cmap )
    
    
    trainingRads = trainingInput.thetas
    trainingThetas = trainingRads * (180/math.pi)
    axs[1].hist( trainingThetas, bins=90 )
    # axs[1].set_xlabel( r'$\theta$' )
    axs[1].set_ylabel(  'Training \n frequency'  )
    axs[1].get_xaxis().set_ticks([])
    
    
    trainingRads = testingInput.thetas
    trainingThetas = trainingRads * (180/math.pi)
    axs[2].hist( trainingThetas, bins=90 )
    axs[2].set_xlabel( r'$\theta$' )
    axs[2].set_ylabel(  'Testing \n frequency'  )
    axs[2].get_xaxis().set_ticks( list(range(0,361,60)) )
    
    
        
        
    plt.suptitle( 'Subnetwork Tuning: epoch {}'.format(epochNum) )
    
    




#%%









#%% Is reconstruction error...  min'd  OR  just in null(D)?


epochNum = nEpochs 


convergedStates = testingData[ epochNum ][ 'convergedStates_encoding' ].detach().numpy()


H = epochModels[ epochNum ].H.detach().numpy()  
reconstructed = H @ convergedStates


reconErr = convergedStates[ :, 0:-1 ]   -   reconstructed[ :, 1:: ]
nConvergStates = reconErr.shape[1]




xPts = [ range(nConvergStates) ] 

# plt.plot(  np.linalg.norm( reconErr, axis=0 ), label='reconErr'  )
plt.scatter(  xPts,   np.linalg.norm( reconErr, axis=0 ), label='reconErr', s=4  )



# D = epochModels[ epochNum ].D.detach().numpy()
plt.scatter(  xPts,   np.linalg.norm( D @ reconErr, axis=0 ) , label='D*reconErr', s=4 )


plt.legend( )
plt.title( 'Norm of reconstruction error' )









#%%



plt.plot( testingData[ epochNum ][ 'state' ].detach().numpy()[:,0:100].T )


plt.scatter( [stimTimeInds[0:4]]*d,   D @ convergedStates[ :, 0:4 ]  )






#%% Reconstruction 


epochNum = nEpochs
epochNum = min( [ 1000, nEpochs ] )
# epochNum = 750
# epochNum = 500
# epochNum = 0


model = epochModels[ epochNum ] 
modelData = testingData
modelInput = testingInput


fig, axs = plt.subplots( ) 
# fig, axs = plt.subplots( 3, 1 ) 


labels = [ ]
handles = [ ]



nBetasToPlot = 6

printLegend = True
# printLegend = False

nEncodSteps_testing = modelData['nEncodSteps_testing']
nRetenSteps = modelData['nRetenSteps']
nTimeIndsToPlot = (nEncodSteps_testing + nRetenSteps) * nBetasToPlot






#-----------------------------------------------------------------------------------------
''' State '''
#-----------------------------------------------------------------------------------------
D = model.D  
states = modelData[ epochNum ][ 'state' ]               ## ( N, nTimes )
decoded = D @ states                                    ## ( d, nTimes-1 )
# decoded = decoded[ :, 1:: ].detach().numpy()
decoded = decoded.detach().numpy()



decodedToPlot = decoded[ :, 0:nTimeIndsToPlot ]

plt.scatter(  np.array([range(nTimeIndsToPlot)]*d).T,  decodedToPlot.T, s=2, c='red' )
stateLine = plt.plot( decodedToPlot.T, linewidth=1, c='red' )


labels.append( r'$\mathbf{Dr}(t)$' )
handles.append( stateLine[0] )





#-----------------------------------------------------------------------------------------
''' Reconstructed '''
#-----------------------------------------------------------------------------------------
H = model.H

reconstructed1 = H @ states                                         ## ( N, nTimes )
decodedRecon1 = D @ reconstructed1                                  ## ( d, nTimes-1 )
# decodedRecon1 = decodedRecon1[ :, 1:: ].detach().numpy()
decodedRecon1 = decodedRecon1.detach().numpy()


plt.scatter(  np.array([range(nTimeIndsToPlot)]*d).T,  decodedRecon1[ :, 0:nTimeIndsToPlot ].T, s=2, c='orange' )
reconLine1 = plt.plot( decodedRecon1[ :, 0:nTimeIndsToPlot ].T , linewidth=1,  c='orange' )


labels.append( r'$\mathbf{DHr}(t)$' )
handles.append( reconLine1[0] )


#---------------------------------------------


reconstructed2 = H @ H @ states                                     ## ( N, nTimes )
decodedRecon2 = D @ reconstructed2                                  ## ( d, nTimes-1 )
# decodedRecon2 = decodedRecon2[ :, 1:: ].detach().numpy()
decodedRecon2 = decodedRecon2.detach().numpy()


plt.scatter(  np.array([range(nTimeIndsToPlot)]*d).T,  decodedRecon2[ :, 0:nTimeIndsToPlot ].T, s=2, c='green' )
reconLine2 = plt.plot( decodedRecon2[ :, 0:nTimeIndsToPlot ].T , linewidth=1,  c='green' )

    
labels.append( r'$\mathbf{DH^2r}(t)$' )
handles.append( reconLine2[0] )


#---------------------------------------------


H3 = H @ H @ H
reconstructed3 = H3 @ states                                        ## ( N, nTimes )
decodedRecon3 = D @ reconstructed3                                  ## ( d, nTimes-1 )
# decodedRecon3 = decodedRecon3[ :, 1:: ].detach().numpy()
decodedRecon3 = decodedRecon3.detach().numpy()


# reconLine3 = plt.plot( decodedRecon3[ :, 0:nTimeIndsToPlot ].T , linewidth=2,  c='cyan' )

    
# labels.append( r'$\mathbf{DH^3r}(t)$' )
# handles.append( reconLine3[0] )




#---------------------------------------------


H4 = H @ H @ H @ H
reconstructed4 = H4 @ states                                        ## ( N, nTimes )
decodedRecon4 = D @ reconstructed4                                  ## ( d, nTimes-1 )
# decodedRecon4 = decodedRecon4[ :, 1:: ].detach().numpy()
decodedRecon4 = decodedRecon4.detach().numpy()


# reconLine4 = plt.plot( decodedRecon4[ :, 0:nTimeIndsToPlot ].T , linewidth=2,  c='blue' )

    
# labels.append( r'$\mathbf{DH^4r}(t)$' )
# handles.append( reconLine4[0] )





#-----------------------------------------------------------------------------------------
''' Stimuli '''
#-----------------------------------------------------------------------------------------
stimMat = modelInput.stimMat 


stimTimeInds = modelData[ 'stimTimeInds' ]



for k in range( nBetasToPlot ):    
    plt.axvline(  stimTimeInds[k],  c='k',  linewidth=1  ) 
        
    stimDot = plt.scatter(  [ stimTimeInds[k] ]*d,  stimMat[:,k], facecolors='none', edgecolors='r', s=50  ) 


labels.append( r'$\mathbf{x}(t)$' )
handles.append( stimDot )



#-----------------------------------------------------------------------------------------
''' Reference States '''
#-----------------------------------------------------------------------------------------
if nRetenSteps > 0:
    refStates = modelData[ epochNum ][ 'convergedStates_retention' ]    ## ( N, nStims+1 )
else: 
    refStates = modelData[ epochNum ][ 'convergedStates_encoding' ]     ## ( N, nStims+1 )


decodedRefs = D.detach() @ refStates[ :, 1:: ].detach().numpy()  ## ( d, nStims )


for k in range( nBetasToPlot ):
    delayStart = stimTimeInds[k+1]-nRetenSteps
    delayEnd = delayStart + nRetenSteps
    plt.axvline(  delayStart,  c='gray',  linestyle='--',  linewidth=1  ) 
            
    refDot = plt.scatter(  [ delayEnd ]*d,  decodedRefs[:,k], facecolors='none', edgecolors='orange', s=50, marker='s'  ) 



labels.append( r'$\mathbf{r}^-(t)$' )
handles.append( refDot )




#-----------------------------------------------------------------------------------------
''' Put it all together '''
#-----------------------------------------------------------------------------------------
# plt.title( 'Reconstruction' )
plt.suptitle( 'Reconstruction: epoch {}'.format( epochNum ) )


if printLegend:
    plt.legend(  handles, labels  )


plt.xlabel( 't' )
plt.ylabel( r'$\mathbf{DH}^q\mathbf{r}(t)$' )





#%% Stability of H 


def numberOfStableEigvals( matrix ): 
    
    
    eigvals = torch.linalg.eigvals( matrix )
    
    eigvalsReal = torch.real( eigvals )
    hurwitzBool = eigvalsReal < 0
    
    nStable = sum( hurwitzBool )
    
    
    return nStable


#-----------------------------------------------------------------------------------------


hurwitzCount_H = np.zeros(  [ len(epochNumsToTest), 2 ]  )
# hurwitzCount_D = np.zeros(  [ len(epochNumsToTest), 2 ]  )


for i in range( len(epochNumsToTest) ):
    
    epochNum = epochNumsToTest[ i ]


    #--------------------------------------------
    H = epochModels[epochNum].H.detach()
    
    nStable = numberOfStableEigvals( H )
    nUnstable = N - nStable
    
    hurwitzCount_H[ i ] = [ epochNum, nUnstable ]
    #--------------------------------------------
    
    
    # #--------------------------------------------
    # D = epochModels[epochNum].D.detach()
    
    # nStable = numberOfStableEigvals( D )
    # nUnstable = N - nStable
    
    # hurwitzCount_D[ i ] = [ epochNum, nUnstable ]
    # #--------------------------------------------


print( )
print( '--------------------------------------------' )
print( 'Number of unstable eigvals by epochNum:  H ' )
print( '--------------------------------------------' )
print( hurwitzCount_H )
# print( )
# print( 'D' )
# print( hurwitzCount_D )






#%% condition number 


print( )
print( 'Condition number of H: ' )
print( )


for i in range( len(epochNumsToTest) ):
    
    epochNum = epochNumsToTest[ i ]


    #--------------------------------------------
    H = epochModels[epochNum].H.detach().numpy()
    
    condNum = np.linalg.cond( H )
    
    print( epochNum, condNum )
    #--------------------------------------------
    

#======================================================================
    
    
print( )
print( 'Condition number of D: ' )
print( )


for i in range( len(epochNumsToTest) ):
    
    epochNum = epochNumsToTest[ i ]


    #--------------------------------------------
    D = epochModels[epochNum].D.detach().numpy()
    
    condNum = np.linalg.cond( D )
    
    print( epochNum, condNum )
    #--------------------------------------------
    







#%% refStates versus origStims 



epochNum = 400

modelData = testingData
modelInput = testingInput
model = epochModels[epochNum] 


if nRetenSteps > 0:
    refStates = modelData[ epochNum ][ 'convergedStates_retention' ]    ## ( N, nStims+1 )
else: 
    refStates = modelData[ epochNum ][ 'convergedStates_encoding' ]     ## ( N, nStims+1 )



D = model.D 
decodedRefs = D @ refStates[ :, 1:: ]           ## ( d, nStims )


stimMat = modelInput.stimMat                    ## ( d, nStims )


diff = decodedRefs - stimMat 
diffNorms = torch.linalg.norm( diff, ord=2, dim=0 )


print(  np.round( diffNorms.detach().numpy(), 2 )  )


# plt.title( 'Decoded State' )




#%% refStates versus state 



epochNum = 100

modelData = testingData
modelInput = testingInput
model = epochModels[epochNum] 


if nRetenSteps > 0:
    refStates = modelData[ epochNum ][ 'convergedStates_retention' ]    ## ( N, nStims+1 )
else: 
    refStates = modelData[ epochNum ][ 'convergedStates_encoding' ]     ## ( N, nStims+1 )


D = model.D  
states = modelData[ epochNum ][ 'state' ]               ## ( N, nTimes )
decoded = D @ states[ :, 1:: ]                          ## ( d, nTimes-1 )
decoded = decoded.detach().numpy()

decodedRefs = D.detach() @ refStates[ :, 1:: ].detach().numpy()  ## ( d, nStims )


stimMat = modelInput.stimMat 



nBetasToPlot = 4
nEncodSteps_testing = modelData['nEncodSteps_testing']
nRetenSteps = modelData['nRetenSteps']
nTimeIndsToPlot = (nEncodSteps_testing + nRetenSteps) * nBetasToPlot


stimTimeInds = modelData[ 'stimTimeInds' ]




    
for k in range( nBetasToPlot ):
    delayStart = stimTimeInds[k+1]-nRetenSteps
    
    plt.axvline(  stimTimeInds[k],  c='k',  linewidth=1  )         
    plt.axvline(  delayStart,  c='gray',  linewidth=1  ) 
            
    stimDot = plt.scatter(  [ stimTimeInds[k] ]*d,  stimMat[:,k],  c='red'  ) 
    refDot = plt.scatter(  [ delayStart ]*d,  decodedRefs[:,k],  c='green'  ) 



stateLine = plt.plot( decoded[ :, 0:nTimeIndsToPlot ].T , linewidth=3, c='k')

# # plt.scatter(   [list(range(nTimeIndsToPlot))] * d ,   decoded[ :, 0:nTimeIndsToPlot ].T,   linewidth=0.1   )
# for i in range( d ):
#     plt.scatter(   list(range(nTimeIndsToPlot)),   decoded[ i, 0:nTimeIndsToPlot ].T,   linewidth=0.1,   c='k'   )


plt.title( 'Decoded State: $\mathbf{D}\mathbf{r}(t)$' )






labels = [ '$\mathbf{Dr}(t)$',  '$\mathbf{z}^-(t)$', '$\mathbf{x}(t)$' ]
plt.legend(  [stateLine[0], refDot, stimDot],  labels   )






# diff = decoded - stimMat 
# diffNorms = torch.linalg.norm( diff, ord=2, dim=0 )


# print(  np.round( diffNorms.detach().numpy(), 2 )  )



#%% refStates versus reconstructed



epochNum = 100

modelData = testingData
modelInput = testingInput
model = epochModels[epochNum] 


if nRetenSteps > 0:
    refStates = modelData[ epochNum ][ 'convergedStates_retention' ]    ## ( N, nStims+1 )
else: 
    refStates = modelData[ epochNum ][ 'convergedStates_encoding' ]     ## ( N, nStims+1 )


states = modelData[ epochNum ][ 'state' ]                           ## ( N, nTimes )

H = model.H
reconstructed = H @ states                                          ## ( N, nTimes )

D = model.D  
decodedRecon = D @ reconstructed[ :, 1:: ]                          ## ( d, nTimes-1 )
decodedRecon = decodedRecon.detach().numpy()

decodedRefs = D.detach() @ refStates[ :, 1:: ].detach().numpy()     ## ( d, nStims )


states = modelData[ epochNum ][ 'state' ]               ## ( N, nTimes )
decoded = D @ states[ :, 1:: ]                          ## ( d, nTimes-1 )


stimMat = modelInput.stimMat 



nBetasToPlot = 4
nEncodSteps_testing = modelData['nEncodSteps_testing']
nRetenSteps = modelData['nRetenSteps']
nTimeIndsToPlot = (nEncodSteps_testing + nRetenSteps) * nBetasToPlot


stimTimeInds = modelData[ 'stimTimeInds' ]




    
for k in range( nBetasToPlot ):
    delayStart = stimTimeInds[k+1]-nRetenSteps
    
    stimTime = plt.axvline(  stimTimeInds[k],  c='k',  linewidth=1  )         
    delayTime = plt.axvline(  delayStart,  c='gray',  linewidth=1  ) 
            
    stimDot = plt.scatter(  [ stimTimeInds[k] ]*d,  stimMat[:,k],  c='red'  ) 
    refDot = plt.scatter(  [ delayStart ]*d,  decodedRefs[:,k],  c='green'  ) 




# stateLine = plt.plot( decoded[ :, 0:nTimeIndsToPlot ].T , linewidth=3, c='k',  label='Dr(t)'  )
stateLine = plt.plot( decoded[ :, 0:nTimeIndsToPlot ].T , linewidth=3, c='k' )

# reconLine = plt.plot( decodedRecon[ :, 0:nTimeIndsToPlot ].T , linewidth=3,  c='orange',  label='DHr(t)' )
reconLine = plt.plot( decodedRecon[ :, 0:nTimeIndsToPlot ].T , linewidth=3,  c='orange' )




# # plt.scatter(   [list(range(nTimeIndsToPlot))] * d ,   decodedRecon[ :, 0:nTimeIndsToPlot ].T,   linewidth=0.1   )
# for i in range( d ):
#     plt.scatter(   list(range(nTimeIndsToPlot)),   decodedRecon[ i, 0:nTimeIndsToPlot ].T,   linewidth=0.1,   c='k'   )

plt.title( 'Decoded Reconstruction: $\mathbf{DH}\mathbf{r}(t)$' )



# plt.legend( [stateLine, reconLine, stimDot, refDot], ['Dr(t)', 'DHr(t)', 'x(t)', '$z^-(t)$'] )
plt.legend(  )







#%% refStates versus reconstructed (q=2)



epochNum = 100

modelData = testingData
modelInput = testingInput
model = epochModels[epochNum] 


if nRetenSteps > 0:
    refStates = modelData[ epochNum ][ 'convergedStates_retention' ]    ## ( N, nStims+1 )
else: 
    refStates = modelData[ epochNum ][ 'convergedStates_encoding' ]     ## ( N, nStims+1 )


states = modelData[ epochNum ][ 'state' ]                           ## ( N, nTimes )


H = model.H
D = model.D  


reconstructed1 = H @ states                                         ## ( N, nTimes )
decodedRecon1 = D @ reconstructed1[ :, 1:: ]                        ## ( d, nTimes-1 )
decodedRecon1 = decodedRecon1.detach().numpy()

reconstructed2 = H @ H @ states                                     ## ( N, nTimes )
decodedRecon2 = D @ reconstructed2[ :, 1:: ]                        ## ( d, nTimes-1 )
decodedRecon2 = decodedRecon2.detach().numpy()



decodedRefs = D.detach() @ refStates[ :, 1:: ].detach().numpy()     ## ( d, nStims )


stimMat = modelInput.stimMat 



nBetasToPlot = 4
nEncodSteps_testing = modelData['nEncodSteps_testing']
nRetenSteps = modelData['nRetenSteps']
nTimeIndsToPlot = (nEncodSteps_testing + nRetenSteps) * nBetasToPlot


stimTimeInds = modelData[ 'stimTimeInds' ]




    
for k in range( nBetasToPlot ):
    delayStart = stimTimeInds[k+1]-nRetenSteps
    
    stimTime = plt.axvline(  stimTimeInds[k],  c='k',  linewidth=1  )         
    delayTime = plt.axvline(  delayStart,  c='gray',  linewidth=1  ) 
            
    # stimDot = plt.scatter(  [ stimTimeInds[k] ]*d,  stimMat[:,k],  c='red'  ) 
    refDot = plt.scatter(  [ delayStart ]*d,  decodedRefs[:,k],  c='green'  ) 




# stateLine = plt.plot( decoded[ :, 0:nTimeIndsToPlot ].T , linewidth=3, c='k',  label='Dr(t)'  )
# stateLine = plt.plot( decoded[ :, 0:nTimeIndsToPlot ].T , linewidth=3, c='k' )

# reconLine = plt.plot( decodedRecon[ :, 0:nTimeIndsToPlot ].T , linewidth=3,  c='orange',  label='DHr(t)' )



# stateLine = plt.plot( decoded[ :, 0:nTimeIndsToPlot ].T , linewidth=2, c='red' )
reconLine1 = plt.plot( decodedRecon1[ :, 0:nTimeIndsToPlot ].T , linewidth=2,  c='orange' )
reconLine2 = plt.plot( decodedRecon2[ :, 0:nTimeIndsToPlot ].T , linewidth=2,  c='gold' )




plt.title( 'Reconstruction' )



# labels = [ '$\mathbf{Dr}(t)$',  '$\mathbf{DHr}(t)$',  '$\mathbf{DH}^2\mathbf{r}(t)$', '$\mathbf{x}(t)$']
# plt.legend(  [stateLine[0], reconLine1[0], reconLine2[0], refDot],  labels   )



# labels = [ '$\mathbf{DHr}(t)$',  '$\mathbf{DH}^2\mathbf{r}(t)$', '$\mathbf{z}^-(t)$', '$\mathbf{x}(t)$']
# plt.legend(  [reconLine1[0], reconLine2[0], refDot, stimDot],  labels   )



labels = [ '$\mathbf{DHr}(t)$',  '$\mathbf{DH}^2\mathbf{r}(t)$', '$\mathbf{z}^-(t)$']
plt.legend(  [reconLine1[0], reconLine2[0], refDot],  labels   )








#%% 


nBetasToLookAt = 5
nBetasToLookAt = 20
timeIndToGoTo = (nBetasToLookAt * nTotalEvolSteps) + 1

threshold = sWeightList[0] * stepSize.detach().numpy()
    

legendEntries = [ ]


    
for epochNum in epochNumsToTest:

    
    #--------------------------------------------------------
    ''' Create figure '''
    #--------------------------------------------------------
    if d == 2:     
        fig = plt.figure().add_subplot(projection='3d')
        # fig = plt.figure()
    else: 
        fig = plt.figure() 
    
    
    
    #--------------------------------------------------------
    ''' Get state evolution '''
    #--------------------------------------------------------
    stateEvol = testingData[ epochNum ][ 'state' ]
    
    model = epochModels[ epochNum ]
    D = model.D
    decoded = D @ stateEvol 

    
    #-------------------------------------------------------------------------------------
    ''' Plot (decoded) state evolution '''
    #-------------------------------------------------------------------------------------
    decodedToPlot = decoded[ :, 0:timeIndToGoTo  ].detach().numpy()

    if d == 2:
        # plt.scatter( decodedToPlot[0,:], decodedToPlot[1,:], c='k', s=3 )
        
        decodedColors = [ mpl.colormaps['viridis'](i) for i in range( 256-timeIndToGoTo, 256 ) ]
        # plt.scatter( decodedToPlot[0,:], decodedToPlot[1,:], s=4, c=decodedColors )
        # cbar = plt.colorbar( label='t' )
        
        x = decodedToPlot[0,:]
        y =  decodedToPlot[1,:]
        z = np.linspace( 0, timeIndToGoTo, timeIndToGoTo )
        
        fig.scatter( x, y, z, s=5, c='k' )
        fig.plot( x, y, z, c='k', linewidth=0.5 )
        # fig.scatter( x, y, z, s=5, c=decodedColors )
        legendEntries.append( 'decoded' )
        # fig.scatter( x, y, z, s=5, c='k' )

    else: 
        plt.plot( decodedToPlot.T, c='k' )
        legendEntries.append( 'decoded' )
    # plt.plot( decodedToPlot.T, c='k' )
    


    
    #-------------------------------------------------------------------------------------
    ''' Plot the input stims '''
    #-------------------------------------------------------------------------------------
    stimsToPlot = testingInput.stimMat[ :, 0:nBetasToLookAt ]
    
    x = stimsToPlot[ 0, : ].numpy()
    y = stimsToPlot[ 1, : ].numpy()
    convergeTimes = np.array(  [ (stimTimeInds[i]+nTotalEvolSteps) for i in range(nBetasToLookAt) ]  )
    
    fig.scatter( x, y, convergeTimes, s=10, c='r', marker='*' )
    legendEntries.append( 'stims' )
    
    
    
        
        
    #-------------------------------------------------------------------------------------
    ''' Adjust fig '''
    #-------------------------------------------------------------------------------------
    plt.legend( legendEntries )
    plt.title( 'epoch: ' + str(epochNum) )
    
    fig.set_xlabel( 'x' )
    fig.set_ylabel( 'y' )
    fig.set_zlabel( 't' ) 
    
    # plt.colorbar( fig )
    # fig.view_init(elev=45, azim=90, roll=270)
    
        
    
    

#%%
    
    
for epochNum in epochNumsToTest:

    fig = plt.figure()    

    
    stateEvol = testingData[ epochNum ][ 'state' ]
    
    
    stateToPlot = stateEvol[ :, 0:timeIndToGoTo  ]
    
    stimsToPlot = testingInput.stimMat[ :, 0:nBetasToLookAt ]
    
    
    # plt.plot( decoded.detach().numpy().T )
    plt.plot( stateToPlot.detach().numpy().T, c='k' )
    
    
    
    
    
    for i in range( nBetasToLookAt ):
        
        currStim = testingInput.stimMat[ :, i ]
        
        
        stimTime = stimTimeInds[i]
        retenTime = stimTime + nEncodSteps_testing
        
        plt.axvline( retenTime, c='gray', linestyle='--', linewidth=1 )
        
        convergeTime = stimTime + nTotalEvolSteps
        plt.axvline( convergeTime, c='gray', linewidth=2 )
        
        # print( )
        # print( 'retenTime', retenTime )
        # print( 'convergTime', convergTime )
        
        
        # plt.scatter( [convergeTime]*d, currStim, c='r' )
        
        
        
    # plt.legend( [ 'z(t)', 'start of reten', 'convergence', 'goal (stim)' ] )
    plt.title( 'epoch: ' + str(epochNum) )
    plt.ylabel( 'z' )
    plt.xlabel( 't' )
        
    plt.axhline( threshold, c='gray', linewidth=0.5 )
    plt.axhline( -threshold, c='gray', linewidth=0.5 )

    
    
    
    
    
    
    
    
    










#=========================================================================================
#=========================================================================================
#%% SAVE 
#=========================================================================================
#=========================================================================================



#-----------------------------------------------------------------------------------------
''' Where to save '''
#-----------------------------------------------------------------------------------------

if simOptions[ 'loadPrevData' ]:
    saveFolder = referenceFolder
else:
    saveFolder = nameSaveDir( simOptions, weightCombos, nEpochs )


print( )
print( 'saveFolder: ', saveFolder )
print( )






#-----------------------------------------------------------------------------------------
''' Save '''
#-----------------------------------------------------------------------------------------
saveDir = os.path.join( currDir, saveFolder ) 

filenames = saveModelInfo( locals(), saveDir )



# figFilenames = saveFigureDict(  figDict,  saveDir,  imgTypes=['.svg', '.png']  )






#=========================================================================================
#=========================================================================================
#%% LOAD 
#=========================================================================================
#=========================================================================================


simOptions['nProxSteps'] = 200
simOptions['Epochs'] = nEpochs = 5000
simOptions['learningRate'] = 0.01
simOptions['maxIter'] = 200
simOptions['trainMatsDirectly'] = False


#-----------------------------------------------------------------------------------------

if simOptions['learnSingleStimUntilConvergence']:
    saveFolder = 'retrainOnStim' + str( simOptions['maxIter'] )
else: 
    saveFolder = 'singleTrainOnStim'
    
    
saveFolder = saveFolder  +  '_prox'  +  str( simOptions['nProxSteps'] )
saveFolder = saveFolder  +  '_lr'  +  str( simOptions['learningRate'] )

saveFolder = os.path.join( saveFolder, 'Epochs' + str(nEpochs) )  

#-----------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------
saveDir = os.path.join( currDir, saveFolder ) 

modelInfoDict = getModelInfo( saveDir )


trainingModel = modelInfoDict[ 'trainingModel' ]
trainingInput = modelInfoDict[ 'trainingInput' ]
trainingData = modelInfoDict[ 'trainingData' ]

epochModels = modelInfoDict[ 'epochModels' ]
testingInput = modelInfoDict[ 'testingInput' ]


simOptions = modelInfoDict[ 'simOptions' ]

epochNumsToTest = simOptions['epochNumsToTest']
learningRate = simOptions['learningRate']
nProxSteps = simOptions['nProxSteps']
# optimizer = simOptions['optimizer']
simOptions['optimizer'] = optimizer = torch.optim.Adam( trainingModel.parameters(), lr=simOptions['learningRate'] )
#-----------------------------------------------------------------------------------------









#%%


convergStates = [ testingData[i]['convergedStates'].numpy() for i in epochNumsToTest ]
convergStates = np.array( convergStates )


decodersForEpochsToTest = [ epochModels[i].computeDandH()[0].detach().numpy() for i in epochNumsToTest ]
decodersForEpochsToTest = np.array( decodersForEpochsToTest )


fig = plt.figure( )
for epochInd in range( 3 ):
    currEpochStates = convergStates[ epochInd ]           # ( N, nBetas )
    plt.plot( currEpochStates.T )
plt.title( 'States' )




convergDecoded = decodersForEpochsToTest @ convergStates        # ( nEpochNumsToTest, d, nBetas )

fig = plt.figure( )
for epochInd in range( 3 ):
    currEpochDecoded = convergDecoded[ epochInd ]           # ( d, nBetas )
    plt.plot( currEpochDecoded.T )
plt.title( 'Decoded' )



plt.figure() 
stims = testingInput.stimMat.numpy()
plt.plot( stims.T )
plt.title( 'Stims' )


#% %
plt.figure()
for epochInd in range( 3 ):
    currEpochDecoded = convergDecoded[ epochInd ]           # ( d, nBetas )
    encodingDiff = stims - currEpochDecoded[ :, 1:: ]
    plt.plot( encodingDiff.T )
plt.title( 'Encoding Diff' )


#%%


plt.figure()

for epochInd in range( 3 ):
    currEpochDecoded = convergDecoded[ epochInd ]           # ( d, nBetas )
    encodingDiff = stims - currEpochDecoded[ :, 1:: ]
    
    normedError = np.linalg.norm(  encodingDiff, axis=0  )
    denom = np.linalg.norm(  stims, axis=0  )

    percentError = normedError / denom
    percentError = percentError.clip( max=1 ) 

    finalVal = (1 - percentError)

    plt.plot( finalVal.T )
    
    print( epochNumsToTest[epochInd],'... ', np.mean( finalVal ) )
plt.title( 'RP, q=0' )










#%% 
 





#%% 




    
    
naiveModel = epochModels[ 0 ]
initW = naiveModel.W 
initMs = naiveModel.Ms 
initMd = naiveModel.Md

finalModel = epochModels[  epochNumsToTest[-1]  ]
finalW = finalModel.W 
finalMs = finalModel.Ms 
finalMd = finalModel.Md




figW, axsW = plt.subplots( 1,2 )
axsW[0].imshow( initW.detach().numpy() )
axsW[1].imshow( finalW.detach().numpy() )
# plt.colorbar( )
axsW[0].set_title( 'W' )


figMs, axsMs = plt.subplots( 1,2 )
axsMs[0].imshow( initMs.detach().numpy() )
axsMs[1].imshow( finalMs.detach().numpy() )
# plt.colorbar()
axsMs[0].set_title( 'Ms' )


figMd, axsMd = plt.subplots( 1,2 )
axsMd[0].imshow( initMd.detach().numpy() )
axsMd[1].imshow( finalMd.detach().numpy() )
# plt.colorbar()
axsMd[0].set_title( 'Md' )
    
    

#%%



fig = plt.figure() 
plt.imshow(  trainingData['W'][0].detach().numpy()  )
plt.colorbar()
plt.title( 'W (epoch 0)' )

fig = plt.figure() 
plt.imshow(  trainingData['W'][nEpochs].detach().numpy()  )
plt.colorbar()
plt.title( 'W (epoch 800)' )



fig = plt.figure() 
plt.imshow(  trainingData['Md'][0].detach().numpy()  )
plt.colorbar()
plt.title( 'Md (epoch 0)' )

fig = plt.figure() 
plt.imshow(  trainingData['Md'][nEpochs].detach().numpy()  )
plt.colorbar()
plt.title( 'Md (epoch 800)' )



maxMs = max(  torch.max(finalMs), torch.max(initMs)  )
minMs = max(  torch.min(finalMs), torch.min(initMs)  )


fig = plt.figure() 
plt.imshow(  trainingData['Ms'][0].detach().numpy(), vmin=minMs, vmax=maxMs  )
cbar = plt.colorbar(  )
# cbar.clim( [ -11, 11 ] )
plt.title( 'Ms (epoch 0)' )

fig = plt.figure() 
plt.imshow(  trainingData['Ms'][nEpochs].detach().numpy(), vmin=minMs, vmax=maxMs  )
cbar = plt.colorbar(  )
# cbar.clim( [ -11, 11 ] )
plt.title( 'Ms (epoch 800)' )









#%%



#%%

saveDir = currDir 

filenames = saveModelInfo( localVars(), saveDir )



#%%

import pickle



with open('trainingModel.pickle', 'wb') as handle:
    pickle.dump(trainingModel, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('trainingInput.pickle', 'wb') as handle:
    pickle.dump(trainingInput, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('trainingData.pickle', 'wb') as handle:
    pickle.dump(trainingData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
with open('testingModels.pickle', 'wb') as handle:
    pickle.dump(epochModels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('testingInput.pickle', 'wb') as handle:
    pickle.dump(trainingModel, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)





#%%


fig = plt.figure() 

testStimMat = testingInput.stimMat


for epochInd in range( len(epochNumsToTest) ):
    
    epochNum = epochNumsToTest[ epochInd ]

    
    if epochNum == 0:
        currLabel = 'Before training'
    else:
        currLabel = 'epoch ' + str(epochNum)
        
    currColor = colors[ epochInd ]



    proxEvol = proximalEvol[ epochNum ]                             # ( nBetas, N, nProxSteps )
    
    D = epochModels[ epochNum ].computeDandH()[0]                 # ( d, N )
    decodedEvol = D @ proxEvol                                      # ( nBetas, d, nProxSteps )
    
    normsToAvg = torch.zeros( nTestBetas, nProxSteps )
    
    for i in range( nTestBetas ):
        
        currStim = testStimMat[:,i]
        
        decodedDiff = [  (decodedEvol[i,:,k] - currStim)  for k in range(nProxSteps) ]         # ( d, nProxSteps ) 
        diffNorm = torch.norm( torch.tensor(decodedDiff), p=2, dim=0 )                               # ( 1, nProxSteps )
        
        diffNorm_normalized = diffNorm / torch.norm( currStim )
        
        normsToAvg[ i ] = diffNorm
        normsToAvg[ i ] = diffNorm_normalized
        
    # proxDiff = proxEvol[ :, :, 1:: ]  -  proxEvol[ :, :, 0:-1 ]     # ( N, nBetas, nProxSteps-1 )
    # diffNorm = torch.norm( proxDiff, p=2, dim=0 )                   # ( nBetas, nProxSteps-1 )
    
    # avgConvergence = torch.mean( diffNorm, dim=0 )                  # ( nProxSteps-1 )    
    # plt.plot( diffNorm.numpy(), c=currColor, label=currLabel, linewidth=5 )
    # plt.scatter( list(range(nProxSteps-1)), avgConvergence.numpy(), c=currColor, label=currLabel, s=5 )

    
    avgConvergence = torch.mean( normsToAvg, dim=0 )
    plt.scatter( list(range(nProxSteps)), avgConvergence.numpy(), c=currColor, label=currLabel, s=5 )
    
        
    
plt.title( 'Proximal convergence' )
    






#%%


''' Forward Horizon to plot '''
#-------------------------------
FH = nBetas - 1
# # FH = nBetas - 2
# FH = int( nBetas / 3 ) 

FH = 4

FH = min(  [ FH, 7 ]  )
    
    
    

fig, axs = plt.subplots( 1, 1 )
colors = [ 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'black' ]

        
    
    











#%%






# def initVarStorage


for stimInd in range( nTestBetas ):

    print( '\ni={}'.format(stimInd) )    
    print( '--------' )    

    for k in range( nProxSteps ):    

        t = stimInd * nProxSteps + k       
        print( 't={}'.format(t) )
        
        
        
#%%
        


nEncodSteps = 10 
nRetenSteps = 5
nTotalEvolSteps = nEncodSteps + nRetenSteps



nStims = 3        
        
nTimes = (nTotalEvolSteps * nStims)  +  1


test = torch.tensor( np.linspace( 0, nTimes-1, nTimes ) )

for i in range( 1, nStims+1 ):
    
    startTime = (i-1) * nTotalEvolSteps + 1
    endTime = startTime + nEncodSteps

    startTime2 = endTime
    endTime2 = startTime2 + nRetenSteps
    
    print( )
    print( 'startTime', startTime )
    print( 'endTime', endTime )
    print( test[ startTime:endTime ] )
    
    print( 'startTime2', startTime2 )
    print( 'endTime2', endTime2 )
    print( test[ startTime2:endTime2 ] )




#%%

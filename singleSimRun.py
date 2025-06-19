#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:58:31 2024

@author: bethannajones
"""







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
#%% Training
#=========================================================================================
#=========================================================================================


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


optimizer = torch.optim.Adam( trainingModel.parameters(), lr=learningRate )




simOptions['optimizer'] = optimizer






#=========================================================================================
''' TRAIN '''
#=========================================================================================
print(  )
print(  '=========================' )
print(  'TRAINING ' )
print(  '=========================' )
print(  'nEpochs = {}'.format(nEpochs)  )
print(  'Cost weight combo: {}'.format(weightCombos[0])  )
print(  )
print(  'nEncodSteps = {}'.format(nEncodSteps_training)  )
print(  'learningRate = {}'.format(learningRate)  )
print(  'learnSingleStimUntilConvergence:', simOptions['learnSingleStimUntilConvergence'] ) 
if simOptions['learnSingleStimUntilConvergence']: 
    print(  '\t maxIter = {}'.format(simOptions['parameters']['maxIter'])  )
print(  )
print(  )



start = time.time()


[ trainingModel, trainingData ] = trainModel( trainingModel, trainingInput, simOptions,
                                             # includeDelay=simOptions['includeDelayInTraining'],  
                                             )


print( )
printTimeElapsed(  start,  time.time()  )




#% %
#=========================================================================================
''' Record data '''
#=========================================================================================

if simOptions['includeDelayInTraining']:
    nEvolSteps_training = nEncodSteps_training + nRetenSteps
    trainingData[ 'nRetenSteps' ] = nRetenSteps
else:
    nEvolSteps_training = nEncodSteps_training
    
nTrainingTimes = (nEvolSteps_training * nEpochs) + 1     ## number of time steps 


    

stimTimeInds = [  (nEncodSteps_training*i)+1  for i in range(nTrainingBetas)  ]
convergedTimeIndsReten  = [  ti-1 for ti in stimTimeInds[1::]  ]  +  [ nTrainingTimes-1 ]

convergedTimeInds = [  ti-1 for ti in stimTimeInds[1::]  ]  +  [ nTrainingTimes-1 ]

trainingData[ 'stimTimeInds' ] = stimTimeInds
trainingData[ 'convergedTimeInds' ] = convergedTimeInds
trainingData[ 'nEncodSteps' ] = nEncodSteps_training


testingData[ 'convergedTimeIndsReten' ] = convergedTimeIndsReten








#% %


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


for epochNum in epochNumsToTest:
    
    epochModel = memoryModel( d, N, weightCombos[0], simOptions )

    
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
testingData = { }

testingData[ 'nEncodSteps_testing' ] = nEncodSteps_testing



#=========================================================================================
''' TESTING '''
#=========================================================================================
## https://www.youtube.com/watch?v=Z_ikDlimN6A&t=22898s


print(  )
print(  '=========================' )
print(  'TESTING ' )
print(  '=========================' )
print(  'Cost weight combo: {}'.format(weightCombos[0])  )
print(  'epochNumsToTest = {}'.format(epochNumsToTest)  )
print(  )
print(  'nEncodSteps = {}'.format(nEncodSteps_training)  )
print(  'nRetenSteps = {}'.format(nRetenSteps)  )
print(  )
print(  )







toPrintInd = 0


#------------------------------------------
start = time.time()
printActualTime( )
#------------------------------------------


for epochNum in epochNumsToTest:         # extra idx to account for init

    #------------------------------------------
    ''' The current model '''
    #------------------------------------------
    epochModel = epochModels[ epochNum ]
    # epochModel.to( device )
    epochModel.eval()                           # Set to evaluate 


    epochModel, testingData_epoch = testModel( epochModel, testingInput, simOptions )
        
    testingData[ epochNum ] = testingData_epoch



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
nTotalEvolSteps = nEncodSteps_testing + nRetenSteps
nTestingTimes = (nTotalEvolSteps * nTestBetas)  +  1      ## number of time steps 

stimTimeInds = [  (nTotalEvolSteps*i)+1  for i in range(nTestBetas)  ]
convergedTimeIndsReten  = [  ti-1 for ti in stimTimeInds[1::]  ]  +  [ nTestingTimes-1 ]

convergedTimeInds = [  ti-nRetenSteps-1 for ti in stimTimeInds[1::]  ]  +  [ nTestingTimes-nRetenSteps-1 ]


testingData[ 'stimTimeInds' ] = stimTimeInds
testingData[ 'convergedTimeInds' ] = convergedTimeInds
testingData[ 'nEncodSteps' ] = nEncodSteps_testing
testingData[ 'nRetenSteps' ] = nRetenSteps

testingData[ 'convergedTimeIndsReten' ] = convergedTimeIndsReten



delayPeriodStartTimeInds = [ '' ]
testingData[ 'delayPeriodStartInds' ] = delayPeriodStartTimeInds



#% %




# #=========================================================================================
# ''' Plot encoding performance '''
# #=========================================================================================
# for epochNum in [  epochNumsToTest[0], epochNumsToTest[-1]  ]:    
    
#     epochModel = epochModels[ epochNum ]
#     fig = plotModelEncodingAccur( epochModel, testingInput, testingData )

    






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
    saveFolder = nameSaveDir( simOptions, weightCombos[0], nEpochs )


print( )
print( 'saveFolder: ', saveFolder )
print( )






#-----------------------------------------------------------------------------------------
''' Save '''
#-----------------------------------------------------------------------------------------
saveDir = os.path.join( currDir, saveFolder ) 

filenames = saveModelInfo( locals(), saveDir )



# figFilenames = saveFigureDict(  figDict,  saveDir,  imgTypes=['.svg', '.png']  )






















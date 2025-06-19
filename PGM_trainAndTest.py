#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:35:35 2024

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

# import tensorflow as tf

# from torch.utils.data import Dataset, DataLoader


import numpy as np 
import random




#----------------------------
''' Visualization '''
#-----------------------------
import matplotlib 

import time
from datetime import datetime




#-----------------------------
''' Saving '''
#-----------------------------
import pickle 

import smtplib



import copy 




from PGM_analysis import saveCurrentTrainingInfo





#=========================================================================================
#=========================================================================================
#%% Evolution 
#=========================================================================================
#=========================================================================================


# def evolveProx( model, currStim, stimInd, initState, refState, nProxSteps, useQ2Also=False ): 
def evolveProx( model, currStim, initState, refState, nProxSteps, 
                               prevState=None, useQ2Also=False, refState2=None ): 
    '''  ''' 
    
    
    #------------------------------------------------------
    ''' Set up ''' 
    #------------------------------------------------------
    N = initState.shape[0]
    proxEvol = torch.zeros( [ N, nProxSteps ] )
    
    
    
    #------------------------------------------------------
    ''' 0. Initialize ''' 
    #------------------------------------------------------
    currState = initState
    # currState = initState.clone()
    
    if prevState is None:
        prevState = currState.clone()



    #=====================================================================================
    ''' 1. Evolve the dynamics ''' 
    #=====================================================================================
    if type(model.retentionAlpha) is not str:
        alpha = model.computeAlpha( currStim ) 
    
    
    
    for k in range( nProxSteps ):
        
        
        if type(model.retentionAlpha) is str:
            alpha = model.computeAlpha( currStim, currState, refState )
        
        
        #----------------------------------------------------
        ''' a. Compute the new state '''
        #---------------------------------------------------- 
        proxState = model( currStim, refState, currState, 
                                      useQ2Also=useQ2Also, refState2=refState2 )  
            
        
        retenState = model.retentionWeight * prevState
        # retenState = prevState
        
        # nullSoln = model.computeNullSoln( refState, prevState )
        # retenState = prevState + nullSoln
        # # retenState = nullSoln
        
        
        proxDynam = alpha * proxState
        nullDynam = (1-alpha) * retenState 
        
        newState = proxDynam  +  nullDynam
        
        
        
        
        #----------------------------------------------------
        ''' b. Update for next step '''
        #----------------------------------------------------
        # prevState = currState 
        # currState = newState 
        prevState = currState.clone()
        currState = newState.clone() 
        # prevState = currState.clone().detach()
        # currState = newState.clone().detach()
        
        
        #----------------------------------------------------
        ''' c. Store '''
        #----------------------------------------------------
        proxEvol[ :, k ] = newState.reshape( N, ).clone()
        # proxEvol[ :, k ] = newState.reshape( N, ).clone().detach()
        # proxEvol[ :, k ] = newState.reshape( N, )
        
    
    
    #===============================================
    ''' 2. Grab final state ''' 
    #===============================================
    convergedState = newState    
    # convergedState = newState.detach()    
    
    
    
    return convergedState, proxEvol









# def proxAndLearn( model, currStim, epochNum, currState, refState, nProxSteps, optimizer ):
    
        
#     #-------------------------------------------------------------------------------------
#     ''' 1. Forward pass (proximal gradient method for nProxSteps) '''
#     #-------------------------------------------------------------------------------------
#     convergedState, proxEvol = evolveProx( model, currStim, epochNum, currState, refState, nProxSteps )
    
#     # trainingData[ 'convergedStates' ][ :, epochNum ] = convergedState.reshape( N, )
#     # currState = convergedState
    
#     # startTime = (epochNum-1) * nProxSteps + 1
#     # trainingData[ 'state' ][  :,  startTime : (startTime + nProxSteps)  ] = proxEvol
    
    
#     #-------------------------------------------------------------------------------------
#     ''' 2. Update connections '''
#     #-------------------------------------------------------------------------------------
#     currState = convergedState.detach()                                                  # remove the grad assoc. with currState
    
#     if simOptions['Kafashan']:     
#         model, connectionVars = model.KafashanLearning( currStim, currState, refState, connectionVars )
        
#     else:
#         loss = model.lossFunc( currStim, currState, refState )      ## 2a. Calculate loss
        
#         optimizer.zero_grad()                                               ## 2b. Set gradient back to 0
#         loss.backward( )                                                    ## 2c. Backpropgate the error 
#         optimizer.step()                                                    ## 2d. Optimize
    
        
        
    
    
#     return model, proxEvol, convergedState






#=========================================================================================
#=========================================================================================
##%% 
#=========================================================================================
#=========================================================================================





# def trainModel( model, trainingInput, nEpochs, optimizer, nProxSteps=100, simOptions=None ): 
def trainModel( model, trainingInput, simOptions=None, optimizer=None, 
                                       safetySaving=False, loadSafetySave=False,
                                       ): 
    '''   '''


    #=====================================================================================
    ''' Get information to use '''
    #=====================================================================================
    if simOptions is None:
        
        nEpochs = trainingInput.nBetas
        
        epochNumsToTest = [ 0, 200, 600, 5000, nEpochs ] 
        epochNumsToTest =  sorted(   list(  set([ i for i in epochNumsToTest if i<=nEpochs ])  )   )

        simOptions = {  'loadPrevData' : False,
                        'endWithZeroStim' : False,  
                        'storeAllConnMats' : False, 
                        'learnSingleStimUntilConvergence' : False, 
                        'Kafashan' : False, 
                        'epochNumsToTest' : epochNumsToTest,
                        'useQ2Also' : False,
                        'includeDelayInTraining' : True, 
                      }
        
    else: 
        epochNumsToTest = simOptions[ 'epochNumsToTest' ]
       
        
       
        
    try:
        trainBeforeDecay = simOptions[ 'trainBeforeDecay' ]
        trainAfterDecay = simOptions[ 'trainAfterDecay' ]
    except:
        trainBeforeDecay = True
        trainBeforeDecay = False
        
        
        
        
    if simOptions['learnSingleStimUntilConvergence']:
        # maxIter = simOptions['parameters']['maxIter']
        maxIter = simOptions['maxIter']
    else:
        maxIter = 1
        
      
        
    useQ2Also = simOptions[ 'useQ2Also' ]
    includeDelay = simOptions[ 'includeDelayInTraining' ]
        
    
    if (optimizer is None):
        if 'optimizer' in simOptions.keys(): 
            optimizer = simOptions[ 'optimizer' ]
        else:
            raise Exception( '[trainModel] Requires optimizer (given explicitly or via simOptions). ' )
        
    
    
    ''' Useful parameters '''
    #-------------------------------------------------------------------------------------
    nEpochs = simOptions[ 'parameters' ][ 'nEpochs' ]
    # nEncodSteps = simOptions[ 'parameters' ][ 'nEncodSteps_training' ]
    nEncodSteps = simOptions[ 'nEncodSteps' ]
    
    #----------------------------------------------------------
    if includeDelay:
        # nRetenSteps = simOptions[ 'nRetenSteps' ]
        nRetenSteps = simOptions[ 'nRetenSteps' ]
        nTotalEvolSteps = nEncodSteps + nRetenSteps
        
        if (nRetenSteps == 0):
            includeDelay = False
    else:
        nTotalEvolSteps = nEncodSteps
        
    nTimes = nTrainingTimes = (nTotalEvolSteps * nEpochs) + 1     ## number of time steps 
    #----------------------------------------------------------
    
    N = model.networkDim
    d = model.signalDim
    
    
    
    
    
    
    #=====================================================================================
    ''' STORAGE Initialization '''
    #=====================================================================================
    trainingData = initTrainingData( model, simOptions, includeDelay=includeDelay )
    
    
    
    ''' For plotting training performance '''
    #-------------------------------------------------------------------------------------
    z = 10
    frontAndBackEndEpochNums = list(range(0,z)) + list(range(nEpochs-z,nEpochs)) 
    frontAndBackEndEpochNums = list( set(frontAndBackEndEpochNums) )
         
    trainingData[ 'frontAndBackEnd' ] = {  'D' : torch.zeros(  [ 2*z, d, N ]  ),
                                           'H' : torch.zeros(  [ 2*z, N, N ]  )   }
    
    trainingData[ 'frontAndBackEnd' ][ 'D' ][ 0, :, : ] = model.D
    trainingData[ 'frontAndBackEnd' ][ 'H' ][ 0, :, : ] = model.H




    #=====================================================================================
    ''' Initialize variables '''
    #=====================================================================================
    currState = trainingData[ 'state' ][ :, 0 ].reshape(  N, 1  )  
    prevState = currState.clone()
    
    
    refState = currState.clone()
    if useQ2Also:
        refState2 = refState.clone()
    else: 
        refState2 = None

    
    #-----------------------------------------

    zeroStim = torch.zeros( [ d, 1 ] )
    zeroState = torch.zeros( [ N, 1 ] )



    #=====================================================================================
    ''' TRAIN '''
    #=====================================================================================    
    model.train()                           # Set to train 
    model.to( simOptions['device'] )


    
    
    #-------------------------------------------------
    ''' Load in data is need be '''
    #-------------------------------------------------
    startEpochNum = 1
    
    # if loadSafetySave:
        
    #     varDict = loadMidTrainData( simOptions )
        
    #     trainingInput = varDict[ 'trainingInput' ]
    #     model = varDict[ 'trainingModel' ]
    #     trainingData = varDict[ 'trainingData' ]
        
    #     mostRecentEpoch = list(  trainingData.keys()  )[0]
        
    #     startEpochNum = 
        




    
    for epochNum in range(startEpochNum,nEpochs+1):         # extra idx to account for naive model
    # for epochNum in range(1,nEpochs+1):         # extra idx to account for naive model
    # for epochNum in range(1,nEpochs):         # extra idx to account for naive model
        
    
        if (epochNum % 100) == 0:
            print( 'EPOCH {}'.format( epochNum ) )
    
    
        
        #---------------------------------------------------------------------------------
        ''' 0. Stimulus for this training epoch ''' 
        #---------------------------------------------------------------------------------
        currStim = trainingInput.getStim( epochNum-1 ).reshape( d,1 )
        
        # if addNoise:
        #     currStim = currStim + stimNoise[ :, stimInd-1 ]
        
        currStim.to( simOptions['device'] ) 
        
        
        #---------------------------------------------------------------------------------
        ''' 1. Evolve proximal dynamics and learn connections '''
        #---------------------------------------------------------------------------------
        if simOptions[ 'trainMatsDirectly' ]:
            prevMatDict = { 'prevD' : model.D.clone(),
                            'prevH' : model.H.clone(),
                            }
        else: 
            prevMatDict = { 'prevW' : model.W.clone(),
                            'prevMs' : model.Ms.clone(),
                            'prevMd' : model.Md.clone(),
                            }
        
        # refState = trainingData[ 'convergedStates' ][ :, epochNum-1 ].reshape( N,1 )         # indexing is ahead by 1 
       
        
       
        #=================================================================================
        for i in range( maxIter ):
            
            
            #-----------------------------------------------------------------------------
            ''' 1aa. Forward pass - ENCODING '''
            #-----------------------------------------------------------------------------
            convergedState, proxEvol = evolveProx( model, currStim, currState, refState, nEncodSteps,
                                                                      prevState=prevState, useQ2Also=useQ2Also, refState2=refState2 )
            # prevState = proxEvol[ :, -2 ].detach().reshape( N, 1 )
            # prevState = proxEvol[ :, -2 ].clone().reshape( N, 1 )



            if trainBeforeDecay:
                
                
                ''' 1ab. Update connections '''
                #--------------------------------------------            
                if simOptions['Kafashan']:     
                    r = convergedState.clone()                 ## current state r(t) 
                    model, connectionVars = model.KafashanLearning( currStim, r, refState, connectionVars )
                    
                else:
                    # print( epochNum )
    
                    r = convergedState.clone()                 ## current state r(t) 
                    loss = model.lossFunc( currStim, r, refState, r_tm2=refState2 )      ## 2a. Calculate loss
                    
                    optimizer.zero_grad()                                               ## 2b. Set gradient back to 0
                    loss.backward( )                                                    ## 2c. Backpropgate the error 
                    # loss.backward( retain_graph=True )                                                    ## 2c. Backpropgate the error 
                    optimizer.step()                                                    ## 2d. Optimize
                
                
                ''' 1ac. Update connection mats '''
                #--------------------------------------------
                if simOptions['trainMatsDirectly']:
                    [ W, Ms, Md ] = model.computeConnectionMats( )
                    model.W = W
                    model.Md = Md
                    model.Ms = Ms
    


            #-----------------------------------------------------------------------------
            ''' 1ba. Forward pass - RETENTION '''
            #-----------------------------------------------------------------------------
            if includeDelay:        
                
                
                ''' 2b. Reference state ''' 
                #---------------------------------------------
                if simOptions['takeAwayRef']:
                    refState = zeroState
                    
                
                convergedState, proxEvol2 = evolveProx( model, zeroStim, convergedState.clone(), refState, nRetenSteps, 
                                                                       prevState=prevState, useQ2Also=useQ2Also, refState2=refState2 )
                
    
            
            if trainAfterDecay:
            
                
                ''' 1ab. Update connections '''
                #--------------------------------------------            
                if simOptions['Kafashan']:     
                    model, connectionVars = model.KafashanLearning( currStim, currState, refState, connectionVars )
                    
                else:
                    # print( epochNum )
    
                    r = convergedState.clone()                 ## current state r(t) 
                    loss = model.lossFunc( currStim, r, refState, r_tm2=refState2 )      ## 2a. Calculate loss
                    
                    optimizer.zero_grad()                                               ## 2b. Set gradient back to 0
                    loss.backward( )                                                    ## 2c. Backpropgate the error 
                    # loss.backward( retain_graph=True )                                                    ## 2c. Backpropgate the error 
                    optimizer.step()                                                    ## 2d. Optimize
                
                
                
                ''' 1ac. Update connection mats '''
                #--------------------------------------------
                if simOptions['trainMatsDirectly']:
                    [ W, Ms, Md ] = model.computeConnectionMats( )
                    model.W = W
                    model.Md = Md
                    model.Ms = Ms
    
            





            
            ''' 1d. Check for convergence '''
            #--------------------------------------------
            convergenceBool, prevMatDict = checkMatConvergence( model, prevMatDict, simOptions['trainMatsDirectly'] )
            
            if convergenceBool: 
                break
        #=================================================================================
        
        
        
        
        
        
        
        #---------------------------------------------------------------------------------
        ''' 2. Update variables for next iteration/epoch '''
        #---------------------------------------------------------------------------------
        # currState = convergedState.detach().clone()         # remove the grad assoc. with currState  
        # refState = convergedState.detach().clone()
        
        # prevState = currState.clone()         # remove the grad assoc. with currState  
        currState = convergedState.clone()         # remove the grad assoc. with currState  
        
        
        if useQ2Also:
            refState2 = refState.clone()
        refState = convergedState.clone()
        
        
        
        #-------------------------------------------------------------------------------------
        ''' 3. Save parameters '''
        #------------------------------------------------------------------------------------- 
        trainingData[ 'convergedStates' ][ :, epochNum ] = convergedState.clone().reshape( N, )
        
        start1 = (epochNum-1) * nEncodSteps + 1
        end1 = start1 + nEncodSteps
        trainingData[ 'state' ][  :,  start1:end1  ] = proxEvol
        
        if includeDelay:
            start2 = end1
            end2 = start2 + nRetenSteps
            trainingData[ 'state' ][  :,  start2:end2  ] = proxEvol2

        
        
        
        
        #-----------------------
        if epochNum != nEpochs:
            trainingData[ 'matNorms' ][ :, epochNum ] = torch.tensor(  computeMatNorms(model)  )
        #-----------------------
        

        if simOptions['storeAllConnMats']:
            storageInd = epochNum
            
        else: 
            
            #--------------------------------
            if (epochNum in frontAndBackEndEpochNums):                
                ind = frontAndBackEndEpochNums.index( epochNum ) 
                trainingData[ 'frontAndBackEnd' ][ 'D' ][ ind, :, : ] = model.D
                trainingData[ 'frontAndBackEnd' ][ 'H' ][ ind, :, : ] = model.H
            #--------------------------------      
            
            if (epochNum in epochNumsToTest):
                storageInd = epochNumsToTest.index( epochNum ) 
            else: 
                continue 
                
        
        
        if simOptions[ 'trainMatsDirectly' ]:
            trainingData[ 'D' ][ storageInd, :, : ] = model.D
            trainingData[ 'H' ][ storageInd, :, : ] = model.H
        else: 
            trainingData[ 'W' ][ storageInd, :, : ] = model.W
            trainingData[ 'Ms' ][ storageInd, :, : ] = model.Ms 
            trainingData[ 'Md' ][ storageInd, :, : ] = model.Md 
            
           
            
        
        #=================================================================================
        ''' Safety save (in case of memory crash) '''
        #================================================================================= 
        if safetySaving and (epochNum in epochNumsToTest):
            
            varDict = { 'trainingModel' : model,  
                        'trainingInput' : trainingInput,  
                        'trainingData' : trainingData,  
                       }
            
            saveCurrentTrainingInfo( varDict )    
            

    
    
    return model, trainingData











# def testModel( model, testingInput, optimizer, simOptions ): 
def testModel( model, testingInput, simOptions, addNoise=False ): 
    '''   '''
    
    
    
    #=====================================================================================
    ''' Get information to use '''
    #=====================================================================================
    nTestBetas = testingInput.nBetas
    model.to( simOptions['device'] )
    
    nStims = testingInput.nBetas
    
    #-----------------------------------------
    nEpochs = simOptions[ 'parameters' ][ 'nEpochs' ]
    
    # nEncodSteps = simOptions[ 'nEncodSteps' ] 
    nEncodSteps = simOptions[ 'nEncodSteps_testing' ] 
    # nEncodSteps = simOptions[ 'parameters' ][ 'nEncodSteps_testing' ] 
    # nRetenSteps = simOptions[ 'parameters' ][ 'nRetenSteps' ]
    # nRetenSteps = simOptions[ 'nRetenSteps' ]
    nRetenSteps = simOptions[ 'nRetenSteps_ext' ]
    nTotalEvolSteps = nEncodSteps + nRetenSteps
    
    nTimes = nTestingTimes = (nTotalEvolSteps * nStims) + 1     ## number of time steps 
    #-----------------------------------------
    
    
    N = model.networkDim
    d = model.signalDim    
    
    
    
    if addNoise:
        stimNoise = testingInput.stimNoise
    
    
    
    useQ2Also = simOptions[ 'useQ2Also' ]

    
    
    
    
    #=====================================================================================
    ''' STORAGE Initialization '''
    #=====================================================================================
    stateEvol = torch.zeros( [N, nTimes] )                      ## 
    
    gradEvol = torch.zeros( [N, nTimes] )                       ## 
    convergedStates = torch.zeros( [N, nTestBetas+1] )          ## 
    
    
    #---------------------------------------------------------
    
    initState = simOptions[ 'parameters' ][ 'initState' ]  
    # initState_rep = initState.repeat( nEpochNumsToTest, 1, 1 )
    
    stateEvol[ :, 0 ] = initState.reshape(  stateEvol[ :, 0 ].shape  )
    convergedStates[ :, 0 ] = initState.reshape(  convergedStates[ :, 0 ].shape  ) 
    
    #---------------------------------------------------------
    
    
    modelData = {   'state' : stateEvol,
                    'convergedStates_encoding' : convergedStates,           # end of encoding phase
                    'convergedStates_retention' : convergedStates.clone(),  # end of retention phase
                 }
    
    
    #=====================================================================================
    ''' Initialize variables '''
    #=====================================================================================
    # currState = initState.clone()
    currState = modelData[ 'state' ][ :, 0 ].reshape(  N, 1  )  
    prevState = currState.clone()

    refState = currState.clone()
    
    if useQ2Also:
        refState2 = refState.clone()
    else: 
        refState2 = None


    #-----------------------------------------

    
    zeroStim = torch.zeros( [ d, 1 ] )
    zeroState = torch.zeros( [ N, 1 ] )
    
    

        
    #=====================================================================================
    ''' TEST '''
    #=====================================================================================

    for stimInd in range( 1, nTestBetas+1 ):
    
        
        # if (stimInd % 100) == 0:
        if (stimInd % 200) == 0:
            print( 'StimInd {}'.format( stimInd ) )    
    
        
        #---------------------------------------------------------------------------------
        ''' 1. ENCODING Phase '''
        #---------------------------------------------------------------------------------


        ''' 1a. Current stimulus ''' 
        #---------------------------------------------
        currStim = testingInput.getStim( stimInd-1 ).reshape( d,1 )
        
        if addNoise:
            currStim = currStim + stimNoise[ :, stimInd-1 ]
        
        currStim.to( simOptions['device'] ) 
        
        
        
        ''' 1b. Evolve using proximal ''' 
        #---------------------------------------------
        convergedState, proxEvol = evolveProx( model, currStim, currState, refState, nEncodSteps, 
                                                              prevState=prevState, useQ2Also=useQ2Also, refState2=refState2 )
        
        currState = convergedState.clone()
        prevState = proxEvol[ :, -2 ].clone().reshape( N, 1 )

        
        
        ''' 1c. Store results ''' 
        #---------------------------------------------
        modelData[ 'convergedStates_encoding' ][ :, stimInd ] = convergedState.reshape( N, )
        
        start = (stimInd-1) * nTotalEvolSteps + 1               ## +1 since we start with an initState
        end = start + nEncodSteps
        modelData[ 'state' ][ :, start:end ] = proxEvol 
        
        
        
        
        #---------------------------------------------------------------------------------
        ''' 2. RETENTION Phase '''
        #---------------------------------------------------------------------------------
        
        if nRetenSteps > 0:
            
            ''' 2a. Current stimulus ''' 
            #---------------------------------------------
            if simOptions['takeAwayStim']:
                currStim = zeroStim
            
            
            ''' 2b. Reference state ''' 
            #---------------------------------------------
            if simOptions['takeAwayRef']:
                refState = zeroState
            
            
            ''' 2c. Evolve using proximal ''' 
            #---------------------------------------------
            # convergedState2, proxEvol2 = evolveProx( model, currStim, convergedState, refState, nRetenSteps )
            convergedState2, proxEvol2 = evolveProx( model, currStim, currState, refState, nRetenSteps, 
                                                                    useQ2Also=useQ2Also, refState2=refState2 )
            
            currState = convergedState2.clone()
            prevState = proxEvol2[ :, -2 ].clone().reshape( N, 1 )

            
            
            ''' 2d. Store results ''' 
            #---------------------------------------------
            # modelData[ 'convergedStates_retention' ][ :, stimInd ] = convergedState2.reshape( N, )
            modelData[ 'convergedStates_retention' ][ :, stimInd ] = currState.reshape( N, )
            
            start2 = end
            end2 = start2 + nRetenSteps
            modelData[ 'state' ][ :, start2:end2 ] = proxEvol2 
        

        #---------------------------------------------------------------------------------
        ''' 3. Update for the next stimulus '''
        #---------------------------------------------------------------------------------
        if useQ2Also:
            refState2 = refState.clone()
        
        refState = currState.clone()    ## Will be state at the end of the encoding (nRetenSteps=0) or the retention phase (nRetenSteps>0)
        
    
    return model, modelData
    



    
    
    
def initTrainingData( model, simOptions, includeDelay=True ):
    '''  ''' 
    
    #-------------------------------------------------------------------------------------
    nEpochs = simOptions[ 'parameters' ][ 'nEpochs' ]
    # nEncodSteps = simOptions[ 'parameters' ][ 'nEncodSteps_training' ]
    nEncodSteps = simOptions[ 'nEncodSteps' ]
    
    
    if includeDelay:
        # nRetenSteps = simOptions[ 'parameters' ][ 'nRetenSteps' ]
        nRetenSteps = simOptions[ 'nRetenSteps' ]
        nTotalEvolSteps = nEncodSteps + nRetenSteps
        
        nTimes = (nTotalEvolSteps * nEpochs) + 1     ## number of time steps 
        
    else: 
        nTimes = nTrainingTimes = (nEncodSteps * nEpochs) + 1     ## number of time steps 
    
    
    
    N = model.networkDim
    d = model.signalDim
    #-------------------------------------------------------------------------------------
    
    
    trainingData = { }
    
    
    #-------------------------------------------------------------------------------------    
    epochNumsToTest = simOptions['epochNumsToTest']
    if 0 not in epochNumsToTest:    
        epochNumsToTest = [ 0 ] + epochNumsToTest
    nEpochNumsToTest = len( epochNumsToTest )
    
    
    if simOptions['storeAllConnMats']:
        nMatsToSave = nEpochs + 1 
    else: 
        nMatsToSave = nEpochNumsToTest
    #-------------------------------------------------------------------------------------
    
    
    #-------------------------------------------------------------------------------------
    stateEvol = torch.zeros( [N, nTimes] )                          ## 
    convergedStates = torch.zeros( [N, nEpochs+1] )          ## 
    
    initState = simOptions[ 'parameters' ][ 'initState' ]    
    # currState = initState
    
    stateEvol[ :, 0 ] = initState.reshape(  stateEvol[ :, 0 ].shape  )
    convergedStates[ :, 0 ] = initState.reshape(  convergedStates[ :, 0 ].shape  )    
    
    
    trainingData[ 'state' ] = stateEvol
    trainingData[ 'convergedStates' ] = convergedStates
    #-------------------------------------------------------------------------------------
    
    
    
    #-------------------------------------------------------------------------------------
    if simOptions[ 'trainMatsDirectly' ]:
        
        DMats = torch.zeros( [nMatsToSave, d, N] )                        ## 
        HMats = torch.zeros( [nMatsToSave, N, N] )                        ## 
        
        # DMats[ 0, :, : ] = model.D
        # HMats[ 0, :, : ] = model.H
        DMats[ 0, :, : ] = model.D.clone()
        HMats[ 0, :, : ] = model.H.clone()
        
        trainingData[ 'D' ] = DMats
        trainingData[ 'H' ] = HMats
        
        trainingData[ 'matNorms' ] = torch.zeros( [2, nEpochs] ) 
        
    else: 
        
        WMats = torch.zeros( [nMatsToSave, N, d] )                        ## 
        MdMats = torch.zeros( [nMatsToSave, N, N] )                       ## 
        MsMats = torch.zeros( [nMatsToSave, N, N] )                       ## 
        
        WMats[ 0, :, : ] = model.W 
        MdMats[ 0, :, : ] = model.Md
        MsMats[ 0, :, : ] = model.Ms
        
        trainingData[ 'W' ] = WMats
        trainingData[ 'Md' ] = MdMats
        trainingData[ 'Ms' ] = MsMats
        
        trainingData[ 'matNorms' ] = torch.zeros( [3, nEpochs] ) 
    
    
    trainingData[ 'matNorms' ][ :, 0 ] = torch.tensor(  computeMatNorms(model)  )
    #-------------------------------------------------------------------------------------
    


    
    
    return trainingData 

    








def checkMatConvergence( model, prevMatDict, trainMatsDirectly ):
    
    
    #-------------------------------------------------------------------------------------
    ''' Check if current model mats are close enough to saved previous mats '''
    #-------------------------------------------------------------------------------------
    if trainMatsDirectly:
        
        testD = torch.allclose( prevMatDict['prevD'], model.D )
        testH = torch.allclose( prevMatDict['prevH'], model.H )
        
        boolVector = [ testD, testH ]
        
    else: 
        
        testW = torch.allclose( prevMatDict['prevW'], model.W )
        testMs = torch.allclose( prevMatDict['prevMs'], model.Ms )
        testMd = torch.allclose( prevMatDict['prevMd'], model.Md )
        
        boolVector = [ testW, testMs, testMd ]
        
        

    

    #-------------------------------------------------------------------------------------
    ''' Update the stored mats '''
    #-------------------------------------------------------------------------------------    
    if trainMatsDirectly:
        prevMatDict['prevD'] = model.D.clone()
        prevMatDict['prevH'] = model.H.clone()
        # prevMatDict = { 'prevD' : model.D.clone(),
        #                 'prevH' : model.H.clone(),
        #                 }
    else: 
        prevMatDict['prevW'] = model.W.clone()
        prevMatDict['prevMs'] = model.Ms.clone()
        prevMatDict['prevMd'] = model.Md.clone()
        # prevMatDict = { 'prevW' : model.W.clone(),
        #                 'prevMs' : model.Ms.clone(),
        #                 'prevMd' : model.Md.clone(),
        #                 }


    #-------------------------------------------------------------------------------------
    ''' Return results '''
    #-------------------------------------------------------------------------------------
    if np.all( boolVector ):
        return True, prevMatDict
    else:
        return False, prevMatDict
    
    
    
    
     
def computeMatNorms( model, order='fro' ):
    
    
    normList = [ ]
    
    
    # matNames = model.parameterNames
    # for matName in matNames:
        # mat = model.matName
        
        
    for mat in model.parameters():
        matNorm = torch.linalg.norm( mat, ord=order )
        normList.append( matNorm )
    
    
    return normList


    
    # D, H = [  model.D,  model.H  ]
    
    # DNorm = torch.linalg.norm( D, p=2 )
    # HNorm = torch.linalg.norm( H, p=2 )
    
    # return DNorm, HNorm
    # 


    
# def computeConnMatNorms( model ):
    
#     W, Ms, Md = 
    
#     WNorm = torch.linalg.norm( W, p=2 )
#     MsNorm = torch.linalg.norm( Ms, p=2 )
#     MdNorm = torch.linalg.norm( Md, p=2 )
    
    
#     return WNorm, MsNorm, MdNorm
    
    



#=========================================================================================
#=========================================================================================
#%% 
#=========================================================================================
#=========================================================================================

def getEpochNumsToTest( nEpochs, maxColors=9 ):
    '''  ''' 
    
    epochNumsToTest = [ 0, 1, 10, 100 ]
    epochNumsToTest = [ 0, 50, 100, 200, 400 ]
    epochNumsToTest = [ 0, 250, 500, 1000, 2000, 3000, 4000, 5000, 10000 ]
    epochNumsToTest = [ 0, 500, 1000, 2000, 3000, 4000, 5000, 10000 ]
    # epochNumsToTest = [ 0, 10, 20, 80, nEpochs ]

    


    if nEpochs <= 10000:
        epochNumsToTest = [ 0, 2000, 4000, 6000, 8000, 10000 ] 
    
    
    if nEpochs <= 8000:
        epochNumsToTest = [ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000 ] 
        
        
    if nEpochs < 5000:
        epochNumsToTest = [ 0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000 ] 


    if nEpochs <= 1000:
        epochNumsToTest = [ 0, 100, 200, 300, 400, 500, 750, 1000 ] 

            
        if nEpochs <= 700:
            epochNumsToTest = list( range(0, nEpochs+100, 100) )

            if nEpochs <= 300:
                epochNumsToTest = list( range(0, nEpochs+50, 50) )
                
                if nEpochs <= 100:
                    epochNumsToTest = list( range(0, nEpochs+25, 25) )
                    
                    
                    if nEpochs <= 25:
                        epochNumsToTest = list( range(0, nEpochs+5, 5) )
                    
                        if nEpochs <= 10:
                            epochNumsToTest = list( range(0, nEpochs+2, 2) )
                




    epochNumsToTest = [ i for i in epochNumsToTest if i<=nEpochs ]
    epochNumsToTest = sorted(   list(  set(epochNumsToTest)  )   )

    nEpochNumsToTest = len( epochNumsToTest )

    
    
    return epochNumsToTest 
    
    
    
    
    
#=========================================================================================
#=========================================================================================
#%% Print Info
#=========================================================================================
#=========================================================================================





def printTimeElapsed( start, end ):
    '''  '''
    
    
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print('\tTime elapsed:  ( h, m, s ) = ( {}, {}, {} )'.format( round(hours), round(minutes), round(seconds,2) ))
    
    # if hours > 0:
    #     print('\tHours:', hours )
    # if minutes > 0:
    #     print('\tMinutes:', minutes )
    # print('\tSeconds:', round(seconds,4) )
    # # print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    
    return hours, minutes, seconds






def printActualTime( ):
    '''  '''
    
    
    ''' Format the time '''
    #-----------------------------------------------------------
    # realTime = datetime.now()
    
    # minuteStr = str(realTime.minute)
    # if len(minuteStr) == 1:
    #     minuteStr = '0' + minuteStr
    
    # realTimeReadable = str(realTime.hour) + ':' + minuteStr
    
    [ realTime, realTimeReadable ] = getCurrTime( )
    
    
    
    ''' Print '''
    #-----------------------------------------------------------
    # print( '\tActual Time: ', realTimeReadable )
    print( 'Actual Time: ', realTimeReadable )
    
    
    return realTime.hour, realTime.minute







def getCurrTime(  ):
    
    
    realTime = datetime.now()
    
    minuteStr = str(realTime.minute)
    if len(minuteStr) == 1:
        minuteStr = '0' + minuteStr
    
    realTimeReadable = str(realTime.hour) + ':' + minuteStr
    
    
    return realTime, realTimeReadable




def sendEmailWithStatus( subject='', body='', emailAddress=None, emailPW=None ):
    ## https://stackoverflow.com/questions/26981591/how-to-send-myself-an-email-notification-when-a-suds-job-is-done
    
    
    #---------------------------------------------------
    ''' Get email address, etc. '''
    #---------------------------------------------------
    if emailAddress is None:
        emailAddress = 'bethannacode@gmail.com'
        emailPW = 'uqbb rxzr nbks wbzh'                     ## App Password
    
    
    mail = smtplib.SMTP( 'smtp.gmail.com', 587 )
    
    
    #---------------------------------------------------
    ''' Set up the email '''
    #---------------------------------------------------
    mail.ehlo()
    mail.starttls()
    
    
    [ realTime, realTimeReadable ] = getCurrTime( ) 
    body = body + '\n\nTime: {}'.format( realTimeReadable )
    
    
    message = 'Subject: {}\n\n{}'.format( subject, body )
    
    
    #---------------------------------------------------
    ''' Send '''
    #---------------------------------------------------
    mail.login( emailAddress, emailPW )
    mail.sendmail( emailAddress, emailAddress, message )   ## sender, destination, content
    mail.close()
    
    
    
    print( 'Email sent to: ', emailAddress )


    return 



# sendEmailWithStatus(  contentText='Finished running code',  emailAddress='bethannacode@gmail.com',  emailPW='uqbb rxzr nbks wbzh'  )









def printProgressBar( percentage, nBins=40 ):
    
    percentFloor = int( np.floor(percentage*nBins) )
    percentRemaining = nBins - percentFloor
    sys.stdout.write( '\r' )
    sys.stdout.write(  "\t[{}{}] \t {}%".format(
                                            '='*percentFloor, 
                                            ' '*percentRemaining, 
                                            int(np.round(100*percentage,0)))  
                                            )
    sys.stdout.flush( )
    
    return




# def printProgressBar( percentage ):
#     ''' 
#     https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
#     '''
    
#     raise Exception( '[printProgressBar] not yet defined' )
    
    
#     return 






def printSimOptions( simOptions ):
    
    print(  '=========================' )
    print(  'Simulation Options:' )
    print(  '=========================' )

    print( '\t loadPrevData: ', simOptions['loadPrevData'] )   
    print( '\t endWithZeroStim: ', simOptions['endWithZeroStim'] )   
    print( '\t storeAllConnMats: ', simOptions['storeAllConnMats'] )   
    print( '\t learnSingleStimUntilConvergence: ', simOptions['learnSingleStimUntilConvergence'] )    
    print( '\t Kafashan: ', simOptions['Kafashan'] )     
    print( '\t trainMatsDirectly: ', simOptions['trainMatsDirectly'] )     
    print( '\t device: ', simOptions['device'] )    
    print( )
    print( '\t epochNumsToTest: ', simOptions['epochNumsToTest'] )  
    
    if torch.all( initState == 0 ):
        print( '\t initState:  zeroVector' )    
    else:
        print( '\t initState: ', simOptions['initState'].T )    
    
    return 




def printSimOptions2( simOptions, printBasic=False ):
    
    print(  '=======================================' )
    print(  'Simulation Options:' )
    print(  '=======================================' )
    #-------------------------------------------------------------------------------------
    if printBasic:
        print( 'Basic:' )
        print( '\t loadPrevData: ', simOptions['loadPrevData'] )   
        print( '\t endWithZeroStim: ', simOptions['endWithZeroStim'] )   
        # print( '\t storeAllConnMats: ', simOptions['storeAllConnMats'] )   
        # print( '\t requireStable: ', simOptions['requireStable'] )   
        # print( '\t processingTermWithD: ', simOptions['processingTermWithD'] )   
        print( '\t device: ', simOptions['device'] )   
        
        print( )

    #-------------------------------------------------------------------------------------
    
    # print( )
    
    #-------------------------------------------------------------------------------------
    # print( 'Training:' )
    # print( '\t learnSingleStimUntilConvergence: ', simOptions['learnSingleStimUntilConvergence'] )    
    # print( '\t Kafashan: ', simOptions['Kafashan'] )     
    # print( '\t trainMatsDirectly: ', simOptions['trainMatsDirectly'] )     
    #-------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------
    print( 'Parameters:' )
    print( '\t Weight combo(s): ', simOptions[ 'weightCombos' ] )  
    print( '\t Attention (e,r):  ({},{}) '.format( simOptions['encodingAlpha'], simOptions['retentionAlpha'])    )
    #-------------------------------------------------------------------------------------
    
    print( )

         
    #-------------------------------------------------------------------------------------
    # print( 'Training:' )
    # print( '\t nEpochs: ', simOptions['parameters']['nEpochs'] )  
    print( 'Training Epochs: ', simOptions['parameters']['nEpochs'] )  
    print( '\t epochNumsToTest: ', simOptions['epochNumsToTest'] )  
    print( '\t trainBeforeDecay: ', simOptions['trainBeforeDecay'] )  
    print( '\t trainAfterDecay: ', simOptions['trainAfterDecay'] )  
    # print( '\t maxIter: ', simOptions['parameters']['maxIter'] )  
    print( '\t maxIter: ', simOptions['maxIter'] )  
    # print( '\t learningRate: ', simOptions['parameters']['learningRate'] )  
    #-------------------------------------------------------------------------------------
     
    print( )
    
    #-------------------------------------------------------------------------------------
    
    print( 'Time steps: (training, testing) ' )
    # print( '\t nEncodSteps: ({},{})'.format( simOptions['parameters']['nEncodSteps_training'], simOptions['parameters']['nEncodSteps_testing'])   )  
    print( '\t nEncodSteps: ({},{})'.format( simOptions['nEncodSteps'], simOptions['nEncodSteps_testing'])   )  
    print( '\t nRetenSteps: ({},{})'.format( simOptions['nRetenSteps'], simOptions['nRetenSteps_ext'])  )  
    # print( '\t nEncodSteps_training: ', simOptions['parameters']['nEncodSteps_training'] )  
    # print( '\t nEncodSteps_testing: ', simOptions['parameters']['nEncodSteps_testing'] )  
    # print( '\t nRetenSteps_train: ', simOptions['nRetenSteps'] )  
    # print( '\t nRetenSteps_ext: ', simOptions['nRetenSteps_ext'] )  
    #-------------------------------------------------------------------------------------
    
    print( )
    
    #-------------------------------------------------------------------------------------
    initState = simOptions['parameters']['initState']
    if torch.all( initState == 0 ):
        print( 'initState:  zeroVector' )    
    else:
        # print( 'initState: ', initState.T )   
        print( 'initState: randVector' )   
    #-------------------------------------------------------------------------------------
    
    
    # #-------------------------------------------------------------------------------------
    # print( )
    # print( 'Attention:' )
    # print( '\t retention: ', simOptions['retentionAlpha']  )
    # print( '\t encoding: ', simOptions['encodingAlpha']  )
    # #-------------------------------------------------------------------------------------
    
    
    
    
    return 


#=========================================================================================
#=========================================================================================
#%% Continue training
#=========================================================================================
#=========================================================================================


def pickUpTraining( simOptions, trainingModel, trainingInput, trainingData, newMaxEpoch ):

    
    #-----------------------------------------------------------------------------------------
    ''' Backup old data '''
    #----------------------------------------------------------------------------------------- 
    saveFolder = nameSaveDir( simOptions )
    saveFolder = os.path.join( 'continuedTrainingBackup', saveFolder )  

    currDir = os.getcwd()
    saveDir = os.path.join( currDir, saveFolder ) 


    varNames = [ 'trainingModel', 'trainingInput', 'trainingData'  ]
    filenames = saveModelInfo( locals(), saveDir, varNames=varNames )


    
    #-----------------------------------------------------------------------------------------
    ''' Initialize trainingModel to last epoch '''
    #----------------------------------------------------------------------------------------- 
    oldMaxEpoch = trainingInput.nBetas
    
    trainingModel2 = copy.deepcopy( trainingModel ) 
    trainingModel2 = trainingModel2.setInitConnections(  D = trainingModel.D,  H = trainingModel  )
    
    
    

    
    #-----------------------------------------------------------------------------------------
    ''' Update simOptions '''
    #----------------------------------------------------------------------------------------- 
    simOptions[ 'parameters' ][ 'nEpochs' ] = nNewEpochs
    
    newEpochNumsToTest = getEpochNumsToTest( newMaxEpoch ) 
    epochNumsToTest2 = [  epochNum-oldMaxEpoch  for epochNum in newEpochNumsToTest  ]
    simOptions[ 'epochNumsToTest' ] = epochNumsToTest2
    
    simOptions['parameters'][ 'initState' ] = trainingData[ 'state' ][ :, -1 ]
    
    
    
    #-----------------------------------------------------------------------------------------
    ''' New trainingInput '''
    #----------------------------------------------------------------------------------------- 
    nNewEpochs = newMaxEpoch - oldMaxEpoch
    
    epochNumsToTest
    
    
    trainingInput_new = inputStimuli( d, nNewEpochs, simOptions )
    
    
    
    
    
    
    
    #-----------------------------------------------------------------------------------------
    ''' Run training '''
    #----------------------------------------------------------------------------------------- 
    
    print(  )
    print(  '=========================' )
    print(  'TRAINING (Continued)' )
    print(  '=========================' )
    print(  )
    
    
    start = time.time()
    [startH, startM] = printActualTime( )
    print( )
    
    
    [ trainingModel_new, trainingData_new ] = trainModel( trainingModel2, trainingInput_new, simOptions,
                                                 # includeDelay=simOptions['includeDelayInTraining'],  
                                                 )
    
    
    print( )
    printTimeElapsed(  start,  time.time()  )
    
    [endH, endM] = printActualTime( )
    print( )
    
    
    




    #-----------------------------------------------------------------------------------------
    ''' Combine old and new trainingData, etc.  '''
    #----------------------------------------------------------------------------------------- 
    trainingData_combined = copy.deepcopy( trainingData )


    z = int(   trainingData_combined[ 'frontAndBackEnd' ][ 'D' ].shape[0]/2   )
    trainingData_combined[ 'frontAndBackEnd' ][ 'D' ][ z:: ] = trainingData_new[ 'frontAndBackEnd' ][ 'D' ][ z:: ]
    trainingData_combined[ 'frontAndBackEnd' ][ 'H' ][ z:: ] = trainingData_new[ 'frontAndBackEnd' ][ 'H' ][ z:: ]


    trainingData_combined[ 'state' ] = torch.column_stack(  [ trainingData['state'], trainingData_new['state'][:,1::] ]  )
    trainingData_combined[ 'convergedStates' ] = torch.column_stack(  [ trainingData['convergedStates'], trainingData_new['convergedStates'][:,1::] ]  )
    
    # trainingData_combined[ 'D' ] = torch.column_stack(  [ trainingData['D'], trainingData_new['D'] ]  )
    # trainingData_combined[ 'H' ] = torch.column_stack(  [ trainingData['H'], trainingData_new['H'] ]  )
    trainingData_combined[ 'matNorms' ] = torch.column_stack(  [ trainingData['matNorms'], trainingData_new['matNorms'] ]  )
    
    # trainingData_combined[ 'nRetenSteps' ] = 
    
    nTimes_old = trainingData_new['state'].shape[1]
    trainingData_combined[ 'stimTimeInds' ] = [  (t + nTimes_old) for t in trainingData[ 'stimTimeInds' ] ]
    
    # trainingData_combined[ 'convergedTimeInds' ] = 
    # trainingData_combined[ 'convergedTimeIndsReten' ] = 



    # for epochNum in newEpochNumsToTest: 
        










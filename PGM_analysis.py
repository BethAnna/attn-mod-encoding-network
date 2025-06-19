#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:44:17 2024

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


import pickle


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
import matplotlib.pyplot as plt
import matplotlib as mpl


#-----------------------------
''' Saving '''
#-----------------------------
import pickle 

import smtplib





#=============================================================
''' Import my code '''
#=============================================================
from PGM_model import *
# from PGM_analysis import * 

# from PGM_saving import * 








#=========================================================================================
#=========================================================================================
#%% Analysis
#=========================================================================================
#=========================================================================================


# def encodingError( model, inputObject, modelData ):
#     ''' 
#         L2-norm of error between converged state (right before a new stimulus arrives) and 
#     the original stimulus input. 
#     '''
    
#     raise Exception( '[encodingError] This function is not yet finished.' )
    

    
#     return encErr







def splitBetaProxEvols( model, stateEvol, nTotalEvolSteps ):
    '''  ''' 



    ''' Spilt the stateEvol by beta '''
    #----------------------------------------------------------------
    encodingEvol = stateEvol[ :, 1:: ].detach()                         ## ignore initState
    nBetas = int( encodingEvol.shape[1] / nTotalEvolSteps )
    
    splitEvol = torch.split( encodingEvol.T, nTotalEvolSteps )       ## tuple of proxEvols for each beta 
    splitEvol = torch.tensor(np.array(  [ splitEvol[i].T for i in range(nBetas) ]  ))       ## ( nBetas, N, nTotalEvolSteps )
    
    
    ''' Decode the stateEvol '''
    #----------------------------------------------------------------
    if 'D' in model.parameterNames:
        D = model.D
    else:
        D = model.computeDandH()[0]         # ( d, N )   


    Drepeat = D.repeat( nBetas, 1, 1 )                          ## ( d, N )  -->  ( nBetas, d, N )
    decodedEvol = Drepeat @ splitEvol                           ## ( nBetas, d, nTotalEvolSteps )  <--  (nBetas,d,N) x (nBetas,N,nProxSteps)
    


    return splitEvol, decodedEvol







def proxEvolStepChange( model, stateEvol, nTotalEvolSteps ):
    
    
    splitState, splitDecoded = splitBetaProxEvols( model, stateEvol, nTotalEvolSteps )



    ''' Compute the step changes & their norms '''
    #----------------------------------------------------------------
    decodedDiffs = splitDecoded[ :, :, 1:: ]  -  splitDecoded[ :, :, 0:-1 ]     ## ( nBetas, d, nTotalEvolSteps-1 )
    decodedDiffNorms = torch.norm( decodedDiffs, dim=1 )                        ## ( nBetas, nTotalEvolSteps-1 )

    stateDiffs = splitState[ :, :, 1:: ]  -  splitState[ :, :, 0:-1 ]           ## ( nBetas, N, nTotalEvolSteps-1 )
    stateDiffNorms = torch.norm( stateDiffs, dim=1 )                            ## ( nBetas, nTotalEvolSteps-1 )
        


    return decodedDiffNorms, stateDiffNorms







def convergenceRate( model, inputObject, modelData, tol=0.001 ):
    '''  
        How quickly the state reaches "convergence," i.e., at which proximal step k the 
    following is satisfied: 
        ||  r(t^k) - r(t^{k+1})  ||_2    <   tol 
    for each stimulus x(t).
    '''

    #-------------------------------------------------------
    ''' State evolution '''
    #-------------------------------------------------------
    epochNum = model.epochNum
    
    if epochNum is None:    
        stateEvol = modelData[ 'state' ]
    else: 
        stateEvol = modelData[ epochNum ][ 'state' ]
        
        
    if hasattr( model, 'D' ):
        D = model.D
    else:
        D = model.computeDandH()[0]         # ( d,N )
        
        
    #-------------------------------------------------------
    ''' Input '''
    #-------------------------------------------------------
    convergedTimeInds = modelData[ 'convergedTimeInds' ]
    nBetas = len( convergedTimeInds )
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Analyze '''
    #-------------------------------------------------------------------------------------
    convergTimes = torch.zeros( nBetas,1 )
    
    nEncodSteps = modelData[ 'nEncodSteps' ]
    nRetenSteps = modelData[ 'nRetenSteps' ] 
    nTotalEvolSteps = nEncodSteps + nRetenSteps
    
    proxEvol = torch.zeros( nBetas, model.networkDim, nTotalEvolSteps )
    
    
    
    ''' The step changes during the prox evolution '''
    #----------------------------------------------------------------
    decodedDiffNorms, stateDiffNorms = proxEvolStepChange( model, stateEvol, nTotalEvolSteps )
    
    
    
    
    ''' Find convergence '''
    #----------------------------------------------------------------
    decodedConvergInds = [  torch.where( decodedDiffNorms[i] < tol )[0] for i in range(nBetas)  ]
    decodedConvergInds = [ inds[0].item() if inds.numel() != 0 else None  for inds in decodedConvergInds ] 
    

    stateConvergInds = [  torch.where( stateDiffNorms[i] < tol )[0] for i in range(nBetas)  ]
    stateConvergInds = [ inds[0].item() if inds.numel() != 0 else None  for inds in stateConvergInds ]
    

    return decodedConvergInds, stateConvergInds
    
    
    
    
    


def compareProxStepError( model, inputObject, modelData ):
    
    
    #-------------------------------------------------------
    ''' State evolution '''
    #-------------------------------------------------------
    if (not hasattr(model,'epochNum')) or (model.epochNum is None):
        raise Exception( '[compareProxStepError] This function is only coded up for testing models' )
        
    
    epochNum = model.epochNum
    stateEvol = modelData[ epochNum ][ 'state' ]    ## a trained model (after epochNum epochs)
        
    
    convergedTimeInds = modelData[ 'convergedTimeInds' ]
    nBetas = len( convergedTimeInds )
    
    
    nEncodSteps = modelData[ 'nEncodSteps' ] 
    nRetenSteps = modelData[ 'nRetenSteps' ] 
    nTotalEvolSteps = nEncodSteps + nRetenSteps
        
    
    
    #-------------------------------------------------------
    ''' Split state and decoded values by beta '''
    #-------------------------------------------------------
    # splitState, splitDecoded = splitBetaProxEvols( model, stateEvol, nProxSteps ) 
    splitState, splitDecoded = splitBetaProxEvols( model, stateEvol, nTotalEvolSteps ) 
    
    # print( stateEvol.shape )

    
    #-------------------------------------------------------------------------------------
    ''' Compare each proximal evolution step to original input '''
    #-------------------------------------------------------------------------------------
    encodingErrorNorms = torch.zeros( [nBetas, nTotalEvolSteps] )
    encodingPercentErrors = torch.zeros( [nBetas, nTotalEvolSteps] )
    
    
    for i in range(nBetas): 
        currBeta = inputObject.stimMat[ :, i ].reshape( [model.signalDim,1] )   # ( d, 1 )
        
        proxEvol = splitState[ i ]                                              # ( N, nTotalEvolSteps )
        decodedProxEvol = splitDecoded[ i ]                                     # ( d, nTotalEvolSteps )
    
        # print( currBeta.shape )
        currBetaRep = currBeta.repeat( 1, nTotalEvolSteps )
    
        # encodingError = decodedProxEvol - currBeta                      # ( d, nTotalEvolSteps )
        encodingError = decodedProxEvol - currBetaRep                      # ( d, nTotalEvolSteps )
        encodingErrorNorm = torch.linalg.norm( encodingError, dim=0 )   # ( 1, nTotalEvolSteps )
        encodingErrorNorms[ i ] = encodingErrorNorm

        
        encodingPercentError = encodingErrorNorm / torch.linalg.norm( currBeta )
        encodingPercentErrors[ i ] = encodingPercentError
        
        
    encodingErrorNorms = encodingErrorNorms.detach()
    encodingPercentErrors = encodingPercentErrors.detach()
        
    
    return encodingErrorNorms, encodingPercentErrors
    
    
    



# def compareProxSteps( model, inputObject, modelData ):

    
    

#     return 



# def plotConvergence( model, inputObject, modelData ):
    
    
#     return 




def analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=15, 
                                   compareToRefState=False ):


    
    # testingAnalysis = { 'recoveryPerf' : { },
    #                     'reconR' : { },
    #                     'reconX' : { },
                        
    #                     'encodingError' : { },
    #                     'convergenceRate' : { },
    #                     # 'reconX' : { },
    #                     # 'reconX' : { },
    #                   }
    
    
    
        
    recovPerfDict = { }
    recovPerfRetenDict = { }
    reconRDict = { }
    reconXDict = { }
    
    
    
    convergenceDict = { }
    proxEvolErrorDict = { }
    
    
    costTermsDict = { }
    
    
    # epochNumsToTest = simOptions[ 'epochNumsToTest' ]
    epochNumsToTest = list( epochModels.keys() )
    
    
    
    for epochNum in epochNumsToTest:
    # for epochNum in epochNumsToTest[0:2]:
        
        # print(  )
        # print( 'epochNum:', epochNum )
        
        
        epochModel = epochModels[ epochNum ]
        
        
        #-------------------------------------------------------------------------------------
        ''' Recovery Performance '''
        #-------------------------------------------------------------------------------------
        # if simOptions[ 'parameters' ][ 'nRetenSteps' ] > 0:
        if simOptions[ 'nRetenSteps' ] > 0:
            [ recovPerf, recovPerfReten, reconR, reconX ] = recoveryPerf( epochModel, testingInput, testingData, 
                                                                 forwardHorizon=FH, retenData=True, 
                                                                 compareToRefState=compareToRefState )
            recovPerfRetenDict[ epochNum ] = recovPerfReten
            
            
        else: 
            [ recovPerf, reconR, reconX ] = recoveryPerf( epochModel, testingInput, testingData, 
                                                                 forwardHorizon=FH, 
                                                                 compareToRefState=compareToRefState )
        
        
        recovPerfDict[ epochNum ] = recovPerf
        reconRDict[ epochNum ] = reconR
        reconXDict[ epochNum ] = reconX
        
        
        
        # #-------------------------------------------------------------------------------------
        # ''' Convergence '''
        # #-------------------------------------------------------------------------------------
        # # convergTimes, proxEvol = convergenceRate( epochModel, testingInput, testingData )
        
        # # convergenceDict[ epochNum ] = convergTimes
        # # proximalEvol[ epochNum ] = proxEvol
        
        # convergenceDict[ epochNum ] = {}
        
        # [decodedConvergInds, stateConvergInds] = convergenceRate( epochModel, testingInput, testingData )
        # convergenceDict[ epochNum ][ 'decoded' ] = decodedConvergInds
        # convergenceDict[ epochNum ][ 'state' ] = stateConvergInds
        
        
        proxEvolErrorDict[ epochNum ] = { }
        encodingErrorNorms, encodingPercentErrors = compareProxStepError( epochModel, testingInput, testingData )
        proxEvolErrorDict[ epochNum ][ 'absolute' ] = encodingErrorNorms 
        proxEvolErrorDict[ epochNum ][ 'percent' ] = encodingPercentErrors 
        
        
        # #-------------------------------------------------------------------------------------
        # ''' Cost Terms '''
        # #-------------------------------------------------------------------------------------
        # epoch_costs = costTerms_model( epochModel, testingInput, testingData )
        # costTermsDict[ epochNum ] = epoch_costs
        
        
        
        
    
    analysisDict = {    'recovPerfDict' : recovPerfDict,
                        'reconRDict' : reconRDict,
                        'reconXDict' : reconXDict,
                        
                        'proxEvolErrorDict' : proxEvolErrorDict,
                        
                        'compareToRefState' : compareToRefState,
                      }
        

    # if simOptions[ 'parameters' ][ 'nRetenSteps' ] > 0:
    if simOptions[ 'nRetenSteps' ] > 0:
        analysisDict[ 'recovPerfRetenDict' ] = recovPerfRetenDict



    return analysisDict












def recoveryPerf( model, inputObject, modelData, forwardHorizon=None, retenData=False, 
                         compareToRefState=False, 
                           # decoded=True, 
                           decoded=False, 
                         ):
    '''  Compares how well converged representations are able to reconstruct previous stimuli. 
    
        Compares:  D * H^q * r(t)   versus   x(t-q)
    '''
    
    
    ''' Check '''
    #-------------------------------------------------
    if (not decoded) and (not compareToRefState):
        raise Exception( 'Cannot have  decoded = compareToRefState = False' )
    
    
    ''' Ground Truth '''
    #-------------------------------------------------
    X = inputObject.stimMat
    [ d, nBetas ] = X.shape 
    
    
    
    ''' Network Encoding '''
    #-------------------------------------------------    
    if ( not hasattr(model,'epochNum') )  or  ( model.epochNum is None ):
        raise Exception( '[recoveryPerf] Given memoryModel.epochNum is None---cannot compute RP on the training model. ' )
    else:     
        epochNum = model.epochNum

    
    # convergedStates = modelData[ epochNum ][ 'convergedStates' ]
    convergedStates = modelData[ epochNum ][ 'convergedStates_encoding' ]
    # convergedStates = modelData[ epochNum ][ 'convergedStates_retention' ]
    [ N, nConvergedStates ] = convergedStates.shape 
    

    

    
    ''' Storage '''
    #-------------------------------------------------
    if forwardHorizon is None:
        forwardHorizon = nBetas - 1
    
    nBetasForFH = nBetas - forwardHorizon    
    
    recovPerf = torch.zeros(  [ nBetasForFH, forwardHorizon+1 ]  )
    reconR = torch.zeros(  [ nBetasForFH, forwardHorizon+1, N ]  )
    reconX = torch.zeros(  [ nBetasForFH, forwardHorizon+1, d ]  )
    
    
    
    ''' Do we have retention phase data? '''
    #-------------------------------------------------
    if retenData:
        convergedStatesReten = modelData[ epochNum ][ 'convergedStates_retention' ]
        recovPerfReten = torch.zeros(  [ nBetasForFH, forwardHorizon+1 ]  )
    
    
        
    

    ''' Model Parameters '''
    #-------------------------------------------------
    if hasattr(model,'D')  and  (model.D in model.parameters()):
        [ D, H ] = [ model.D, model.H ]
    else: 
        [ D, H ] = model.computeDandH( )
    # D = D.detach()
    # H = H.detach() 
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Compare decoded network activity to original input '''
    #-------------------------------------------------------------------------------------
    for i in range( 1, nBetasForFH+1 ):
        
        # print( )
        # print( '==============================' )
        # print( 'BETA', i )
        # print( '==============================' )
        
        
        if compareToRefState:
            # refState = convergedStates[ :, i ].detach()                    ## current stimulus 
            refState = convergedStates[ :, i ]                    ## current stimulus 
            refState = convergedStates[ :, i-1 ]                    ## current stimulus 
            # refState = convergedStatesReten[ :, i-1 ]                    ## current stimulus 
            x = D @ refState
        else:  ## compare directly to stim
            x = X[ :, i-1 ]                                     ## current stimulus 
        
        
        
        
        

        maxQ = forwardHorizon
        
        
        for q in range( maxQ + 1 ):

        
            #--------------------------------
            ''' Go back q times '''
            #-------------------------------- 
            Hq = torch.matrix_power( H, q )                 ## H^q
            
            stateInd = i + q                                ## effectively: ind for subsequent stim
            # r = convergedStates[ :, stateInd-1 ] 
            # r = convergedStates[ :, stateInd ].detach()              ## convergedState after curr stim
            r = convergedStates[ :, stateInd ]              ## convergedState after curr stim
            
            
            #--------------------------------
            ''' Reconstruct '''
            #-------------------------------- 
            # if q > 0:
            #     q = 1
        
            
            
            rHat = Hq @ r                           ## r(tp)  for  p = 1, 2, ..., (nBetas-q)
            xHat = D @ rHat 
            
            
            
            #--------------------------------
            ''' Compare '''
            #-------------------------------- 
            if torch.any(  torch.isnan( xHat )  ):
                print(  'NaN found in xHat for q={}'.format(q)  )
            
            
            # print( 'x =xnorm:', torch.norm(xHat) )

            # print( x) 
            # print( xHat) 


            if decoded: 
                RP = normError( x, xHat )
            else: 
                # RP = normError( r, rHat )
                RP = normError( refState, rHat )

            
            # RP = normError( x, xHat )
            
            
            
            # RP = normError( r, rHat )
            
            # # corr = np.corrcoef( x, xHat )[ 0, 1 ]
            # toCorr = torch.stack(  [ x, xHat ],  dim=1  ).T
            # # print( toCorr) 
            # corr = torch.corrcoef( toCorr )[ 0, 1 ]
            
            
            
            # if q <= 4:
            #     print( 'q =', q )
            #     print( '\t xHat =', xHat )
            #     print( '\t RP =', RP )
            #     print( '\t corr =', corr )




            #--------------------------------
            ''' Store '''
            #-------------------------------- 
            recovPerf[ (i-1), q ] = RP
            reconR[ (i-1), q ] = rHat      
            reconX[ (i-1), q ] = xHat   
            
            
            
            #-----------------------------------------------------------
            ''' Repeat for retention data (if applicable) '''
            #----------------------------------------------------------- 
            if retenData:
                
                rReten = convergedStatesReten[ :, stateInd ]    # The converged states (after the retnetion period)
                
                rHatReten = Hq @ rReten                         # Go back q stims
                xHatReten = D @ rHatReten                       # Reconstruct input
                
                if torch.any(  torch.isnan( xHatReten )  ):
                    print(  'NaN found in xHatReten for q={}'.format(q)  )
                
                
                if decoded: 
                    RPReten = normError( x, xHatReten )             # Compare
                else: 
                    # RPReten = normError( r, rHatReten )             # Compare
                    RPReten = normError( refState, rHatReten )             # Compare
                
                
                # RPReten = normError( x, xHatReten )             # Compare
                # # RPReten = normError( r, rHatReten )             # Compare
            
                recovPerfReten[ (i-1), q ] = RPReten            # Store
            #-----------------------------------------------------------
            
            
            
    if retenData: 
        return recovPerf, recovPerfReten, reconR, reconX
    
    else: 
        return recovPerf, reconR, reconX

             



def retentionPerf( model, inputObject, modelData, forwardHorizon=None, retenData=False ):
    '''  Compares how well converged representations are able to reconstruct previous stimuli. 
    
        Compares:  D * H^q * r(t)   versus   x(t-q)
    '''
    
    
    ''' Ground Truth '''
    #-------------------------------------------------
    X = inputObject.stimMat
    [ d, nBetas ] = X.shape 
    
    
    
    ''' Network Encoding '''
    #-------------------------------------------------    
    if ( not hasattr(model,'epochNum') )  or  ( model.epochNum is None ):
        raise Exception( '[recoveryPerf] Given memoryModel.epochNum is None---cannot compute RP on the training model. ' )
    else:     
        epochNum = model.epochNum

    
    convergedStates = modelData[ epochNum ][ 'convergedStates_encoding' ]
    [ N, nConvergedStates ] = convergedStates.shape 
    

    

    
    ''' Storage '''
    #-------------------------------------------------
    if forwardHorizon is None:
        forwardHorizon = nBetas - 1
    
    nBetasForFH = nBetas - forwardHorizon    
    
    
    retenPerf = torch.zeros(  [ nBetasForFH, forwardHorizon+1 ]  )

    # recovPerf = torch.zeros(  [ nBetasForFH, forwardHorizon+1 ]  )
    # reconR = torch.zeros(  [ nBetasForFH, forwardHorizon+1, N ]  )
    # reconX = torch.zeros(  [ nBetasForFH, forwardHorizon+1, d ]  )
    
    
        
    

    ''' Model Parameters '''
    #-------------------------------------------------
    if hasattr(model,'D')  and  (model.D in model.parameters()):
        [ D, H ] = [ model.D, model.H ]
    else: 
        [ D, H ] = model.computeDandH( )
    # D = D.detach()
    # H = H.detach() 
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Compare decoded network activity to original input '''
    #-------------------------------------------------------------------------------------
    for i in range( 1, nBetasForFH+1 ):
        
        x = X[ :, i-1 ]                                     ## current stimulus 
        # maxQ = nBetas - i 
        # maxQ = min( forwardHorizon, nBetas-i )
        maxQ = forwardHorizon
        
        
        for q in range( maxQ + 1 ):

            #--------------------------------
            ''' Go back q times '''
            #--------------------------------             
            stateInd = i + q                                ## effectively: ind for subsequent stim
            r = convergedStates[ :, stateInd ]              ## convergedState after curr stim
            
            
            #--------------------------------
            ''' Decode '''
            #-------------------------------- 
            xHat = D @ r 
            
            
            #--------------------------------
            ''' Compare '''
            #-------------------------------- 
            if torch.any(  torch.isnan( xHat )  ):
                print(  'NaN found in xHat for q={}'.format(q)  )
            
            RP = normError( x, xHat )


            #--------------------------------
            ''' Store '''
            #-------------------------------- 
            recovPerf[ (i-1), q ] = RP
            reconR[ (i-1), q ] = rHat      
            reconX[ (i-1), q ] = xHat   
            
            
            
            #-----------------------------------------------------------
            ''' Repeat for retention data (if applicable) '''
            #----------------------------------------------------------- 
            if retenData:
                
                rReten = convergedStatesReten[ :, stateInd ]    # The converged states (after the retnetion period)
                
                rHatReten = Hq @ rReten                         # Go back q stims
                xHatReten = D @ rHatReten                       # Reconstruct input
                
                if torch.any(  torch.isnan( xHatReten )  ):
                    print(  'NaN found in xHatReten for q={}'.format(q)  )
                
                RPReten = normError( x, xHatReten )             # Compare
            
                recovPerfReten[ (i-1), q ] = RPReten            # Store
            #-----------------------------------------------------------
            
            
            
    if retenData: 
        return recovPerf, recovPerfReten, reconR, reconX
    
    else: 
        return recovPerf, reconR, reconX

             







# def retentionPerf( model, inputObject, modelData, forwardHorizon=None, retenData=False ):
#     ''' 
#         Compares how the converged representations have retained previous stimuli. 
        
#         Compares:    D * r(t)   versus   x(t-q)
#     '''
    
    
#     # raise Exception( '[retentionPerf] Function not yet finished!' )
    
    
#     ''' Ground Truth '''
#     #-------------------------------------------------
#     X = inputObject.stimMat
#     [ d, nBetas ] = X.shape 
    
    
    
#     ''' Network Encoding '''
#     #-------------------------------------------------    
#     if ( not hasattr(model,'epochNum') )  or  ( model.epochNum is None ):
#         raise Exception( '[recoveryPerf] Given memoryModel.epochNum is None---cannot compute RP on the training model. ' )
#     else:     
#         epochNum = model.epochNum

    
#     convergedStates = modelData[ epochNum ][ 'convergedStates' ]
#     [ N, nConvergedStates ] = convergedStates.shape 
    

    

    
#     ''' Storage '''
#     #-------------------------------------------------
#     if forwardHorizon is None:
#         forwardHorizon = nBetas - 1
    
#     nBetasForFH = nBetas - forwardHorizon    
    
#     retenPerf = torch.zeros(  [ nBetasForFH, forwardHorizon+1 ]  )

    
        
    

#     ''' Model Parameters '''
#     #-------------------------------------------------
#     if hasattr(model,'D')  and  (model.D in model.parameters()):
#         D = model.D
#     else: 
#         D= model.computeDandH()[ 0 ]
#     D = D.detach()
    
    
    
#     #-------------------------------------------------------------------------------------
#     ''' Compare decoded network activity to original input '''
#     #-------------------------------------------------------------------------------------
#     for i in range( 1, nBetasForFH+1 ):
        
#         # print( )
#         # print( '==============================' )
#         # print( 'BETA', i )
#         # print( '==============================' )
        
#         x = X[ :, i-1 ].reshape(d,1)                ## ( d, 1 )
        
        
#         #----------------------------------------
#         ''' The converged states after ti '''
#         #---------------------------------------- 
#         FHInd = i + forwardHorizon + 1              ## from current stim state to end of forwardHorizon
#         # FHInd = i + forwardHorizon                  ## from current stim state to end of forwardHorizon
#         r = convergedStates[ :, i:FHInd ]           ## ( N, forwardHorizon+1 )
         
        
#         #--------------------------------
#         ''' Decode post-stim states '''
#         #-------------------------------- 
#         xHat = D @ r                                ## ( d, forwardHorizon+1 )
        
        
#         #--------------------------------
#         ''' Compare '''
#         #-------------------------------- 
#         if torch.any(  torch.isnan( xHat )  ):
#             print(  'NaN found in xHat for q={}'.format(q)  )
        
        
#         xRep = x.repeat( 1, forwardHorizon+1 )      ## ( d, forwardHorizon+1 )
#         retenPerf[ i-1 ] = normError( xRep, xHat )  ## ( forwardHorizon+1, 1 )
        
        
#         # retenPerf[ i-1 ] = normError( x, xHat )

        
            
            
            
            
#     return retenPerf

             






def costTerms_model( model, inputObject, modelData ):
    '''
    '''
    
    ''' Ground Truth '''
    #-------------------------------------------------
    stimMat = inputObject.stimMat
    [ d, nBetas ] = stimMat.shape 
    
    
    
    ''' Network Encoding '''
    #-------------------------------------------------
    epochNum = model.epochNum
    
    if epochNum is None:
        # convergedStates = modelData[ 'convergedStates' ]
        raise Exception( '[recoveryPerf] Given memoryModel.epochNum is None; cannot compute RP on the training model. ' )
    
    
    # convergedStates = modelData[ epochNum ][ 'convergedStates' ]
    convergedStates = modelData[ epochNum ][ 'convergedStates_encoding' ]
    convergedStates_retention = modelData[ epochNum ][ 'convergedStates_retention' ]
    [ N, nConvergedStates ] = convergedStates.shape 
    

    
    ''' Storage '''
    #-------------------------------------------------
    costTerms = { 'errLoss'     :   torch.zeros( [nBetas,1] ),
                  'hLoss'       :   torch.zeros( [nBetas,1] ),
                  'effLoss'     :   torch.zeros( [nBetas,1] ),
                  'sLoss'       :   torch.zeros( [nBetas,1] ),
                  }
          
    

    ''' Model Parameters '''
    #-------------------------------------------------
    # [ D, H ] = model.computeDandH( )
    [ D, H ] = [  model.D,  model.H  ]
    
    nRetenSteps = modelData[ 'nRetenSteps' ]
    
    
    
    
    for i in range( 1, nBetas+1 ):
        
        currStim = stimMat[ :, i-1 ] 
        currState = convergedStates[ :, i ]
        
        
        if nRetenSteps > 0:
            refState = convergedStates_retention[ :, i ]
        else:
            refState = convergedStates[ :, i-1 ]
    
    
    
        # [ errLoss, hLoss, effLoss, sLoss ] = model.costTerms( currStim, currState, refState, D, H )
        [ errLoss, hLoss, effLoss, sLoss, eigLoss ] = model.costTerms( currStim, currState, refState, D, H )
        
        costTerms[ 'errLoss' ][ i-1 ] = errLoss
        costTerms[ 'hLoss' ][ i-1 ] = hLoss
        costTerms[ 'effLoss' ][ i-1 ] = effLoss
        costTerms[ 'sLoss' ][ i-1 ] = sLoss
        
    

    return costTerms





# def func( model, inputObject, modelData ):
#     '''
#     '''
    

#     return 





# def func( model, inputObject, modelData ):
#     '''
#     '''
    

#     return 





def normError( x, xHat, normAxis=0, unitVectors=False, floorVal=True ):
    '''
        Computes the normed percent error between x and its approximation xHat. 
    '''


    #-------------------------------------------------------------------------------------
    shapeX = x.shape
    shapeXHat = xHat.shape
    
    if len(shapeX) != len(shapeXHat):
        raise Exception( '[normError] Unable to compare data of different sizes ' )
    if len( shapeX ) > 2:
        raise Exception( '[normError] Currently unable to compare data with more than 2 dimensions ' )
    #-------------------------------------------------------------------------------------
    
    
    #-------------------------------------------------------------------------------------
    ''' Compute unit vectors '''
    #-------------------------------------------------------------------------------------
    if unitVectors: 
        
        xNorm = torch.norm( x, dim=normAxis )
        if not torch.isclose( xNorm, 0 ):
            x = x / xNorm
        
        xHatNorm = torch.norm( xHat, dim=normAxis )
        if not torch.isclose( xHatNorm, 0 ):
            xHat = xHat / xHatNorm
    #-------------------------------------------------------------------------------------
    

    
    #-------------------------------------------------------------------------------------
    '''  Normed Percent Error:  | (x-xHat) |  /  | x |  '''
    #-------------------------------------------------------------------------------------
    error = x - xHat
    normedError = torch.norm(  error, p=2, dim=normAxis  )
    
    denom = torch.norm(  x, p=2, dim=normAxis  )
    
    
    # if denom != 0:
    #     percentError = normedError / denom 
    if torch.all( denom != 0 ):
        percentError = normedError / denom 
    else:    
        return 0

    # else: 
    #     nonZeroInds = (denom != 0).nonzeros()[0]
    #     # zeroInds = (denom == 0).nonzeros()[0]
    #     # denom[ zeroInds ] 
    #     percentError = normedError / denom[ nonZeroInds ] 
    
    #-------------------------------------------
    if floorVal:
        percentError = percentError.clip( max=1 ) 
    #-------------------------------------------
    
    
    #-------------------------------------------------------------------------------------
    ''' Percent Recovery '''
    #-------------------------------------------------------------------------------------
    finalVal = (1 - percentError)
    
    # if torch.any(  torch.isnan( xHat )  ):
    #     print( 'xHat = D @ rHat ', xHat )
    #     # print( '\t rHat ', rHat )
    
    
    
    return finalVal








# def 




#=========================================================================================
#=========================================================================================
#%% Matrix structure, etc. 
#=========================================================================================
#=========================================================================================



def numberOfStableEigvals( matrix ): 
    
    
    eigvals = torch.linalg.eigvals( matrix )
    
    eigvalsReal = torch.real( eigvals )
    hurwitzBool = eigvalsReal < 0
    
    nStable = sum( hurwitzBool )
    
    
    return nStable









def printStabilityOfH( epochModels ):
    ''' '''
    
    
    epochNumsToTest = list( epochModels.keys() )
    
    
    hurwitzCount_H = np.zeros(  [ len(epochNumsToTest), 2 ]  )
    cNums = np.zeros(  [ len(epochNumsToTest), 2 ]  )
    
    
    for i in range( len(epochNumsToTest) ):
        
        epochNum = epochNumsToTest[ i ]
    
    
        #--------------------------------------------
        H = epochModels[epochNum].H.detach()
        
        nStable = numberOfStableEigvals( H )
        nUnstable = N - nStable
        
        hurwitzCount_H[ i ] = [ epochNum, nUnstable ]
        #--------------------------------------------
        
        
        #--------------------------------------------
        cNum = np.linalg.cond( H )
        cNums[ i ] = condNum
        #--------------------------------------------

        

    
    print( )
    print( '--------------------------------------------' )
    print( 'Number of unstable eigvals by epochNum:  H ' )
    print( '--------------------------------------------' )
    print( hurwitzCount_H )
    
    
    print( )
    print( '--------------------------------------------' )
    print( 'Condition number by epochNum:  H ' )
    print( '--------------------------------------------' )
    print( cNums )
    
    
    
    



def compareLearnedStructures( epochModels, Hmin=None, Hmax=None ):

    cmap = mpl.colormaps[ 'seismic' ]
    
    epochNumsToTest = list( epochModels.keys() )
    nEpochNumsToTest = len( epochNumsToTest )
    
    
    Hs = [   epochModels[ epochNum ].H.detach().numpy() for epochNum in epochNumsToTest   ]
    # Hvmin = np.min( np.min(Hs) )
    # Hvmax = np.max( np.max(Hs) )
    if Hmax is None:
        Hvmax = np.max(  np.max( np.abs(Hs) )  )
    else: 
        Hvmax = Hmax
        
    if Hmin is None:
        Hvmin = (-1) * Hvmax
    else: 
        Hvmin = Hmin
    
    
    Ds = [   epochModels[ epochNum ].D.detach().numpy() for epochNum in epochNumsToTest   ]
    # Dvmin = np.min( np.min(Ds) )
    # Dvmax = np.max( np.max(Ds) )
    Dvmax = np.max(  np.max( np.abs(Ds) )  )
    Dvmin = (-1) * Dvmax
    
    
    
    # [  nEpochNumsToTest % (nEpochNumsToTest-i) for i in range(1,nEpochNumsToTest) ]
    # a
    
    
    # Hfig, Haxs = plt.subplots( nEpochNumsToTest, 1 )
    
    
    if nEpochNumsToTest == 8:
        Hfig, Haxs = plt.subplots( 4, 2 )
        Hinds1 = [ 0, 1, 2, 3, 0, 1, 2, 3 ]
        Hinds2 = [ 0, 0, 0, 0, 1, 1, 1, 1 ]
    elif (nEpochNumsToTest == 7): 
        Hfig, Haxs = plt.subplots( 4, 2 )
        Hinds1 = [ 0, 1, 2, 0, 1, 2, 3 ]
        Hinds2 = [ 0, 0, 0, 1, 1, 1, 0 ]
    elif (nEpochNumsToTest == 6):
        Hfig, Haxs = plt.subplots( 3, 2 )
        Hinds1 = [ 0, 1, 2, 0, 1, 2 ]
        Hinds2 = [ 0, 0, 0, 1, 1, 1 ]
    else: 
        Hfig, Haxs = plt.subplots( nEpochNumsToTest, 1 )
        
    Hfig.subplots_adjust( hspace=0.5, wspace=-0.65 )
    
    
    
    Dfig, Daxs = plt.subplots( nEpochNumsToTest, 1 )
    
    
    
    for i in range( len(epochNumsToTest) ): 
        
        H = Hs[i]
        D = Ds[i]
        
        
        
        
        
        if Haxs.shape[0] == nEpochNumsToTest:   
            currHax = Haxs[ i ] 
            
            # Haxs[i].imshow( H, vmin=Hvmin, vmax=Hvmax, cmap=cmap )
            
            # Haxs[i].set_ylabel( str(epochNumsToTest[i]) )
           
            # Haxs[i].axes.get_xaxis().set_ticks([])
            # Haxs[i].axes.get_yaxis().set_ticks([])
            
            
        else:        
            Hind1 = Hinds1[ i ] 
            Hind2 = Hinds2[ i ]
            currHax = Haxs[ Hind1, Hind2 ]

            # Hind1 = Hinds1[ i ] 
            # Hind2 = Hinds2[ i ]
            # Himg = Haxs[Hind1, Hind2].imshow( H, vmin=Hvmin, vmax=Hvmax, cmap=cmap )
            
            # Haxs[Hind1, Hind2].set_ylabel( str(epochNumsToTest[i]) )
            
            # Haxs[Hind1, Hind2].axes.get_xaxis().set_ticks([])
            # Haxs[Hind1, Hind2].axes.get_yaxis().set_ticks([])
            
            
            
        Himg = currHax.imshow( H, vmin=Hvmin, vmax=Hvmax, cmap=cmap )
        currHax.set_ylabel( str(epochNumsToTest[i]) )
        currHax.axes.get_xaxis().set_ticks([])
        currHax.axes.get_yaxis().set_ticks([])
            
        
            
        Dimg = Daxs[i].imshow( D, vmin=Dvmin, vmax=Dvmax, cmap=cmap )
        Daxs[i].set_ylabel( str(epochNumsToTest[i]) )
        Daxs[i].axes.get_xaxis().set_ticks([])
        Daxs[i].axes.get_yaxis().set_ticks([])
        
        

        
        
        
        # Hfig.colorbar() 
        # Dfig.colorbar() 
    
    
    Hfig.suptitle( 'Evolution of H over epochNum' )
    Dfig.suptitle( 'Evolution of D over epochNum' )
    
    
    caxH = Hfig.add_axes(  [0.75, 0.15, 0.05, 0.75]  )
    # divider3 = make_axes_locatable( Haxs[:, :] )
    # caxH = divider3.append_axes("right", size="20%", pad=0.05)
    plt.colorbar( Himg, caxH ) 
    
    caxD = Dfig.add_axes(  [0.75, 0.15, 0.05, 0.75]  )
    # divider3 = make_axes_locatable( Daxs[i] )
    # caxD = divider3.append_axes("right", size="20%", pad=0.05)
    plt.colorbar( Dimg, caxD ) 

    
    
    return Hfig, Dfig 






def compareHq( epochModels, maxQ=6 ):


    Hqfig, Hqaxs = plt.subplots( maxQ, 1 )
    cmap = mpl.colormaps[ 'seismic' ]
    
    
    
    epochNumsToTest = list( epochModels.keys() )
    
    
    Hs = [   epochModels[ epochNum ].H.detach().numpy() for epochNum in epochNumsToTest   ]
    Hvmin = np.min( np.min(Hs) )
    Hvmax = np.max( np.max(Hs) )
    
    H = Hs[ -1 ]
    # H = Hs[ 0 ]
    
    
    
    for q in range( 1, maxQ+1 ):
        
    
        Hq = torch.linalg.matrix_power( torch.tensor(H), q )
        
        # Hqaxs[q].imshow( Hq, vmin=np.min(H), vmax=np.max(H), cmap=cmap )
        Hqaxs[q-1].imshow( Hq, vmin=Hvmin, vmax=Hvmax, cmap=cmap )
        
        Hqaxs[q-1].axes.get_yaxis().set_ticks([])
        Hqaxs[q-1].axes.get_xaxis().set_ticks([])
        
        Hqaxs[q-1].set_ylabel( 'q = ' + str(q) )
    
    
    
    Hqfig.suptitle( r'$\mathbf{H}^q$' )
    # Hqaxs[i].axes.get_yaxis().set_visible(False)



    return Hqfig




def plotMatEigvals( A, fig=None ): 
    
    if fig is None:
        fig = plt.figure( ) 
    
    
    eigvals = torch.linalg.eig( A )[ 0 ]
    
    
    x = torch.real( eigvals ).detach().numpy()
    y = torch.imag( eigvals ).detach().numpy()
    
    anyComplex = np.any(  y != 0  )
    if anyComplex:
        plt.xlabel( 'Real' )
        plt.ylabel( 'Imaginary' )
        
    plt.scatter( x, y )
    plt.title( 'Eigenvalues' )
    
    
    return fig 




def plotComplexEigvals( eigvals, fig=None, color='k' ): 
    
    x = torch.real( eigvals ).detach().numpy()
    y = torch.imag( eigvals ).detach().numpy()
    
    plt.scatter( x, y, c=color, s=15 )
    
    plt.xlabel( 'Real' )
    plt.ylabel( 'Imaginary' )
        
    # plt.title( 'Eigenvalues' )
    
    
    return fig 





# def plotLearnedMatrixStructures( epochModels ):
    
#     epochNumsToTest = list( epochModels.keys() )
#     nEpochNums = len( epochNumsToTest )
    
    
#     fig, axs = plt.subplots( 2, nEpochNums, height_ratios=[5,1] )
    
#     Hs = [  epochModels[i].H for i in epochNumsToTest  ]
#     vmin = min(  [ torch.min(H) for H in Hs ]  )
#     vmax = max(  [ torch.max(H) for H in Hs ]  )
    
    
#     for i in range( nEpochNums ):
#     # for epochNum in epochNumsToTest:
        
#         # epochNum = epochNumsToTest[i]
#         # model = epochModels[ epochNum ]
#         # D = model.D
#         # H = model.H 
#         H = Hs[ i ]
        
#         eigvals = torch.linalg.eig( H )[ 0 ]
        
#         axs[0,i].imshow(  H.detach().numpy(),  vmin=vmin,  vmax=vmax  )
#         plotComplexEigvals(  eigvals,  fig=axs[1,i]  )
        
        
#     # plt.colorbar()
    
    
#     return fig 





# def plotLearnedMatrixStructures( epochModels ):
    
#     epochNumsToTest = list( epochModels.keys() )
#     nEpochNums = len( epochNumsToTest )
    
    
#     # fig, axs = plt.subplots( 2, nEpochNums, height_ratios=[5,1] )
    
#     Hs = [  epochModels[i].H for i in epochNumsToTest  ]
#     vmin = min(  [ torch.min(H) for H in Hs ]  )
#     vmax = max(  [ torch.max(H) for H in Hs ]  )
    
#     fig, axs = plt.subplots( 1, nEpochNums )
#     # fig, axs = plt.subplots( 2, 4 )
#     for i in range( nEpochNums ):
#         H = Hs[ i ].detach().numpy()
#         axs[i].imshow(  H, vmin=vmin, vmax=vmax  )
#     plt.colorbar( )
    
    
    
    
#     Ds = [  epochModels[i].D for i in epochNumsToTest  ]
#     vmin = min(  [ torch.min(D) for D in Ds ]  )
#     vmax = max(  [ torch.max(D) for D in Ds ]  )
    
#     fig, axs = plt.subplots( 1, nEpochNums )
#     # fig, axs = plt.subplots( 2, 4 )

#     for i in range( nEpochNums ):
#         D = Ds[ i ].detach().numpy()
#         axs[i].imshow(  D, vmin=vmin, vmax=vmax  )
#     plt.colorbar(  )
        
        
        
    
    
#     # # for i in range( nEpochNums ):
#     # for epochNum in epochNumsToTest:
        
#     #     # epochNum = epochNumsToTest[i]
#     #     model = epochModels[ epochNum ]
#     #     # D = model.D
#     #     H = model.H 
#     #     # H = Hs[ i ]
        
#     #     fig, axs = plt.subplots( 2,1 )
        
        
        
#     #     eigvals = torch.linalg.eig( H )[ 0 ]
        
#     #     axs[0].imshow(  H.detach().numpy(),  vmin=vmin,  vmax=vmax  )
#     #     plotComplexEigvals(  eigvals,  fig=axs[1]  )
        
        
#     # # plt.colorbar()
    
    
#     return fig 




def plotLearningEigvals( epochModels ):
    
    
    epochNumsToTest = list( epochModels.keys() )
    nEpochNums = len( epochNumsToTest )
    
    
    #---------------------------------------------------------
    fig = plt.figure()    
    
    Hs = [  epochModels[i].H for i in epochNumsToTest  ]
    vmin = min(  [ torch.min(H) for H in Hs ]  )
    vmax = max(  [ torch.max(H) for H in Hs ]  )
    #---------------------------------------------------------
    
    
    
    #---------------------------------------------------------
    colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]
    nColors = len( colors )
    
    colorDict = { }
    for i in range( nColors ):
        colorDict[ i ] = colors[ nEpochNums - i - 1 ]
    
    colorDict = updateColorDictLabels( colorDict, epochNumsToTest )
    #---------------------------------------------------------
    
    
    
    # for epochNum in epochNumsToTest:
    for i in range(  nEpochNums  ):
        
        epochNum = epochNumsToTest[ i ]
        H = epochModels[ epochNum ].H 
        eigvals = torch.linalg.eig( H )[ 0 ]
        
        fig = plotComplexEigvals(  eigvals,  fig=fig, color=colorDict[epochNum] )
        
        
    plt.axvline( x=0, c='gray', linewidth=1 )
        
        
    
    cbar = discreteColorbar( colorDict, fig, colorbarLabel='Epoch' )

    
    
    
    return fig 




def plotLearnedSingValsOfD( epochModels, colorDict,): #cmap=mpl.colormaps['viridis'] ): 
    

    epochNumsToTest = list(  epochModels.keys()  )
    nModels = len( epochNumsToTest )
    DList = [  epochModels[num].D.detach().numpy() for num in epochNumsToTest  ]    
    
    d = DList[0].shape[0]
    # colors = [  cmap( i/nModels )  for i in range(nModels) ] 
    colors = [  colorDict[i] for i in range(nModels)  ]  
    
    
    colorbarDict = { }
    singVals = np.zeros(  [ nModels, d ]  ) 
    for i in range( nModels ):
        
    
        D = DList[ i ]
        [ U, S, V ] = np.linalg.svd( D )
        singVals[ i ] = S
        
        
        colorbarDict[ epochNumsToTest[i] ] = colors[i]
    
    fig = plt.figure( )
    # plt.scatter( [range(nModels)]*d, singVals.T, c=colors )
    plt.scatter( singVals[:,0], singVals[:,1], c=colors )
    # plt.scatter( [range(nModels)]*d, singVals.T, c=colors )


    cbar = discreteColorbar( colorbarDict, fig, colorbarLabel='Epochs' )
    
    plt.suptitle( 'Singular values of D' )



    return fig, singVals





def compareLearnedBasisVectors( epochModels, baseType='D' ):
    
    
    epochNumsToTest = list(  epochModels.keys()  )
    nModels = len( epochNumsToTest )

    if baseType == 'D':
        matList = [  epochModels[num].D.detach().numpy() for num in epochNumsToTest  ]
    elif baseType == 'H':
        matList = [  epochModels[num].H.detach().numpy() for num in epochNumsToTest  ]
    else: 
        raise Exception( 'Did not understand baseType' )
    
    
    nBases = matList[0].shape[0] 
    
    basisVectCorrs = np.zeros( [nModels, nBases, nBases] )
    
    for i in range( nModels ):
        basisVects = matList[ i ]               ## ( nBases, N )
        
        corr = np.corrcoef( basisVects )        ## correlate the ROWS
    
        basisVectCorrs[ i ] = corr
        
        
        
    
    
    
    return 










#=========================================================================================
#=========================================================================================
#%% Plotting
#=========================================================================================
#=========================================================================================


    
# def plotRP( recoveryPerfs, nBetas, forwardHorizon=None, 
def plotRP( analysisDict, nBetas, forwardHorizon=None, 
                                   firstBetaToPlot=1, 
                                   epochNumsToPlot=None, 
                                   colors=None, colorbarLabel='Epoch',
                                   endOfEncoding=True,
                                   ):
    ''' 
        Recovery performance---norm (percent) error between original stimulus and what
    is reconstructed via the history matrix after the fact. 
    
        Each line represents the reovery performace after a given nuber of trianing  
    epochs, averaged over each stimulus (or, if firstStimOnly=True, is just wrt the 
    first stimulus).            


    Plots the reconstruction of beta1 (unless forwardHorizon < nBetas)
                      
    '''
    
    
    if endOfEncoding:
        recoveryPerfs = analysisDict[ 'recovPerfDict' ]
        titleStr = 'Recovery Performance: end of encoding period'
        yLabel = r'percentError$( \mathbf{z}(t_i-q), \mathbf{H^q}\mathbf{z}(t_i) )$'

    else:
        recoveryPerfs = analysisDict[ 'recovPerfRetenDict' ]    
        titleStr = 'Recovery Performance: end of delay period'
        yLabel = r'percentError$( \mathbf{z}(t_i-q), \mathbf{H^q}\mathbf{z}(t_i) )$'
        
        
    try: 
        if not analysisDict[ 'compareToRefState' ]:
            yLabel = r'percentError$( \mathbf{u}(t_i-q), \mathbf{DH^q}\mathbf{z}(t_i) )$'
    except: 
        pass

    
    
    
    epochNumsToTest = list(  recoveryPerfs.keys()  )
    
    if epochNumsToPlot is None:
        epochNumsToPlot = epochNumsToTest
        
     
        
    if forwardHorizon is None:
        forwardHorizon = nBetas - 1             # maxQ 
    #---------------------------------------------------------------------------------
    
    
    #---------------------------------------------------------------------------------
    ''' Make sure the colors are in order '''
    #---------------------------------------------------------------------------------
    if colors is None:
        colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]
        
        cmap = mpl.colormaps['hsv']
        nColors = len(epochNumsToPlot)
        cmap = cmap.resampled(  nColors+2  )            ## +2 to avoid the red at the end 
        
        colors = [  cmap(i) for i in range( nColors )  ]
        
    colorLabels = epochNumsToTest
        
        
    if type( colors ) is list:
        
        colorDict = { }
        for i in range( len(colors) ):
            colorDict[ i ] = colors[i]
            
    else: 
        colorDict = colors
    
    
    colorDict = updateColorDictLabels( colorDict, epochNumsToPlot )
    #---------------------------------------------------------------------------------
    
        
            
    #--------------------------------------
    ''' Create the figure '''
    #--------------------------------------
    fig, ax = plt.subplots( 1,1 )

    plt.suptitle( titleStr )    
    plt.ylabel( yLabel )
    plt.xlabel( 'q' )
        
    
    # for epochInd in range( len(epochNumsToTest) ):
    for epochInd in range( len(epochNumsToPlot) ):
        
        # epochNum = epochNumsToTest[ epochInd ]
        epochNum = epochNumsToPlot[ epochInd ]
        # currColor = colors[ epochInd ]
        currColor = colorDict[ epochNum ]
        
        # colorDict[ epochNum ]

        
        #--------------------------------------
        ''' Get the data for this epoch '''
        #--------------------------------------
        epochData = recoveryPerfs[ epochNum ]                   # ( nBetas, nBetas )
        
        
        maxBetaForFH = nBetas - forwardHorizon                  # beta index i
        maxQForFH = forwardHorizon                              # max q
        
        maxBetaInd = maxBetaForFH                               # since i = 1,2,3,...,nBetas
        maxQInd = maxQForFH + 1                                 # since q = 0,1,2,...,(nBetas-1)

        toAvg = epochData[ 0:maxBetaInd, 0:maxQInd ]      # ( maxBetaForFH, forwardHorizon+1 )
        toPlot = torch.mean( toAvg, dim=0 )
        
        # arrayToPlot = toPlot.detach().numpy().T
        arrayToPlot = toPlot.detach().numpy().T
    
        #--------------------------------------
        ''' Plot the line for this epoch '''
        #--------------------------------------
        if epochNum == 0:
            lineLabel = 'Before training'
        else: 
            lineLabel = 'After ' + str(epochNum) + ' epochs'
        
        plt.plot( arrayToPlot, c=currColor )
        
        
        
        
        
        x = list( range(forwardHorizon+1) )
        # print( )
        # print( len(x) ) 
        # print( arrayToPlot ) 
        plt.scatter( x, arrayToPlot, label=lineLabel, c=currColor )
        
        
        
    #--------------------------------------
    ''' Wrap up this figure '''
    #--------------------------------------
    xTicks = x
    xTickLabels = [ str(q) for q in  xTicks ]
    plt.xticks(  xTicks, xTickLabels  )
    
    
    # if np.any( arrayToPlot < 0 ):
    #     plt.ylim( [-1, 1] )
    #     yLabel = r'corr$( \mathbf{x}(t_i-q), \mathbf{H^q}\mathbf{r}(t_i) )$'
    #     plt.ylabel( yLabel )
    
    plt.ylim(  [ 0, 1 ]  )
    plt.yticks(  np.linspace( 0, 1, 6 )  )
    
    
    cbar = discreteColorbar( colorDict, fig, colorbarLabel=colorbarLabel )
    
    # plt.legend( ) 
    
    
    figDict = { 'recoveryPerf' : fig }


    return fig
    # return figDict








def plotModelEncodingAccur_training( model, inputObject, modelData, z=5 ): 

    
    modelType = 'Training'


    #-------------------------------------------------------------------------------------
    ''' Input (stimulus) data '''
    #-------------------------------------------------------------------------------------
    stimArray = inputObject.stimMat.numpy( )                    # ( d, nStims )
    [ d, nStims ] = np.shape( stimArray )
    
    stimTimeInds = modelData[ 'stimTimeInds' ]
    convergedTimeInds = modelData[ 'convergedTimeInds' ]

    nEncodSteps = modelData[ 'nEncodSteps' ]
    if 'nRetenSteps' in modelData.keys():
        nTotalEvolSteps = nEncodSteps  +  modelData[ 'nRetenSteps' ]
    else:
        nTotalEvolSteps = nEncodSteps 

    

    #-------------------------------------------------------------------------------------
    ''' Model Activity '''
    #-------------------------------------------------------------------------------------
    
    ''' a. The state evolution '''
    #---------------------------------------------------------------------------------
    stateArray = modelData[ 'state' ].detach()          ## remove grad and change to numpy array
    xLabel = 'Epoch i'
    
    [ N, nTimes ] = stateArray.shape
    
    
    ''' b. Decode the state '''
    #---------------------------------------------------------------------------------
    if 'frontAndBackEnd' in modelData.keys():
        decoders = modelData['frontAndBackEnd']['D']         # ( nAvailTrainingDecoders, d, N )
        nAvailTrainingDecoders = decoders.shape[0]
    else:
        nAvailTrainingDecoders = 2                                                                  # naive and final epoch are always auto-stored in trainingData
        decoders = torch.stack( [ modelData['D'][0], modelData['D'][-1] ], dim=0 )            # ( nAvailTrainingDecoders, d, N )

        
    # stateArray = torch.stack(  [ stateArray[:,(i*nProxSteps):(i*nProxSteps+nProxSteps)] for i in range(nAvailTrainingDecoders) ], dim=0  ) # ( nAvailTrainingDecoders, N, nProxSteps )
    nEachSide = int( nAvailTrainingDecoders/2 )
    decoderNums = list(range(nEachSide)) + list(range(nStims-nEachSide,nStims))
    
    stateArray = torch.stack(  [ stateArray[:,(i*nTotalEvolSteps):(i*nTotalEvolSteps+nTotalEvolSteps)] for i in decoderNums ], dim=0  ) # ( nAvailTrainingDecoders, N, nProxSteps )
    
    
    decodedArray = decoders.detach()  @  stateArray                                                 # ( nAvailTrainingDecoders, d, nProxSteps )
    decodedArray = torch.cat(  [decodedArray[i] for i in range(nAvailTrainingDecoders)],  dim=1  )  # ( d, nAvailTrainingDecoders*nProxSteps )


    ''' c. Where to cut the data along the x-axis '''
    #---------------------------------------------------------------------------------
    z = min(  z,  int(nAvailTrainingDecoders/2)  )              # 
    nEntries = nTotalEvolSteps * z


    print( 'decodedArray.shape[1]', decodedArray.shape[1])  
    cutInds1 = [  nEntries,  decodedArray.shape[1] - nEntries  ]    # decoded
    cutInds2 = [  z,   nStims-z  ]                                  # stims 
    
    


    #-------------------------------------------------------------------------------------
    ''' The data to plot '''
    #-------------------------------------------------------------------------------------
    # f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
        
    nEntries = nTotalEvolSteps * int(nAvailTrainingDecoders/2)


    x1 = list(range(nTimes)) 
    x1 = [ x1[ 0:nEntries ] + x1[ -nEntries:: ]  ]  *  d
    x1 = np.array( x1 )

    y1 = torch.cat(  [ decodedArray[:,0:nEntries], decodedArray[:,-nEntries::] ], dim=1 )
    y1 = y1.numpy()

    #----------------------------------------------

    x2 = [ convergedTimeInds ]  *  d
    x2 = np.array( x2 )    

    y2 = stimArray
    
    
    print( )
    print( 'x1', x1.shape )
    print( 'y1', y1.shape )

    print( 'x2', x2.shape )
    print( 'y2', y2.shape )
    
    print( 'cutInds1', cutInds1 )
    print( 'cutInds2', cutInds2 )

    

    
    #-------------------------------------------------------------------------------------
    ''' PLOT '''
    #-------------------------------------------------------------------------------------
    if nStims > 2*z:
        
        fig, axs = plt.subplots( 1, 2 )
                
        #---------------------------------------------------------------------------------
        # cutInds1 = [  (nProxSteps*z),  nTimes - (nProxSteps*z)  ]
        fig = plotCutAxis( x1, y1, cutInds=cutInds1, fig=fig, 
                                              label='$Dr(t)$', color='k', size=2  )
        
        #-----------------------------------
        
        # cutInds2 = [  z,   nStims-z  ]
        fig = plotCutAxis( x2, y2, cutInds=cutInds2, fig=fig, 
                                              label='$x(t_{i-1})$', color='r', size=10  )
        #---------------------------------------------------------------------------------
        
        
    
        #----------------------------------------------
        # fig.get_axes()[0].set_xlabel( 'stim index i' )
        # fig.get_axes()[1].set_xlabel( 'stim index i' )
        
        yMin = np.min(  [ axs[0].get_ylim()[0],  axs[1].get_ylim()[0] ]  )
        yMax = np.max(  [ axs[0].get_ylim()[1],  axs[1].get_ylim()[1] ]  )
        
        axs[0].set_ylim(  [ yMin, yMax ]  )
        axs[1].set_ylim(  [ yMin, yMax ]  )
        
        fig.supxlabel( xLabel )
        
        xTicks1 = stimTimeInds[ 0:z ]
        xTickLabels1 = range( 1, z+1 )
        axs[0].set_xticks( xTicks1, xTickLabels1 )
        
        xTicks2 = stimTimeInds[ -z:: ]
        xTickLabels2 = range( nStims-z, nStims )
        axs[1].set_xticks( xTicks2, xTickLabels2 )
        
        
        plt.legend( )
        plt.suptitle( 'Encoding: ' +  modelType )
        #----------------------------------------------
        
        
        return fig
    
    
    
    else: 
        
        #-------------------------------------------------------------------------------------
        fig = plt.figure( ) 
    
        plt.scatter(  x1, decodedArray, c='k', s=2, label='$Dr(t)$'  )
        plt.scatter(  x2, stimArray, c='r', label='$x(t_{i-1})$'  )
        
        
        xTicks = stimTimeInds
        xTickLabels = range( nStims )
        
        plt.xticks( xTicks, xTickLabels )
        
        
        plt.legend( )
        plt.title( 'Encoding: ' + modelType )    
        plt.xlabel( 'stim index i' )
        #-------------------------------------------------------------------------------------
        
    

    return fig 





def plotModelEncodingAccur( model, inputObject, modelData, z=5 ): 
    '''  '''
    
    
    modelType = 'Testing (epoch {})'.format( model.epochNum )
    
    
    #-------------------------------------------------------------------------------------
    ''' Input (stimulus) data '''
    #-------------------------------------------------------------------------------------
    stimArray = inputObject.stimMat.numpy( )                    # ( d, nStims )
    # if inputObject.circleStims:
    #     stimArray = self.thetas
    [ d, nStims ] = np.shape( stimArray )
    
    stimTimeInds = modelData[ 'stimTimeInds' ]
    # convergedTimeInds = modelData[ 'convergedTimeInds' ]
    convergedTimeIndsReten = modelData[ 'convergedTimeIndsReten' ]

    nEncodSteps = modelData[ 'nEncodSteps' ]
    nRetenSteps = modelData[ 'nRetenSteps' ] 
    nTotalEvolSteps = nEncodSteps + nRetenSteps

    

    #-------------------------------------------------------------------------------------
    ''' Model Activity '''
    #-------------------------------------------------------------------------------------
    
    ''' a. The state evolution '''
    #---------------------------------------------------------------------------------
    stateArray = modelData[ model.epochNum ][ 'state' ].detach()          ## remove grad and change to numpy array
    xLabel = 'stim index i'

    
    
    ''' b. Decode the state '''
    #---------------------------------------------------------------------------------
    if hasattr(model,'D')  and  (model.D in model.parameters()):
        [ D, H ] = [ model.D, model.H ]
    else: 
        [ D, H ] = model.computeDandH( )
    D = D.detach()
    H = H.detach()
    

    decodedArray = D.detach()  @  stateArray              # ( d, nTimes )
    
    [ d, nTimes ] = np.shape( decodedArray )
    
    
    ''' c. Where to cut the data along the x-axis '''
    #---------------------------------------------------------------------------------
    nEntries = nTotalEvolSteps * z

    cutInds1 = [  nEntries,  nTimes - nEntries  ]    # decoded
    cutInds2 = [  z,   nStims-z  ]                                  # stims 
    



    #-------------------------------------------------------------------------------------
    ''' The data to plot '''
    #-------------------------------------------------------------------------------------
    # f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    
    x1 = [ range(nTimes) ] * d
    x1 = np.array( x1 )

    y1 = decodedArray[ :, 0:nEntries ] + decodedArray[ :, -nEntries:: ]
    y1 = decodedArray.numpy()

    #--------------

    x2 = [ convergedTimeIndsReten ]  *  d
    x2 = np.array( x2 )    

    y2 = stimArray
    
    
    print( )
    print( 'x1', x1.shape )
    print( 'y1', y1.shape )

    print( 'x2', x2.shape )
    print( 'y2', y2.shape )

    

    
    #-------------------------------------------------------------------------------------
    ''' PLOT '''
    #-------------------------------------------------------------------------------------
    if nStims > 2*z:
        
        fig, axs = plt.subplots( 1, 2 )
                
        #---------------------------------------------------------------------------------
        # cutInds1 = [  (nProxSteps*z),  nTimes - (nProxSteps*z)  ]
        fig = plotCutAxis( x1, y1, cutInds=cutInds1, fig=fig, 
                                              # label='$Dr(t)$', color='k', size=2, plotLine=True )
                                              label='$Dr(t)$', color='k', size=2, plotLine=False )
        
        #-----------------------------------
        
        # cutInds2 = [  z,   nStims-z  ]
        fig = plotCutAxis( x2, y2, cutInds=cutInds2, fig=fig, 
                                              label='$x(t_{i-1})$', color='r', size=10  )
        #---------------------------------------------------------------------------------
        
        
    
        #----------------------------------------------
        # fig.get_axes()[0].set_xlabel( 'stim index i' )
        # fig.get_axes()[1].set_xlabel( 'stim index i' )
        
        yMin = np.min(  [ axs[0].get_ylim()[0],  axs[1].get_ylim()[0] ]  )
        yMax = np.max(  [ axs[0].get_ylim()[1],  axs[1].get_ylim()[1] ]  )
        
        axs[0].set_ylim(  [ yMin, yMax ]  )
        axs[1].set_ylim(  [ yMin, yMax ]  )
        
        fig.supxlabel( xLabel )
        
        xTicks1 = stimTimeInds[ 0:z ]
        xTickLabels1 = range( 1, z+1 )
        axs[0].set_xticks( xTicks1, xTickLabels1 )
        
        xTicks2 = stimTimeInds[ -z:: ]
        xTickLabels2 = range( nStims-z, nStims )
        axs[1].set_xticks( xTicks2, xTickLabels2 )
        
        
        plt.legend( )
        plt.suptitle( 'Encoding: ' +  modelType )
        #----------------------------------------------
        
        
        return fig
    
    
    
    else: 
        
        #-------------------------------------------------------------------------------------
        fig = plt.figure( ) 
    
        plt.scatter(  x1, decodedArray, c='k', s=2, label='$Dr(t)$'  )
        plt.scatter(  x2, stimArray, c='r', label='$x(t_{i-1})$'  )
        
        
        xTicks = stimTimeInds
        xTickLabels = range( nStims )
        
        plt.xticks( xTicks, xTickLabels )
        
        
        plt.legend( )
        plt.title( 'Encoding: ' + modelType )    
        plt.xlabel( 'stim index i' )
        #-------------------------------------------------------------------------------------
        
        
        
    return fig 





def plotProxEvolError( proxEvolErrorDict, colors=None, 
                      # errorType='percent' 
                      ):

    epochNumsToPlot = list(  proxEvolErrorDict.keys()  )    
    nEpochNumsToPlot = len( epochNumsToPlot )
    
    
    [ nBetas, nProxSteps ] = proxEvolErrorDict[ epochNumsToPlot[0] ][ 'absolute' ].shape
    
    
    
    
    

    #--------------------------------------
    ''' Create the figure '''
    #--------------------------------------
    # if type(errorType) is str:
    #     nrows = 1
    #     errorTypes = [ errorType ]
    # else:
    #     nrows = len(errorType)
    #     errorTypes = errorType
        
    nrows = 2
    
    
    
    fig, axs = plt.subplots( nrows=nrows, ncols=1, sharex=True,
                                # gridspec_kw={'height_ratios': [2, 1]},
                                )
    titleStr = 'Proximal encoding convergence'
    # fig.set_figheight(8)
    # fig.set_figwidth(8)


    ## https://www.geeksforgeeks.org/how-to-create-different-subplot-sizes-in-matplotlib/
    # spec = mpl.gridspec.GridSpec( ncols=1, nrows=2,
    #                                  # width_ratios=[2, 1], 
    #                                  # wspace=0.5,
    #                                  hspace=0.5, height_ratios=[2, 1])
    
    # ax0 = fig.add_subplot(spec[0])
    # ax1 = fig.add_subplot(spec[1])

    ax0 = axs[0]
    ax1 = axs[1]
    
    plt.suptitle( titleStr )
    
    # yLabel = r'$ | \mathbf{x}(t_i - q) - \mathbf{H}^q\mathbf{z}^q(t_i) | $'
    
    # for errType in errorTypes:
    #     if errType == 'absolute':
    #         ylabel = 'Absolute Error'
    #     elif errType == 'percent':
    #         ylabel = 'Percent Error'
    #     axs
    
    yLabel1 = r'Absolute Error'
    ax0.set_ylabel( yLabel1 )
    # ax0.set_xlabel( 'k' )
    
    yLabel2 = r'Percent Error'
    ax1.set_ylabel( yLabel2 )
    ax1.set_xlabel( 'k' )

    #------------------------------------------------------------
    if colors is None:
        colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]
        
    colorLabels = epochNumsToPlot

    if type( colors ) is list:
        colorDict = { }
        for i in range( len(colors) ):
            colorDict[ i ] = colors[i]
            
    else: 
        colorDict = colors
          
    colorDict = updateColorDictLabels( colorDict, epochNumsToPlot )
    #------------------------------------------------------------
    





        
    #------------------------------------------------------------

    for epochInd in range( nEpochNumsToPlot ):
        
        epochNum = epochNumsToPlot[ epochInd ]
        epochProxError = proxEvolErrorDict[ epochNum ][ 'absolute' ].detach().numpy()            # ( nBetas, nProxSteps )
        epochProxPercentError = proxEvolErrorDict[ epochNum ][ 'percent' ].detach().numpy()      # ( nBetas, nProxSteps )
        
        # print( 'epochProxError', epochProxError.shape )
        # print( 'epochProxPercentError', epochProxPercentError.shape )
        
        avgError = np.mean( epochProxError, axis=0 )                                    # ( 1, nProxSteps )
        avgPercentError = np.mean( epochProxPercentError, axis=0 )                      # ( 1, nProxSteps )
        
        if epochNum == 0:
            label = 'Before training'
        else:
            label = 'After {} epochs'.format( epochNum )
        
        ax0.scatter( range(nProxSteps), avgError, c=colors[epochInd], s=5 )
        ax0.plot( avgError, c=colors[epochInd], label=label )
        
        ax1.scatter( range(nProxSteps), avgPercentError, c=colors[epochInd], s=5 )
        ax1.plot( avgPercentError, c=colors[epochInd], label=label )


        # toPlot[ epochInd ] = avgError
        
        
        
        
    
        #------------------------------------------------------------
    
    # plt.legend()
    cbar = discreteColorbar( colorDict, fig, colorbarLabel='Epoch' )

    

    return fig 




















def plotCutAxis( x, y, cutInds, axis='x', fig=None, label=None, color=None, size=10, plotLine=False ):
    # Reference: https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
    
    #-------------------------------------
    if fig is None:
        fig, axs = plt.subplots( 1, 2 )
    else: 
        axs = fig.get_axes()
    #-------------------------------------
    
    
    #----------------------------------------------
    x1 = x[ :, 0:cutInds[0] ]
    x2 = x[ :, cutInds[1]:: ]
    
    y1 = y[ :, 0:cutInds[0] ]
    y2 = y[ :, cutInds[1]:: ]
    #----------------------------------------------
    
    
    #----------------------------------------------
    axs[0].scatter(  x1, y1, c=color, s=size, label=label )
    axs[1].scatter(  x2, y2, c=color, s=size, label=label )
     
    if plotLine:
        axs[0].plot(  x1, y1, c=color, linewidth=0.5 )
        axs[1].plot(  x2, y2, c=color, linewidth=0.5 )
    #----------------------------------------------
    

    
    #----------------------------------------------
    ## Reference: 
    # hide the spines between ax and ax2
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    
    axs[0].yaxis.tick_left()                # Set ticks to visible on the left 
    
    axs[1].yaxis.tick_right()               # Ticks visible also on the right
    # axs[1].yaxis.set_ticks([])              # Remove ticks on the right hand side
    
    
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
    axs[0].plot((1-d, 1+d), (-d, +d), **kwargs)
    axs[0].plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=axs[1].transAxes)  # switch to the bottom axes
    axs[1].plot((-d, +d), (1-d, 1+d), **kwargs)
    axs[1].plot((-d, +d), (-d, +d), **kwargs)



    return fig
    
    
    
    
    
    
    
    
    
    # #----------------------------------------------
    # if (fig is None):
    #     fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    # else:
    #     ax = fig.axes[0]
    #     ax2 = fig.axes[0]
    # #----------------------------------------------
    
    
    
    # x = np.array( x )
    # y = np.array( y )
    
    # ax.scatter( x.T, y.T )
    # ax2.scatter( x.T, y.T )
    
    
    # #----------------------------------------------
    # ''' Cut '''
    # #----------------------------------------------
    # x1 = x[ :, 0:cutInds[0] ] 
    # x2 = x[ :, cutInds[1] : : ] 
    
    # ax.set_xlim( x1[:,0], x1[:,-1] )
    # ax2.set_xlim( x2[:,0], x2[:,-1] )
    
    
    
    
    
    # #----------------------------------------------
    # # hide the spines between ax and ax2
    # ax.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    # ax2.yaxis.tick_right()
    
    
    
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    # ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    # ax2.plot((-d, +d), (-d, +d), **kwargs)
    
    
    
    
    
    # return fig 













def plotLossTerms( costTermsDict, epochNumsToTest, colors=None ):
    
    
    if colors is None:
        colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]
    
    
    costTypes = list(  costTermsDict[ epochNumsToTest[0] ].keys()  )
    costLabels = { 'errLoss'    :   r'$\|  \mathbf{Dz}(t_i) - \mathbf{x}(t_i) \|_2^2$',
                   'hLoss'      :   r'$\|  \mathbf{Hz}(t_i) - \mathbf{z}(t_{i-1}) \|_2^2$',
                   'effLoss'    :   r'$\| \mathbf{z}(t_i)  \|_2$',
                   'sLoss'      :   r'$\|  \mathbf{z}(t_i) \|_1$',
                   }
    
    nBetas = costTermsDict[ epochNumsToTest[0] ][ costTypes[0] ].shape[0]
    
    
    
    
    
    
    for costType in costTypes:
        
        fig = plt.figure() 
        
        
        for epochInd in range( len(epochNumsToTest) ):        
            epochNum = epochNumsToTest[ epochInd ]
            
            
            if epochNum == 0:
                currLabel = 'Before training'
            else:
                currLabel = 'epoch ' + str(epochNum)
                
            currColor = colors[ epochInd ]
            
            
            currCostVals = np.array(   costTermsDict[ epochNum ][ costType ].detach()   )
            
            plt.plot( currCostVals, c=currColor )
            plt.scatter( list(range(nBetas)), currCostVals, label=currLabel, s=10, c=currColor )
            # currCostType = np.array(  []  )
            
            
        plt.title( costType )
        plt.legend(  )
        
        plt.xlabel( 'i' )
        plt.ylabel( costLabels[ costType ] )
    

    
    
    return fig 










def colorGradient( colormap, nColors, minColorLimit=0.35 ):
    ''' 
        Given a colormap and a number nColors, returns a dictionary of colors within that 
    colormap that form a gradient when plotted together. The returned colorDict uses simple
    numerical indexing as keys with corresponding RGB color entries. 
    
    ** Works best for sequential colormaps ** 
    
    '''
    
    # raise Exception( '[colorScheme] not yet defined' )
    
    if colormap is None:
        colormap = mpl.colormaps[ 'gist_rainbow' ]
        colormap = mpl.colormaps[ 'Reds' ]
        
        # colorDict = { 0 : 'red',
        #               1 : 'orange', 
        #               2 : 'gold',  
        #               3 : 'green',  
        #               4 : 'blue',  
        #               5 : 'purple',  
        #               6 : 'black',  
        #               }
    
    
    
    #--------------------------------------------------------
    ''' Grab the colors to use from the map '''
    #--------------------------------------------------------
    colorSpread = np.linspace(  minColorLimit, 1, nColors  )
    colors = colormap(  colorSpread  )
    
    
    #--------------------------------------------------------
    ''' Create the reference dictionary '''
    #--------------------------------------------------------
    colorDict = { }
    
    for i in range( nColors ):
        colorDict[ i ] = colors[ i ]
    
    
    return colorDict





def discreteColorbar( colorDict, fig, colorbarLabel=None ):
    '''  
        INPUT           TYPE        DESCRIPTION
        colorDict       dict        ( key, entry ) corresponds to ( label, color )
        
        OR 
        
        colorDict can be a list of colors which will be assigned labels 0, ..., len(colorsDict)
            
    '''
    
    
    #--------------------------------------------------------------
    if (type(colorDict) is list) or (type(colorDict) is np.ndarray):
        colors = colorDict
        
        colorDict = { }
        for k in range( len(colors) ):
            colorDict[ k ] = colors[k]
    #--------------------------------------------------------------
            
    colorLabels = list( colorDict.keys() )
    colors = [  colorDict[label] for label in colorLabels  ]
    nColors = len( colors )


    
    cmap =  mpl.colors.LinearSegmentedColormap.from_list( 'Custom cmap', colors, nColors )
    
        
    
    bounds = np.linspace( 0, 1, nColors+1 )             # Need a bound on either side (so:  nColors+1)

    boundMid = ( bounds[1] - bounds[0] )  /  2
    tickLocs = [  (b + boundMid) for b in bounds[0:-1]  ]
    

    ax2 = fig.add_axes( [0.95, 0.1, 0.03, 0.8] )
    
    cbar = mpl.colorbar.ColorbarBase(   ax2,
                                         # mappable, 
                                         cmap = cmap, 
                                         spacing='proportional',
                                         label=colorbarLabel
                                         )
    
    cbar.set_ticks( tickLocs )
    cbar.set_ticklabels( colorLabels )
    
    
    return cbar





# def discretizeColormap( nPts, cmap=mpl.colormaps['hsv'] ):
    
#     colors = cmap(  [ x/nPts for x in range(nPts) ]  )
    
    
    
    
    
#     return discreteColormap








def updateColorDictLabels( colorDict, colorLabels ):
    ''' Assumes one to one for original colorDict keys to new colorLabels'''
    
    origKeys = list(  colorDict.keys()  )
    
    
    newColorDict = { }
    nKeys = min( len(origKeys), len(colorLabels) )
    
    for i in range( nKeys ):
        newColorDict[ colorLabels[i] ] = colorDict[ origKeys[i] ]
        
        
    return newColorDict 






def plotConnMatNorms( trainingData, epochNumsToTest=None, colors=None ):
    '''  '''
    
    matNorms = trainingData[ 'matNorms' ]
    nMats = matNorms.shape[0]
    
    
    if nMats == 2:
        matNames = [ 'D', 'H' ]
    elif nMats == 3:
        matNames = [ 'W', 'Ms', 'Md' ]
    else: 
        raise Exception( '' )
    
    
    normFig, axs = plt.subplots( nMats, 1, sharex=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.2} )
    
    for i in range( nMats ):
        matNorm = matNorms[ i, : ]          # ( 1, nEpochs )
        axs[i].plot( matNorm )
                
        # labelStr = r'$\| {} \|_2$'.format( matNames[i] )
        labelStr = r'$\| {} \|_F$'.format( matNames[i] )
        axs[i].set_ylabel( labelStr )
        
        if epochNumsToTest is not None:
            for epochNum in epochNumsToTest:
                if colors is None:
                    color = 'gray'
                else: 
                    color = colors[  epochNumsToTest.index(epochNum)  ]
                axs[i].axvline( epochNum, c=color, linewidth=1 )
                
    axs[-1].set_xlabel( 'Training epoch' )
    
    # normFig.subplots_adjust( vspace = 0.5 )
    
    
    return normFig




def plotReconstruction( model, modelData, modelInput, colors=None,
                                   nBetasToPlot=6, printLegend=True, maxQ=2, 
                                   plotShifted=True ):
    
    

    epochNum = model.epochNum
    
    if epochNum is None:
        raise Exception( 'Cannot plot reconstruction performance for training model' )
    
    
    #---------------------------------------------------------------------------------
    ''' Colors '''
    #---------------------------------------------------------------------------------
    if colors is None:
        colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]
    
        
    if type( colors ) is list:
        
        colorDict = { }
        for i in range( len(colors) ):
            colorDict[ i ] = colors[i]
            
    else: 
        colorDict = colors
          
        
    colorDict = updateColorDictLabels(  colorDict,  list(range(maxQ+1))  )
    #---------------------------------------------------------------------------------
    
    
    
    #----------------------------------------
    ''' Model Info '''
    #----------------------------------------
    D = model.D  
    H = model.H
    states = modelData[ epochNum ][ 'state' ]               ## ( N, nTimes )
    
    d = model.signalDim
    
    
    # nEncodSteps_testing = modelData[ 'nEncodSteps_testing' ]
    nEncodSteps_testing = modelData[ 'nEncodSteps' ]
    nRetenSteps = modelData[ 'nRetenSteps' ]
    nTotalEvolSteps = nEncodSteps_testing + nRetenSteps
    
    
    
    # betaIndToStartWith = 1
    # nTimeIndsToSkip = nTotalEvolSteps * betaIndToStartWith
    
    # baseTimeIndsToPlot = range(  nTimeIndsToSkip,  nTimeIndsToPlot+nTimeIndsToSkip  )
    # baseTimeIndsToPlot = np.array(  [ baseTimeIndsToPlot ] * d  )
    
    # endTimeInd = nTimeIndsToSkip + nTimeIndsToPlot

    
    
    nTimeIndsToPlot = nTotalEvolSteps * nBetasToPlot
    
    
    
    #----------------------------------------
    ''' Initialize Figure '''
    #----------------------------------------
    fig, axs = plt.subplots( ) 
    # fig, axs = plt.subplots( 3, 1 ) 
    
    
    labels = [ ]
    handles = [ ]
        
    
    
    # baseTimeIndsToPlot = range(  betaIndToStartWith,  nTimeIndsToPlot  )
    baseTimeIndsToPlot = range(  nTimeIndsToPlot  )
    baseTimeIndsToPlot = np.array(  [ baseTimeIndsToPlot ] * d  )

    
        
    
    #-------------------------------------------------------------------------------------
    ''' State & reconstructed '''
    #-------------------------------------------------------------------------------------  
    for q in range( maxQ+1 ):
    
        Hq = torch.linalg.matrix_power( H, q )
        
        reconstructed = Hq @ states
        decodedRecon = D @ reconstructed
        decodedRecon = decodedRecon.detach().numpy()
        
        
        qColor = colorDict[ q ]
    
    
        if plotShifted:
            timeIndsToPlot = baseTimeIndsToPlot - nTotalEvolSteps 
            startInd = q * nTotalEvolSteps
            endInd = startInd +  nTimeIndsToPlot
        else: 
            timeIndsToPlot = baseTimeIndsToPlot
            startInd = 0
            endInd = nTimeIndsToPlot
            
            
        # decodedToPlot = decodedRecon[ :, 0:nTimeIndsToPlot ]
        decodedToPlot = decodedRecon[ :, startInd:endInd ]

    
        plt.scatter(  baseTimeIndsToPlot.T,  decodedToPlot.T, s=2, c=qColor )
        reconLine = plt.plot( decodedToPlot.T , linewidth=1,  c=qColor )
        
        
        
        # label = r'$\mathbf{DHr}(t)$'
        label = r'$\mathbf{D}$'
        if q > 0:
            label = label + r'$\mathbf{H}$'
            if q > 1:
                expon = r'$^{}$'.format( q )
                label = label + expon + r'$\mathbf{r}$(t)'
            else: 
                label = label + r'$\mathbf{r}$(t)'

        
        
        
        labels.append( label )
        handles.append( reconLine )

  

    
    
    #-----------------------------------------------------------------------------------------
    ''' Stimuli '''
    #-----------------------------------------------------------------------------------------
    stimMat = modelInput.stimMat 
    
    
    stimTimeInds = modelData[ 'stimTimeInds' ]
    
    
    
    for k in range( nBetasToPlot ):    
        plt.axvline(  stimTimeInds[k],  c='k',  linewidth=1  ) 
            
        
        # if plotShifted:
        #     stimDot = plt.scatter(  [ stimTimeInds[k]+nEncodSteps_testing ]*d,  stimMat[:,k], facecolors='none', edgecolors='k', s=40, linewidth=2  ) 
        # else:
        #     # stimDot = plt.scatter(  [ stimTimeInds[k] ]*d,  stimMat[:,k], facecolors='none', edgecolors='r', s=40, linewidth=2  ) 
        #     stimDot = plt.scatter(  [ stimTimeInds[k] ]*d,  stimMat[:,k], facecolors='none', edgecolors='k', s=40, linewidth=2  ) 
        stimDot = plt.scatter(  [ stimTimeInds[k] ]*d,  stimMat[:,k], facecolors='none', edgecolors='k', s=40, linewidth=2  ) 
    
    
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
        
        if nRetenSteps > 0: 
            plt.axvline(  delayStart,  c='gray',  linestyle='--',  linewidth=1  ) 
                
        refDot = plt.scatter(  [ delayEnd ]*d,  decodedRefs[:,k], facecolors='none', edgecolors='k', s=40, marker='s', linewidth=2  ) 
        # refDot = plt.scatter(  [ delayEnd ]*d,  decodedRefs[:,k], facecolors='none', edgecolors='orange', s=40, marker='s', linewidth=2  ) 
    
    
    
    labels.append( r'$\mathbf{r}^-(t)$' )
    handles.append( refDot )
    
    
    
    
    #-----------------------------------------------------------------------------------------
    ''' Put it all together '''
    #-----------------------------------------------------------------------------------------
    plt.suptitle( 'Reconstruction: epoch {}'.format( epochNum ) )
    
    
    if printLegend:
        handles.append( reconLine[0] )
        labels.append( r'$\mathbf{DH}^q\mathbf{r}(t)$' )
        plt.legend(  handles, labels  )
    
    
    plt.xlabel( 't' )
    plt.ylabel( r'$\mathbf{DH}^q\mathbf{r}(t)$' )
    
    
    
    cbar = discreteColorbar( colorDict, fig, colorbarLabel='q' )
    # plt.colorbar( )
    
    
    
        
    return fig 










#=========================================================================================
#=========================================================================================
#%% Saving
#=========================================================================================
#=========================================================================================


def saveFigureDict( figDict, saveDir=None, imgTypes=['.jpg', '.svg'] ): 


    #------------------------------------------------
    ''' Data formatting '''
    #------------------------------------------------
    if type(figDict) is not dict:
        
        if type(figDict) is list:
            figList = figDict
            figDict = { }
            for i in range( len(figList) ):
                figDict[ 'fig' + str(i) ] = figList[ i ]
                
        elif type(figDict) is matplotlib.figure.Figure:
            figDict = { 'fig' : figDict }
            
        else: 
            raise Exception( '[saveFigureDict] Cannot process given figDict; please use type dict, list, or fig' )
    
    
    figNames = list( figDict.keys() )
    figList = [ figDict[x] for x in figNames ]
    

    #------------------------------------------------
    ''' Where to save '''
    #------------------------------------------------
    if saveDir is None:
        saveDir = os.getcwd()
    
    if not os.path.exists( saveDir ):
        # os.mkdir( saveDir )
        os.makedirs( saveDir )
    

    
    
    #------------------------------------------------
    ''' Save the figures '''
    #------------------------------------------------
    filenameList = [ ]
    
    print( 'Saving to... ', saveDir  )
    
    
    for i in range( len(figList) ):
        
        figName = figNames[i]
        fig = figList[i]
        
        for imgType in imgTypes:
            # filename = figName + '.pickle'
            filename = figName + imgType
            filenameList.append( filename )
            print( '\t', filename )
            
            absLoc = os.path.join( saveDir, filename )
    
            fig.savefig( absLoc )
    
            # with open( absLoc, 'wb' ) as file:
            #     pickle.dump(  fig,  file,  protocol=pickle.HIGHEST_PROTOCOL  )
    
    
    
    
            # if (imgType == '.pkl') or (imgType == '.pickle'):
            #     with open( absLoc, 'wb' ) as f:
            #         pickle.dump(  fig,  f,  protocol=pickle.HIGHEST_PROTOCOL  )
            # else:
            #     fig.savefig( absLoc )
 
            


    return filenameList







# def nameSaveDir( simOptions, weightCombos, nEpochs, dataFolder='varsAndFigs' ):
def nameSaveDir( simOptions, dataFolder='varsAndFigs', includeTrainTime=True ):



    #-------------------------------------------------------------------------------------
    ''' maxIter '''
    #-------------------------------------------------------------------------------------
    if simOptions['learnSingleStimUntilConvergence']:
        # saveFolder = 'retrainOnStim' + str( simOptions['parameters']['maxIter'] )
        saveFolder = 'retrainOnStim' + str( simOptions['maxIter'] )
    else: 
        saveFolder = 'singleTrainOnStim'
        
    saveFolder = os.path.join( dataFolder, saveFolder )  
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Attention '''
    #-------------------------------------------------------------------------------------
    # alphaSaveStr = 'eAlp' + str( simOptions['encodingAlpha'] ) + 'rAlp' + str( simOptions['retentionAlpha'] ) + '_'
    # saveSubFolder = alphaSaveStr
    alphaSaveFolder = 'eAlp' + str( simOptions['encodingAlpha'] ) + 'rAlp' + str( simOptions['retentionAlpha'] )
    saveFolder = os.path.join( saveFolder, alphaSaveFolder )  
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Length of encoding and retention phases '''
    #-------------------------------------------------------------------------------------
    try: 
        # phaseSaveFolder = 'eSteps{}rSteps{}'.format(  simOptions['nEncodSteps_testing'],  simOptions['nRetenSteps']  )
        phaseSaveFolder = 'eSteps{}rSteps{}'.format(  simOptions['nEncodSteps'],  simOptions['nRetenSteps']  )
    except:
        try:
            phaseSaveFolder = 'eSteps{}rSteps{}'.format(  simOptions['nEncodSteps'],  simOptions['nRetenSteps']  )
        except: 
            phaseSaveFolder = 'eSteps{}rSteps{}'.format(  simOptions['nEncodSteps_testing'],  simOptions['parameters']['nRetenSteps']  )

            
    saveFolder = os.path.join( saveFolder, phaseSaveFolder )  

    


    # #-------------------------------------------------------------------------------------
    # '''  '''
    # #-------------------------------------------------------------------------------------
    # saveFolder = os.path.join( saveFolder, saveSubFolder )  
    # saveFolder = os.path.join( saveFolder, 'Epochs' + str(nEpochs) ) 


    
    #-------------------------------------------------------------------------------------
    '''  '''
    #-------------------------------------------------------------------------------------
    if includeTrainTime:
        saveSubFolder = 'train'
        
        if simOptions['trainBeforeDecay']:
            saveSubFolder = saveSubFolder + 'Before'
        if simOptions['trainAfterDecay']:
            saveSubFolder = saveSubFolder + 'After'
            
        saveSubFolder = saveSubFolder + 'Decay'
        
        if simOptions['learningRate'] != 0.005:
            saveSubFolder = saveSubFolder  +  '_lr'  +  str( simOptions['learningRate'] )

            
    
    
    else: 
    
        try:
            saveSubFolder = 'prox'  +  str( simOptions['nEncodSteps'] )
        except:
            saveSubFolder = 'prox'  +  str( simOptions['nEncodSteps_testing'] )
        # saveSubFolder = saveSubFolder + 'prox'  +  str( simOptions['parameters']['nEncodSteps_testing'] )
        
        
        if simOptions['requireStable']:
            saveSubFolder = saveSubFolder + '_stable'
            
            
        saveSubFolder = saveSubFolder  +  '_lr'  +  str( simOptions['learningRate'] )
        
        
        
        
        
    weightCombos = simOptions[ 'weightCombos' ]
    saveSubFolder = saveSubFolder  +  '_WC'  +  weightCombos[0].replace( '_', '-' )


    nEpochs = simOptions[ 'parameters' ][ 'nEpochs' ]

    saveFolder = os.path.join( saveFolder, saveSubFolder )  
    saveFolder = os.path.join( saveFolder, 'Epochs' + str(nEpochs) )  



    return saveFolder






def extractSimOptionsFromDir( saveDir, simOptions=None ):

    if simOptions is None:
        simOptions = { }
        
    
    # # simOptions['parameters']['maxIter'] = 
    # simOptions['maxIter'] = 
        
    # simOptions['encodingAlpha'] = 
    # simOptions['retentionAlpha'] = 
    
    # simOptions['nEncodSteps_testing'] = 
    # simOptions['parameters']['nRetenSteps'] = 
    
    # simOptions['requireStable'] = simOptions[ 'weightCombos' ]
    # simOptions['learningRate']




    return simOptions





# def delOldSubDirs( newDir, oldDir ):
    
    
#     return 



def copySaveDir( oldDir, dataFolder='varsAndFigs', newDir=None ): 
# def copySaveDir( oldDir, newDir=None, dataFolder='varsAndFigs' ): 
    
    
    #----------------------------------------------
    ''' Get the saved data '''
    #----------------------------------------------
    modelInfoDict = getModelInfo( oldDir )


    #----------------------------------------------
    ''' The new directory to save to '''
    #----------------------------------------------
    if newDir is None:
    
        if 'simOptions' in modelInfoDict.keys():
            simOptions = modelInfoDict[ 'simOptions' ]
        # else: 
        #     simOptions = extractSimOptionsFromDir( oldDir )
        
        newDir = nameSaveDir( simOptions, dataFolder=dataFolder )
        

    #----------------------------------------------
    ''' Resave under the new directory system '''
    #----------------------------------------------
    varNames = list( modelInfoDict.keys() )
    saveModelInfo(  modelInfoDict,  saveDir=newDir,  varNames=varNames  )
    
    
    
    return oldDir, newDir





def renameSaveDir( oldDir, dataFolder='varsAndFigs' ):
    
    
    [ oldDir, newDir ] = copySaveDir( oldDir, dataFolder )
    

    #----------------------------------------------
    ''' Delete the old dir. and its contents '''
    #----------------------------------------------
    # delOldSubDirs( newDir, oldDir )

    
    return 





def loadFigures( figFilenames, saveDir=None, imgTypes=['.pkl'] ):
    
    
    if saveDir is None:
        saveDir = os.getcwd()
    
    
    for figName in figFilenames:
        if '.' not in figName:
            figFilenames.remove( figName )
            for imgType in imgTypes:
                filename = figName + imgType
                figFilenames.append( filename )
            
    
    
    figDict = { }
    
    for filename in figFilenames:
        
        absLoc = os.path.join( saveDir, filename )
        
        fig = pickle.load( file(absLoc,'rb') )
        
        figName = filename[  0 : filename.index('.') ]
        figDict[ figName ] = fig
    
    
    return figDict





def convertSavedFig( figFilenames, saveDir=None, imgType1=['.pkl'], imgType2=['.jpg', '.svg'], delOldFile=False ):

    
    if saveDir is None:
        saveDir = os.getcwd()
        
    if type(figFilenames) is str:
        figFilenames = [ figFilenames ]
    
    
    #-------------------------------------------------------------------------------------
    ''' Convert the fig files '''
    #-------------------------------------------------------------------------------------
    for filename in figFilenames:
        
        if '.' in filename:
            figName = filename[  0 : filename.index('.') ]
        else:
            figName = filename
        
        print( figName )
        
        
        newFilenameList = [ ]
        # figDict = { }
        # 
        for type1 in imgType1:
            #------------------------------------------------
            ''' Open the file '''
            #------------------------------------------------
            filenameToOpen = figName + type1 
            absLoc = os.path.join( saveDir, filenameToOpen )
        
            print( '\t Opening ', filenameToOpen )
            
            file = open( absLoc, 'rb' )
            fig = pickle.load( file )
            
            # figDict[ figName ] = fig
            
            file.close()
            
            
            #------------------------------------------------
            ''' Resave the file '''
            #------------------------------------------------
            for type2 in imgType2:
                newFilename = figName + type2 
                newAbsLoc = os.path.join( saveDir, filenameToOpen )
                
                newFilenameList.append( newFilename )
                print( '\t\t Saving ', newFilename )


                with open( newAbsLoc, 'wb' ) as f:
                    pickle.dump(  fig,  f,  protocol=pickle.HIGHEST_PROTOCOL  )
     
                filenameList.append( newFilename )
                
                
            if delOldFile:
                os.remove( absLoc )



    return newFilenameList 






def saveModelInfo( localVars, saveDir=None, varNames=None, saveNames=None ):
    
    
    if saveDir is None:
        saveDir = os.getcwd()
    
    if not os.path.exists( saveDir ):
        # os.mkdir( saveDir )
        os.makedirs( saveDir )
    
    
    
    if varNames is None:
        varNames = [ 'trainingModel', 'trainingInput', 'trainingData', 
                        'epochModels', 'testingInput', 'testingData',
                        'simOptions' ]
    elif type(varNames) is str:
        varNames = [ varNames ]
        
    if saveNames is None:
        saveNames = varNames
    
    
    filenameList = [ ]
    
    if len(varNames) > 1:
        print( 'Saving to... ', saveDir  )
    
    
    # for varName in varNames:
    for i in range( len(varNames) ):
        
        varName = varNames[ i ]
        saveName = saveNames[ i ]
        
        var = localVars[ varName ]
        
        # filename = varName + '.pickle'
        filename = saveName + '.pickle'
        absLoc = os.path.join( saveDir, filename )
    
        with open( absLoc, 'wb' ) as f:
            pickle.dump(  var,  f,  protocol=pickle.HIGHEST_PROTOCOL  )
 
        filenameList.append( filename )
        
        
        if len(varNames) > 1:
            print( '\t', filename )
    
    
        
    return filenameList 








def turnDictIntoTupleList( dictionary ):
    
    
    listOfTuples = [ ]
    
    for key in dictionary.keys():
        
        dictEntry = dictionary[ key ]
        listEntry = tuple(  [ key, dictEntry ]  )
    
        listOfTuples.append( listEntry )
    
    
    return listOfTuples






def getModelInfo( saveDir=None, varNames=None, saveNames=None ):

    
    if varNames is None:
        varNames = [ 'trainingModel', 'trainingInput', 'trainingData', 
                        'epochModels', 'testingInput', 'testingData',
                        'simOptions' ]
    elif type(varNames) is str:
        varNames = [ varNames ]
        
    if saveNames is None:
        saveNames = varNames
    
    
    nVars = len( varNames )
    modelInfo = { }
    
    
    # for varName in varNames:
    for i in range( len(varNames) ):
        
        
        varName = varNames[ i ]
        saveName = saveNames[ i ]
        
        
        # filename = varName + '.pickle'
        filename = saveName + '.pickle'
        absLoc = os.path.join( saveDir, filename )
        
        with open( absLoc, 'rb' ) as f:
            loadedVar = pickle.load( f )

        if nVars > 1:
            modelInfo[ varName ] = loadedVar
        else:
            modelInfo = loadedVar



    return modelInfo




def initModelAsBefore( saveDir=None, varNames=None ):
    ''' 
        Load the saved data from a previous run, and reset variables we can run again with  
    the same info. 
    '''
    
    modelInfoDict = getModelInfo( saveDir, varNames )
    # varNames = list( modelInfoDict.keys() )

    #-----------------------------------------------------
    ''' Get and reset the training model '''
    #-----------------------------------------------------
    trainingModel = modelInfoDict[ 'trainingModel' ]
    trainingData = modelInfoDict[ 'trainingData' ]
    
    if 'W' in trainingData.keys():
        trainingModel.W = nn.Parameter(  trainingData[ 'W' ][ 0 ]  )
        trainingModel.Ms = nn.Parameter(  trainingData[ 'Ms' ][ 0 ]  )
        trainingModel.Md = nn.Parameter(  trainingData[ 'Md' ][ 0 ]  )
    
    elif 'D' in trainingData.keys():
        trainingModel.D = nn.Parameter(  trainingData[ 'D' ][ 0 ]  )
        trainingModel.H = nn.Parameter(  trainingData[ 'H' ][ 0 ]  )
    
    else: 
        raise Exception( '[initModelAsBefore] Could not correctly assign parameters' )
    
    
    #-----------------------------------------------------
    ''' Get the inputs '''
    #-----------------------------------------------------
    trainingInput = modelInfoDict[ 'trainingInput' ]
    testingInput = modelInfoDict[ 'testingInput' ]
    
    



    return trainingModel, trainingInput, testingInput






def checkLoadIn( localVars, model, inputSignal ):
    
    
    d = localVars[ 'd' ]
    N = localVars[ 'N' ]
    
    
    if model.signalDim != d:
        print( '[checkLoadIn] signalDim of loaded trainingModel does not agree with local defintion' )
        return False
    
    if inputSignal.signalDim != d:
        print( '[checkLoadIn] signalDim of loaded trainingInput does not agree with local defintion' )
        return False
    
        
    if model.networkDim != N:
        print( '[checkLoadIn] networkDim of loaded trainingModel does not agree with local defintion' )
        return False


    # if inputSignal.circleStim:
        
        
        # 
        
        

    
    # if model.epochNum is not None:
    #     if inputSignal.nBetas != localVars[ 'nTestBetas' ]:
    #         print( '[checkLoadIn] networkDim of loaded trainingModel does not agree with local defintion' )
    #         return False


    
     
    return True







#=========================================================================================
#=========================================================================================
#%% Mid-training Saving
#=========================================================================================
#=========================================================================================


def saveCurrentTrainingInfo( simOptions, varDict, midTrainDir='midTrainSave', currDir=None ):

    
    #----------------------------------------------------------------
    ''' Where to save '''
    #----------------------------------------------------------------
    saveFolder = nameSaveDir( simOptions, dataFolder='midTrainSave' )
    
    if currDir is None:
        saveDir = os.getcwd()
        
    saveDir = os.path.join( currDir, saveFolder ) 
    
    
    
    #----------------------------------------------------------------
    '''  '''
    #----------------------------------------------------------------
    varNames = list(  varDict.keys()  ) 
    
    
    filenameList = [ ]
    
    print( )
    print(  'Safety save:  saving to... ',  saveDir  )
    
    
    
    for varName in varNames:
        var = varDict[ varName ]
        
        filename = varName + '.pickle'
        absLoc = os.path.join( saveDir, filename )
    
        with open( absLoc, 'wb' ) as f:
            pickle.dump(  var,  f,  protocol=pickle.HIGHEST_PROTOCOL  )
 
        filenameList.append( filename )
        
        print( '\t', filename )
    
    
        
    return filenameList 





# def loadMidTrainData( simOptions ):
    
    
#     return 




#=========================================================================================
#=========================================================================================
#%% 
#=========================================================================================
#=========================================================================================


def myStatePCA( modelData, epochNum, k=2 ):
    '''  '''
    
    
    #----------------------------------------------------
    ''' Subtract Mean '''
    #----------------------------------------------------
    state = modelData[ epochNum ][ 'state' ]                ## ( N, nTimes )
    centeredData = centerStateData( state )                 ## ( N, nTimes )
     
    centeredData = centeredData.T                           ## To have variables be the columns and samples be rows 
    
    
    #----------------------------------------------------
    ''' Covariance Matrix '''
    #----------------------------------------------------
    covMat = centeredData.T @ centeredData                  ## ( N, N )
    
    
    #----------------------------------------------------
    ''' Eigenanalysis '''
    #----------------------------------------------------
    [ eigVals, eigVects ] = torch.linalg.eig( covMat )
    
    
    
    #----------------------------------------------------
    ''' Principal components '''
    #----------------------------------------------------
    kComponents = eigVects[ :, 0:k ]
    
    
    #----------------------------------------------------
    ''' Projection: Reduce data dimension '''
    #----------------------------------------------------
    projection = kComponents.T @ centeredData
    
    
    return projection





def centerStateData( state ):
    
    [ N, nTimes ] = state.shape
    
    means = torch.mean( state, dim=1 ).reshape( N,1 )       ## ( N,1 )
    meansTile = torch.tile( means, [1, nTimes] )
    
    
    centeredData = state - meansTile
    
    
    return centeredData 




def statePCA( model, modelData, modelInput ):
    
    
    
    
    
    
    return 








def tuningAnalysis( model, modelData, modelInput ):
    ''' Compare the input theta values to the model response  '''
    
    
    #-------------------------------------------------------
    ''' Get the original input '''
    #-------------------------------------------------------
    thetas = modelInput.thetas
    stimTimeInds = modelData[ 'stimTimeInds' ]
    nStims = thetas.shape[ -1 ]
    
    
    #-------------------------------------------------------
    ''' Get the model response '''
    #-------------------------------------------------------
    epochNum = model.epochNum 
    
    immedResponse = modelData[ epochNum ][ 'state' ][ :, stimTimeInds ]                 ## ( N, nStims )
    
    convergedEnc = modelData[ epochNum ][ 'convergedStates_encoding' ][ :, 1:: ]        ## ( N, nStims )
    convergedRet = modelData[ epochNum ][ 'convergedStates_retention' ][ :, 1:: ]       ## ( N, nStims )
    
    
    #-------------------------------------------------------
    ''' Pair slot responses to theta values '''
    #------------------------------------------------------- 
    N = model.networkDim
    
    
    immedPairs = [  ]     
    encPairs = [  ]     
    retPairs = [  ]     
    
    
    immedPairs = np.zeros(  [ N, nStims ]  )
    encPairs = np.zeros(  [ N, nStims ]  )
    retPairs = np.zeros(  [ N, nStims ]  )
    
    
    
    thetaList = thetas[0].numpy()
    degrees = ( thetas[0]  *  (180 / math.pi) ).numpy()             # ( nStims, 1 )

    








    
    for slotInd in range( N ):
        
        # immedPairs_slot = zip(  degrees,  immedResponse[slotInd].numpy()  )     # ( 1, nStims )
                
        # immedPairs.append(  immedPairs_slot  )
        # immedPairs = immedPairs + list( immedPairs_slot )
        immedPairs[ slotInd ] = immedResponse[slotInd].numpy()
        
        
        # encPairs_slot = zip(  degrees,  convergedEnc[slotInd].numpy()  )        # ( 1, nStims )
        # encPairs.append(  encPairs_slot  )
        # encPairs = encPairs + list( encPairs_slot )
        encPairs[ slotInd ] = convergedEnc[slotInd].numpy()


        # retPairs_slot = zip(  degrees,  convergedRet[slotInd].numpy()  )        # ( 1, nStims )
        # retPairs.append(  retPairs_slot  )
        # retPairs = retPairs + list( retPairs_slot )
        retPairs[ slotInd ] = convergedRet[slotInd].numpy()



    
    
    # # for stimInd in range( nStims ): 
        
    # #     currStim = thetas[0][ stimInd ]  *  (180 / math.pi)

    # #     immedPairs_slot = zip(  currStim,  immedResponse[ :, stimInd ]  )
    # #     encPairs_slot = zip(  currStim,  convergedEnc[ :, stimInd ]  )
    # #     retPairs_slot = zip(  currStim,  convergedRet[ :, stimInd ]  )
        
        
        
        
        
    
        
    #     # for slotInd in range( N ):
    #     #     slotResponse_immed = immedResponse[ :, stimInd ]
    #     #     immedPairs.append(   tuple( [currStim, slotResponse_immed] )   )
            
    #     #     slotResponse_enc = convergedEnc[ :, stimInd ]
    #     #     encPairs.append(   tuple( [currStim, slotResponse_enc] )   )
            
    #     #     slotResponse_ret = convergedRet[ :, stimInd ]
    #     #     retPairs.append(   tuple( [currStim, slotResponse_ret] )   ) 
            
    
    
    
    
    #-------------------------------------------------------
    ''' Compare '''
    #------------------------------------------------------- 
    
    # immedPts = zip( *immedPairs )
    fig1, ax1 = plt.subplots(  )
    # plt.scatter(   *zip( *immedPairs )   )
    plt.imshow(  immedPairs  )
    plt.colorbar( ) 
    plt.title( 'Tuning: immediate' )
    
    
    fig2, ax2 = plt.subplots(  )
    # ax2.scatter(   *zip( *encPairs )   )
    plt.imshow(  encPairs  )
    plt.colorbar( ) 
    plt.title( 'Tuning: converged' )

    
    fig3, ax3 = plt.subplots(  )
    # ax2.scatter(   *zip( *retPairs )   )
    plt.imshow(  retPairs  )
    plt.colorbar( ) 
    plt.title( 'Tuning: post-delay' )

    



    # for slotInd in range(N):
        
    #     ax1.scatter(   list(degrees),  list(range(nStims)),  c=immedResponse[ slotInd ].numpy()   )
    #     # ax2.scatter(   list(range(nStims)),  list(range(N)),  c=encPairs[ slotInd ]   )
    #     # ax3.scatter(   list(range(nStims)),  list(range(N)),  c=retPairs[ slotInd ]   )
 

    


    # for slotInd in range(N):
    #     ax1.scatter(   list(range(nStims)),  list(range(N)),  c=immedPairs[ slotInd ]   )
    #     ax2.scatter(   list(range(nStims)),  list(range(N)),  c=encPairs[ slotInd ]   )
    #     ax3.scatter(   list(range(nStims)),  list(range(N)),  c=retPairs[ slotInd ]   )
 


    return fig1, fig2, fig3
    
    
    
    # #-------------------------------------------------------
    # ''' Compare '''
    # #------------------------------------------------------- 
    
    # # stimRep = thetas.reshape( nStims, 1 ).repeat( 1, nStims )
    # stimRep = thetas.reshape( 1, nStims ).repeat( N, 1 ) 
    
    
    # immedCorr = np.corrcoef( stimRep, immedResponse )
    # encCorr = np.corrcoef( stimRep, convergedEnc )
    # retCorr = np.corrcoef( stimRep, convergedRet )

    # stimCorr = np.corrcoef( stimRep )
    
    
    # corrDict = {    'immedCorr' : immedCorr[ N::, N:: ],
    #                 'encCorr'   : encCorr[ N::, N:: ], 
    #                 'retCorr'   : retCorr[ N::, N:: ], 
    #                 'stimCorr'  : stimCorr,    }
    
    
    
    
    # #-------------------------------------------------------
    # ''' Plot '''
    # #------------------------------------------------------- 
    # corrFigDict = { }
    
    # for key in corrDict.keys():
    #     fig, ax = plt.subplots( )
        
    #     plt.imshow(  corrDict[key], vmin=-1, vmax=1  )
        
    #     # CS = ax.contourf(x, y, Z.T, levels=np.linspace(0.9,1.1,51), 
    #     #          vmin=0.9, vmax=1.1, cmap="RdBu", extend='both')
    #     # cb0 = fig.colorbar(CS, ax=ax, ticks=[0.9, 1.0, 1.1], extend="both")
        
    #     cbar = plt.colorbar( )
    #     # cbar.set_clim( -1, 1 )
        
    #     plt.title( key )
        
    #     corrFigDict[ key ] = fig
    
    
    
    
    
    # degrees = thetas  *  (180 / math.pi)
    # plt.plot( degrees,  )
    
    
    
    
    
    
    
    
    
    
    # # return immedCorr, encCorr, retCorr, stimCorr
    # return corrDict, corrFigDict







#%%  



def stateSubplot( model, modelData, modelInput ):

    
    epochNum = model.epochNum
    N = model.networkDim
    
    
    nBetasToPlot = 10 
    nEncodSteps_testing = modelData['parameters']['nEncodSteps_testing']
    nRetenSteps = modelData['parameters']['nRetenSteps']
    nTimeIndsToPlot = (nEncodSteps_testing + nRetenSteps) * nBetasToPlot
    
    
    stimTimeInds = modelInput[ 'stimTimeInds' ]
    
    modelState = modelInput[epochNum]['state']
    
    
    
    fig, axs = plt.subplots( N, 1 ) 
    
    for i in range( N ):
        axs[i].axis('off')
        
        axs[i].plot( modelState[i,0:nTimeIndsToPlot], c='red' ) 
        
        
        for k in range( nBetasToPlot ):
            axs[i].axvline( stimTimeInds[k], c='k', linewidth=1 )         
            axs[i].axvline( stimTimeInds[k]+nRetenSteps, c='gray', linewidth=1 ) 
            
    plt.suptitle( 'State response' ) 
    
    
    return fig 









# def maxIterPlot( ):
    
    
#     return fig 








#%%  06/18/24





def subnetworkTuning( model, testingData, testingInput, trainingInput=None, q=0,
                             cmap=mpl.colormaps['seismic'], retention=False ):

    
    
    converged = True
    
    
    

    #-------------------------------------------------------------------------------------
    ''' Get the model data '''
    #-------------------------------------------------------------------------------------
    epochNum = model.epochNum
    N = model.networkDim
    
    nTestBetas = testingInput.nBetas



    # if not converged: 
    #     nEncodSteps = testingData['nEncodSteps']
    #     nRetenSteps = testingData['nRetenSteps']
    #     stimTimeInds = testingData['stimTimeInds']
    #     [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps, nRetenSteps, stimTimeInds )


    
    if retention:        
        statesToAnalyze = testingData[ epochNum ][ 'convergedStates_retention' ].detach().numpy()
        # if converged:
        #     statesToAnalyze = testingData[ epochNum ][ 'convergedStates_retention' ].detach().numpy()
        # else:
        #     statesToAnalyze = testingData[ epochNum ][ 'state' ][ :, retentionTimeInds ].detach().numpy()
        #     nSteps = nRetenSteps

    else: 
        statesToAnalyze = testingData[ epochNum ][ 'convergedStates_encoding' ].detach().numpy()
        # if converged: 
        #     statesToAnalyze = testingData[ epochNum ][ 'convergedStates_encoding' ].detach().numpy()
        # else: 
        #     statesToAnalyze = testingData[ epochNum ][ 'state' ][ :, encodingTimeInds ].detach().numpy()
        #     nSteps = nEncodSteps

    
    
    
    
    # if q > 1:
    if q > 0:
        H = model.H.detach().numpy()
        Hq = np.linalg.matrix_power( H, q )
        statesToAnalyze = Hq @ statesToAnalyze                  # ( N, nStims )
    
    
    #---------------------------------------------
    ''' Remove the inital state '''
    #---------------------------------------------
    if converged:
        statesToAnalyze = statesToAnalyze[ :, 1:: ]
    nStates = statesToAnalyze.shape[ 1 ]

    
    # print( statesToAnalyze.shape )
    
    #------------------------------------------------
    ''' Baseline (average) value for each subnet '''
    #------------------------------------------------
    subnetMeans = np.mean( statesToAnalyze, axis=1 )                        ## ( N, 1 )
    subnetMeans_tile = np.tile( subnetMeans.reshape(N,1), [1,nStates] )  ## ( N, nStates )
    
    subnetStd = np.std( statesToAnalyze, axis=1 )                           ## ( N, 1 )
    subnetStd_tile = np.tile( subnetStd.reshape(N,1), [1,nStates] )      ## ( N, nStates )
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Set up the figure '''
    #-------------------------------------------------------------------------------------
    if trainingInput is None:
        fig, axs = plt.subplots( 2, 1, height_ratios=[4,1] )
    else: 
        fig, axs = plt.subplots( 3, 1, height_ratios=[4,1,1] )
        
        
    # # norm = mpl.colors.Normalize( vmin=0, vmax=1 )
    # norm = mpl.colors.Normalize( vmin=-1, vmax=1 )
    # mappable = mpl.cm.ScalarMappable( norm=norm, cmap=cmap )
    
    
    
    # subnetAbsMaxs = np.max( abs(statesToAnalyze), axis=1 )                  # ( N, 1 )
    # subnetMins = np.min( statesToAnalyze, axis=1 )                  # ( N, 1 )
    
    
    #-------------------------------------------------------------------------------------
    ''' Set up the colors '''
    #-------------------------------------------------------------------------------------
    colorInds = statesToAnalyze
    
    convergedMax = np.max( np.max( abs(statesToAnalyze) ) )
    

    
    subnetDiffFromMeans = statesToAnalyze - subnetMeans_tile                ## ( N, nStates )
    normalizedSubnets = subnetDiffFromMeans / subnetStd_tile                ## ( N, nStates ) 
    absMaxNormSubnets = np.max(  np.max( abs(normalizedSubnets) )  )
    norm = mpl.colors.Normalize(   vmin = -absMaxNormSubnets,   vmax = absMaxNormSubnets   )
    
    
    # # norm = mpl.colors.Normalize( vmin=0, vmax=1 )
    # norm = mpl.colors.Normalize( vmin=-1, vmax=1 )
    mappable = mpl.cm.ScalarMappable( norm=norm, cmap=cmap )
    
    
    
    # thetas = testingInput.thetas[0]
    # sortedStates = sortStatesByThetas( statesToAnalyze, thetas )
    
    
    
    
    for k in range( nTestBetas ):     
        
        #------------------------------------------------
        ''' The (x,y) coordinates '''
        #------------------------------------------------
        currRad = testingInput.thetas[0][ k ]
        currDeg = currRad  *  ( 180 / math.pi )
        xVals = [ currDeg.detach().numpy() ] * N            ## current degree value
        
        yVals = list(  range( 1, N+1 )  )                   ## subnet indices
    
        
        #------------------------------------------------
        ''' Set the color according to the value '''
        #------------------------------------------------
        if converged: 
            
            currState = statesToAnalyze[ :, k ]                 ## subnet values for this beta_k     
            
            diffFromMean = currState - subnetMeans
            colorInd =  diffFromMean / subnetStd                ## Standardize the data 
            currColors = cmap( colorInd )
    
            axs[0].scatter( xVals, yVals, s=5, c=currColors )
            
            
        else: 
            stateInds = list(  range( k*nSteps,  (k+1)*nSteps )  )    
            # print( k )
            # print( stateInds )
            
            for i in range(nSteps):
                currState = statesToAnalyze[ :, stateInds[i] ]                 ## subnet values for this beta_k     
                
                diffFromMean = currState - subnetMeans
                colorInd =  diffFromMean / subnetStd                ## Standardize the data 
                currColors = cmap( colorInd )
        
                axs[0].scatter( xVals, yVals, s=5, c=currColors )
            
            
        # #------------------------------------------------
        # ''' Set the color according to the value '''
        # #------------------------------------------------
        # # converged = statesToAnalyze[ :, k+1 ]               ##  +1 to account for initState  
        
        # # else: 
        # #     inds = 
        # #     currState = statesToAnalyze[ :, k ]                 ## subnet values for this beta_k     

        # # colorInd = currState / np.max( abs(colorInd)        
        # # colorInd = currState / subnetMeans                  ## Normalize 
        
        # diffFromMean = currState - subnetMeans
        # colorInd =  diffFromMean / subnetStd                ## Standardize the data 
        # currColors = cmap( colorInd )

        # axs[0].scatter( xVals, yVals, s=5, c=currColors )



        # converged = statesToAnalyze[ :, k ]       
        # convergedNorm = np.linalg.norm( converged )
        
        # colorInd = converged
        # colorInd = colorInd / np.max( abs(colorInd) )                       ## Normalize 
        # color = cmap( colorInd )
        
        # axs[0].scatter( xVals, yVals, s=5, c=color )
    
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Put it all together  '''
    #-------------------------------------------------------------------------------------
        
    degreeTicks = list(range(0,361,60))
    
    
    axs[0].set_ylabel( 'Subnetwork ind' )       
    axs[0].get_xaxis().set_ticks( degreeTicks )      
    axs[0].get_xaxis().set_ticklabels( [] )      
    
    
    
    testingRads = testingInput.thetas
    testingThetas = testingRads  *  ( 180 / math.pi ) 
    axs[1].hist( testingThetas, bins=90 )
    axs[1].set_ylabel(  'Testing \n frequency'  )
    
    # axs[1].set_xlabel( r'$\theta$' )
    # axs[1].get_xaxis().set_ticks( degreeTicks )
    
    
    if trainingInput is not None:
        
        trainingRads = trainingInput.thetas 
        trainingThetas = trainingRads * (180/math.pi)
        axs[2].hist( trainingThetas, bins=90 )
        axs[2].set_ylabel(  'Training \n frequency'  )
        
        
        axs[1].get_xaxis().set_ticks( degreeTicks )
        axs[1].get_xaxis().set_ticklabels( [] )
        # axs[1].get_xaxis().set_ticks( [] )
        
        axs[2].set_xlabel( r'$\theta$' )
        axs[2].get_xaxis().set_ticks( degreeTicks )
        
    else: 
        
        axs[1].set_xlabel( r'$\theta$' )
        axs[1].set_ylabel(  'Frequency'  )
        axs[1].get_xaxis().set_ticks( degreeTicks )
    
    
    
    
    #------------------------------------------------
    ''' Colorbar '''
    #------------------------------------------------
    fig.subplots_adjust( right=0.8 )
    # cbar_ax = fig.add_axes(  [0.85, 0.15, 0.05, 0.7]  )
    cbar_ax = fig.add_axes(  [0.85, 0.1, 0.05, 0.8]  )
    # cbarLabel = 'Normalized difference from subnet mean'
    cbarLabel = 'Standardized subnet values'
    plt.colorbar( mappable, cax=cbar_ax, label=cbarLabel )
        
    
        
    if q > 0:
        titleStr = 'Subnetwork Tuning (q={}): epoch {}'.format(q, epochNum)
    else: 
        titleStr = 'Subnetwork Tuning: epoch {}'.format(epochNum)
        # plt.suptitle( 'Subnetwork Tuning: epoch {}'.format(epochNum) )
    
    if retention: 
        titleStr = 'Retention ' + titleStr
    
    
    plt.suptitle( titleStr )
    
    return fig 






# def retentionPeriodTuning( model, testingData, testingInput, trainingInput=None, q=1,
#                              cmap=mpl.colormaps['seismic'] ):

    

#     #-------------------------------------------------------------------------------------
#     ''' Get the model data '''
#     #-------------------------------------------------------------------------------------
#     epochNum = model.epochNum
#     N = model.networkDim
    
#     nTestBetas = testingInput.nBetas



#     nEncodSteps = testingData['nEncodSteps']
#     nRetenSteps = testingData['nRetenSteps']
#     stimTimeInds = testingData['stimTimeInds']
#     [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps, nRetenSteps, stimTimeInds )


#     statesToAnalyze = testingData[ epochNum ][ 'state' ][ :, retentionTimeInds ].detach().numpy()
#     nSteps = nRetenSteps

    
    
    
    
#     if q > 1:
#         H = model.H.detach().numpy()
#         Hq = np.linalg.matrix_power( H, q )
#         statesToAnalyze = Hq @ statesToAnalyze                  # ( N, nStims )
    
    
#     #---------------------------------------------
#     ''' Remove the inital state '''
#     #---------------------------------------------
#     nStates = statesToAnalyze.shape[ 1 ]

    
#     # print( statesToAnalyze.shape )
    
#     #------------------------------------------------
#     ''' Baseline (average) value for each subnet '''
#     #------------------------------------------------
#     subnetMeans = np.mean( statesToAnalyze, axis=1 )                        ## ( N, 1 )
#     subnetMeans_tile = np.tile( subnetMeans.reshape(N,1), [1,nStates] )  ## ( N, nStates )
    
#     subnetStd = np.std( statesToAnalyze, axis=1 )                           ## ( N, 1 )
#     subnetStd_tile = np.tile( subnetStd.reshape(N,1), [1,nStates] )      ## ( N, nStates )
    
    
    
#     #-------------------------------------------------------------------------------------
#     ''' Set up the figure '''
#     #-------------------------------------------------------------------------------------
#     if trainingInput is None:
#         fig, axs = plt.subplots( 2, 1, height_ratios=[4,1] )
#     else: 
#         fig, axs = plt.subplots( 3, 1, height_ratios=[4,1,1] )
        
        
#     # # norm = mpl.colors.Normalize( vmin=0, vmax=1 )
#     # norm = mpl.colors.Normalize( vmin=-1, vmax=1 )
#     # mappable = mpl.cm.ScalarMappable( norm=norm, cmap=cmap )
    
    
    
#     # subnetAbsMaxs = np.max( abs(statesToAnalyze), axis=1 )                  # ( N, 1 )
#     # subnetMins = np.min( statesToAnalyze, axis=1 )                  # ( N, 1 )
    
    
#     #-------------------------------------------------------------------------------------
#     ''' Set up the colors '''
#     #-------------------------------------------------------------------------------------
#     colorInds = statesToAnalyze
    
#     convergedMax = np.max( np.max( abs(statesToAnalyze) ) )
    

    
#     subnetDiffFromMeans = statesToAnalyze - subnetMeans_tile                ## ( N, nStates )
#     normalizedSubnets = subnetDiffFromMeans / subnetStd_tile                ## ( N, nStates ) 
#     absMaxNormSubnets = np.max(  np.max( abs(normalizedSubnets) )  )
#     norm = mpl.colors.Normalize(   vmin = -absMaxNormSubnets,   vmax = absMaxNormSubnets   )
    
    
#     # # norm = mpl.colors.Normalize( vmin=0, vmax=1 )
#     # norm = mpl.colors.Normalize( vmin=-1, vmax=1 )
#     mappable = mpl.cm.ScalarMappable( norm=norm, cmap=cmap )
    
    
    
#     # thetas = testingInput.thetas[0]
#     # sortedStates = sortStatesByThetas( statesToAnalyze, thetas )
    
    
#     #------------------------------------------------
#     ''' The (x,y) coordinates '''
#     #------------------------------------------------
#     xVals = [ nSteps ] * N                              ## steps during retention
#     yVals = list(  range( 1, N+1 )  )                   ## subnet indices

    
#     for k in range( nTestBetas ):     
        
    
#         #------------------------------------------------
#         ''' Set the color according to the value '''
#         #------------------------------------------------
#         stateInds = list(  range( k*nSteps,  (k+1)*nSteps )  )   
        
        
#         for i in range(nSteps):
            
#             currState = statesToAnalyze[ :, stateInds[i] ]                 ## subnet values for this beta_k     
            
#             diffFromMean = currState - subnetMeans
#             colorInd =  diffFromMean / subnetStd                ## Standardize the data 
#             currColors = cmap( colorInd )
    
#             axs[0].scatter( xVals, yVals, s=5, c=currColors )
            
            
#         # #------------------------------------------------
#         # ''' Set the color according to the value '''
#         # #------------------------------------------------
#         # # converged = statesToAnalyze[ :, k+1 ]               ##  +1 to account for initState  
        
#         # # else: 
#         # #     inds = 
#         # #     currState = statesToAnalyze[ :, k ]                 ## subnet values for this beta_k     

#         # # colorInd = currState / np.max( abs(colorInd)        
#         # # colorInd = currState / subnetMeans                  ## Normalize 
        
#         # diffFromMean = currState - subnetMeans
#         # colorInd =  diffFromMean / subnetStd                ## Standardize the data 
#         # currColors = cmap( colorInd )

#         # axs[0].scatter( xVals, yVals, s=5, c=currColors )



#         # converged = statesToAnalyze[ :, k ]       
#         # convergedNorm = np.linalg.norm( converged )
        
#         # colorInd = converged
#         # colorInd = colorInd / np.max( abs(colorInd) )                       ## Normalize 
#         # color = cmap( colorInd )
        
#         # axs[0].scatter( xVals, yVals, s=5, c=color )
    
    
    
    
#     #-------------------------------------------------------------------------------------
#     ''' Put it all together  '''
#     #-------------------------------------------------------------------------------------
        
#     degreeTicks = list(range(0,361,60))
    
    
#     axs[0].set_ylabel( 'Subnetwork ind' )       
#     axs[0].get_xaxis().set_ticks( degreeTicks )      
#     axs[0].get_xaxis().set_ticklabels( [] )      
    
    
    
#     testingRads = testingInput.thetas
#     testingThetas = testingRads  *  ( 180 / math.pi ) 
#     axs[1].hist( testingThetas, bins=90 )
#     axs[1].set_ylabel(  'Testing \n frequency'  )
    
#     # axs[1].set_xlabel( r'$\theta$' )
#     # axs[1].get_xaxis().set_ticks( degreeTicks )
    
    
#     if trainingInput is not None:
        
#         trainingRads = trainingInput.thetas 
#         trainingThetas = trainingRads * (180/math.pi)
#         axs[2].hist( trainingThetas, bins=90 )
#         axs[2].set_ylabel(  'Training \n frequency'  )
        
        
#         axs[1].get_xaxis().set_ticks( degreeTicks )
#         axs[1].get_xaxis().set_ticklabels( [] )
#         # axs[1].get_xaxis().set_ticks( [] )
        
#         axs[2].set_xlabel( r'$\theta$' )
#         axs[2].get_xaxis().set_ticks( degreeTicks )
        
#     else: 
        
#         axs[1].set_xlabel( r'$\theta$' )
#         axs[1].set_ylabel(  'Frequency'  )
#         axs[1].get_xaxis().set_ticks( degreeTicks )
    
    
    
    
#     #------------------------------------------------
#     ''' Colorbar '''
#     #------------------------------------------------
#     fig.subplots_adjust( right=0.8 )
#     # cbar_ax = fig.add_axes(  [0.85, 0.15, 0.05, 0.7]  )
#     cbar_ax = fig.add_axes(  [0.85, 0.1, 0.05, 0.8]  )
#     # cbarLabel = 'Normalized difference from subnet mean'
#     cbarLabel = 'Standardized subnet values'
#     plt.colorbar( mappable, cax=cbar_ax, label=cbarLabel )
        
    
        
#     if q > 1:
#         titleStr = 'Subnetwork Tuning (q={}): epoch {}'.format(q, epochNum)
#     else: 
#         titleStr = 'Subnetwork Tuning: epoch {}'.format(epochNum)
#         # plt.suptitle( 'Subnetwork Tuning: epoch {}'.format(epochNum) )
    
#     if retention: 
#         titleStr = 'Retention ' + titleStr
    
    
#     plt.suptitle( titleStr )
    
#     return fig 







def slotOverlap( x, y ):
    
    
    numerator = abs(x).T  @  abs(y) 
    
    normX = torch.linalg.norm( x, ord=2 )
    normY = torch.linalg.norm( y, ord=2 )
    denom = normX * normY
    
    
    if denom == 0:
        return NaN 
    else: 
        R = numerator / denom
    
    
    return R 





def sortStatesByThetas( stateMat, thetas ):
    ''' Sort the columns of given stateMat by values in theta (ascending order)  '''
    
    
    idx = thetas.argsort()
    sortedStateMat = stateMat[ :, idx ]
    

    return sortedStateMat






def delayPeriodTraj( model, modelData, modelInput ):
    
    fig, axs = plt.subplots( )
    
    
    nEncodSteps = testingData['nEncodSteps']
    stimTimeInds = testingData['stimTimeInds']
    # delayStartTimeInds = [  stimTimeInds[i] + nEncodSteps for   ]
    # delayTimeInds = 
    
    
    
    
    
    
    return fig 




def spectralAnalysis( epochModels ):
    
    
    epochNumsToTest = list( epochModels.keys() )
    
    Hs = [ epochModels[epochNum].H for epochNum in epochNumsToTest ]
    eigvals = [  torch.linalg.eig( H )[0] for H in Hs  ]
    
    
    
    fig, axs = plt.subplots() 
    
    
    return fig 








def xyAccuracy( model, modelData, modelInput ):
    
    #-------------------------------------------------------------------------------------
    ''' Get the model data '''
    #-------------------------------------------------------------------------------------
    epochNum = model.epochNum
    # N = model.networkDim
    
    # state = testingData[ epochNum ][ 'state' ].detach().numpy()

    
    if retention:
        statesToAnalyze = testingData[ epochNum ][ 'convergedStates_retention' ].detach().numpy()
    else: 
        statesToAnalyze = testingData[ epochNum ][ 'convergedStates_encoding' ].detach().numpy()
    
    
    
    if q > 1:
        H = model.H.detach().numpy()
        Hq = np.linalg.matrix_power( H, q )
        statesToAnalyze = Hq @ statesToAnalyze
    # else: 
    #     statesToAnalyze = convergedStates
    
    
     

    #------------------------------------------------
    ''' The (x,y) coordinates '''
    #------------------------------------------------
    trueRadians = modelInput.thetas[0][ k ]
    trueDegrees = currRad  *  ( 180 / math.pi )

    encodedDegrees = statesToAnalyze 
    
    
    
    
    
    
    
    
    return fig 




# #%%

# state = testingData[ epochNum ][ 'state' ]
# A = state.T

# [ proj, fig ] = plotPCA( A, k=2 )


# [ U, S, V ] = torch.pca_lowrank( A )
# projStims = projection = torch.matmul(  A,  V[:, :k]  )

# # projStims = 
# # fig.get_axes()[0].scatter(  x=testingInput.stimMat[0],  y=testingInput.stimMat[1], c='red', marker='*'  );






#=========================================================================================
#=========================================================================================
#%% PCA 
#=========================================================================================
#=========================================================================================


def circleAccuracy( model, modelData, modelInput, trim=False, trimLen=20 ):
    
    epochNum = model.epochNum



    #-------------------------------------------------------------------------------------
    ''' Original input '''
    #-------------------------------------------------------------------------------------
    thetas = modelInput.thetas.numpy()                ## radians '
    degrees = np.rad2deg( thetas[0] )

    stimMat = modelInput.stimMat.numpy()              ## ( x, y )  =  ( cos(theta), sin(theta) )
    
    if trim:
        thetas_trim = thetas[0][ 0:trimLen ]
        stimMat_trim = stimMat[ :, 0:trimLen ]
    
    
    #-------------------------------------------------------------------------------------
    ''' Model response '''
    #-------------------------------------------------------------------------------------
    D = model.D.detach().numpy()
    
    converged = modelData[ nEpochs ][ 'convergedStates_encoding' ][ :, 1:: ].detach().numpy()
    decoded = D @ converged
    
    
    convergedDegrees = coorsToTrueDeg( converged[0], converged[1] ) 
    convergedDegrees = convergedDegrees.numpy()


    if trim:
        converged_trim = converged[ :, 0:trimLen ] 
        # decoded_trim = D @ converged_trim
        decoded_trim = decoded[ :, 0:trimLen ]
    

    
    #-------------------------------------------------------------------------------------
    ''' (x,y) Coordinates '''
    #-------------------------------------------------------------------------------------
    
    fig = plt.figure(  )
    if trim:
        groundTruth = plt.scatter( stimMat_trim[0],  stimMat_trim[1], c='green', label=r'$\mathbf{u}(t)$', s=5 )
        model = plt.scatter( decoded_trim[0],  decoded_trim[1], c='black', label=r'$\mathbf{Dz}(t)$', s=5 ) 
    else:
        groundTruth = plt.scatter( stimMat[0],  stimMat[1], c='green', label=r'$\mathbf{u}(t)$', s=5 )
        model = plt.scatter( decoded[0],  decoded[1], c='black', label=r'$\mathbf{Dz}(t)$', s=5 ) 
    
    plt.legend( )
#% %
    
    
    #-------------------------------------------------------------------------------------
    ''' Degrees '''
    #-------------------------------------------------------------------------------------
    # degrees_trim = np.rad2deg( thetas_trim )
    # convergedDegrees_trim = coorsToTrueDeg( converged_trim[0], converged_trim[1] ) 
    # convergedDegrees_trim = convergedDegrees_trim.numpy() 
    
    fig = plt.figure(  )
    # plt.scatter( degrees_trim, convergedDegrees_trim )
    plt.scatter( degrees, convergedDegrees, s=5 )
    plt.xlabel( 'Ground truth' )
    plt.ylabel( 'Network' )
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Degrees, but in color, plotted with (x,y) '''
    #-------------------------------------------------------------------------------------
    # ## verify that the theta calulation is correct 
    # fig = plt.figure(  )

    # plt.scatter( stimMat[0],  stimMat[1], c=cmap(degrees/360), s=5 )
    # cbar = plt.colorbar( ticks  =  np.linspace( 0, 1, 7 )  )
    # cbar.set_ticklabels(  [ int(x) for x in np.linspace( 0, 360, 7 ) ]  )
    
    
    
    fig = plt.figure(  )
    
    plt.scatter( converged[0],  converged[1], c=cmap(degrees/360), s=5 )
    
    cbar = plt.colorbar( ticks  =  np.linspace( 0, 1, 7 )  )
    cbar.set_ticklabels(  [ int(x) for x in np.linspace( 0, 360, 7 ) ]  )
    
    plt.title( 'Converged' )
    
    
    
    
    
    fig = plt.figure(  )
    
    plt.scatter( decoded[0],  decoded[1], c=cmap(degrees/360), s=5 )
    
    cbar = plt.colorbar( ticks  =  np.linspace( 0, 1, 7 )  )
    cbar.set_ticklabels(  [ int(x) for x in np.linspace( 0, 360, 7 ) ]  )
    
    plt.title( 'Decoded' )
    
    
    
    return 






# def PCA( A, k=3 ):
#     ''' 
#         ** 
#         Assumes that if A is an m-by-n matrix, there are m samples of n variables each. 
#         That is, the rows of A are the observations and the columns are the variables. 
#         **    
            
#     '''
    
#     [ U, S, V ] = torch.pca_lowrank( A )         
#     projection = torch.matmul(  A,  V[:, :k]  )         # project data to first k principal components 
    
    
#     return projection





def inputPCA( modelInput, k=4 ):
    
    
    stimMat = modelInput.stimMat                        # ( d, nStims )
    
    
    [ U, S, V ] = torch.pca_lowrank( stimMat )          # 
    
    
    projection = torch.matmul(  stimMat,  V[:, :k]  )         # project data to first k principal components 
    
    
    
    return projection 
    
    
    
    
    
    
def plotPCA( A, k=2, cmap=mpl.colormaps[ 'viridis' ], stimMat=None, fig=None ):
    ''' 
        Assumes that if A is an m-by-n matrix, there are m samples of n variables each. That 
    is, the rows of A are the observations and the columns are the variables. 
    
    V represents the principal directions 
        
    '''
    
    
    #---------------------------------------------------------
    [ U, S, V ] = torch.pca_lowrank( A )                ## If A is (m,n)  -->  (m,q), (q), (n,q)
    projection = torch.matmul(  A,  V[:, :k]  )         ## project the data to first k principal components 
    
    
    if type( projection ) is torch.Tensor:
        projection = projection.detach().numpy()        ## ( m, k ) 
    #---------------------------------------------------------
    
    
    #---------------------------------------------------------
    nPts = projection.shape[0]
    colors = cmap(   [ x/nPts for x in range(nPts) ]   )
    #---------------------------------------------------------
    
    
    
    #---------------------------------------------------------
    if fig is None:
        fig = plt.figure( figsize=(6, 6) )
        
        # fig.set_cmap = cmap
        # fig = plt.figure( )
    
    if k >= 3:
        ax = fig.add_subplot( projection='3d' )
        scatterFig = ax.scatter( projection[:,0], projection[:,1], projection[:,2], s=5, c=colors )
        # scatterFig = ax.scatter( projection[:,0], projection[:,1], projection[:,2], s=5, c=colors, cmap=cmap )
        
        ax.set_xlabel( 'Component 1' )
        ax.set_ylabel( 'Component 2' )
        ax.set_zlabel( 'Component 3' )

    else: 
        scatterFig = plt.scatter( projection[:,0], projection[:,1], s=5, c=colors )
        # scatterFig = plt.scatter( projection[:,0], projection[:,1], s=5, c=colors, cmap=cmap )
        plt.xlabel( 'Component 1' )
        plt.ylabel( 'Component 2' )
        
        
        
    # sm = plt.cm.ScalarMappable( cmap=cmap )
    # sm.set_clim( vmin=0, vmax=100 )
    # plt.colorbar( sm, label='t', pad=0.15, ax=scatterFig )

        
        
    # scatterFig.set_cmap = cmap
    # plt.colorbar( scatterFig, label='t', pad=0.15, cmap=cmap )
        
    # plt.colorbar( scatterFig, label='t', pad=0.15, fraction=0.03, aspect=10 )
    plt.colorbar( scatterFig, label='t', pad=0.15 )
    plt.title( 'PCA' )
    #---------------------------------------------------------
    
    
    
    
    #---------------------------------------------------------
    if stimMat is not None:
        
        stimCmap = mpl.colormaps[ 'Reds_r' ]

        nBetas = stimMat.shape[1]
        colors = stimCmap(   [ x/nBetas for x in range(nBetas) ]   )

        
        #---------------------
        if k == 3:
            stimScatter = ax.scatter( stimMat[0,:], stimMat[1,:], c=colors, marker='*', s=10 )
        else: 
            stimScatter = plt.scatter( stimMat[0,:], stimMat[1,:], c=colors, marker='*', s=10 )
        #---------------------
        
        
        stimScatter.set_cmap( stimCmap )
        # plt.colorbar( stimScatter, label='t', location='left', fraction=0.03, aspect=10 )
        plt.colorbar( stimScatter, label='t', location='left' )
        
        # stimCbar = plt.colorbar( stimScatter, label='t', location='left' ) 
        
    #---------------------------------------------------------
    
    
    
    return projection, fig 
    
    
    
    
    
    
    
def plotStimLifecyclePCA( A, k=2, cmap1=mpl.colormaps[ 'viridis' ], cmap2=mpl.colormaps[ 'viridis' ], 
                                             stimMat=None, fig=None ):
    ''' 
        Assumes that if A is an m-by-n matrix, there are m samples of n variables each. That 
    is, the rows of A are the observations and the columns are the variables. 
    
    V represents the principal directions 
        
    '''
    
    
    #---------------------------------------------------------
    [ U, S, V ] = torch.pca_lowrank( A )                ## If A is (m,n)  -->  (m,q), (q), (n,q)
    projection = torch.matmul(  A,  V[:, :k]  )         ## project the data to first k principal components 
    
    
    if type( projection ) is torch.Tensor:
        projection = projection.detach().numpy()        ## ( m, k ) 
    #---------------------------------------------------------
    
    
    #---------------------------------------------------------
    nPts = projection.shape[0]
    colors = cmap(   [ x/nPts for x in range(nPts) ]   )
    #---------------------------------------------------------
    
    
    
    #---------------------------------------------------------
    if fig is None:
        fig = plt.figure( figsize=(6, 6) )
        # fig = plt.figure( )
    
    if k >= 3:
        ax = fig.add_subplot( projection='3d' )
        scatterFig = ax.scatter( projection[:,0], projection[:,1], projection[:,2], s=5, c=colors )
        ax.set_xlabel( 'Component 1' )
        ax.set_ylabel( 'Component 2' )
        ax.set_zlabel( 'Component 3' )

    else: 
        scatterFig = plt.scatter( projection[:,0], projection[:,1], s=5, c=colors )
        plt.xlabel( 'Component 1' )
        plt.ylabel( 'Component 2' )
        
        
    # plt.colorbar( scatterFig, label='t', pad=0.15, fraction=0.03, aspect=10 )
    plt.colorbar( scatterFig, label='t', pad=0.15 )
    plt.title( 'PCA' )
    #---------------------------------------------------------
    
    
    
    
    #---------------------------------------------------------
    if stimMat is not None:
        
        stimCmap = mpl.colormaps[ 'Reds_r' ]

        nBetas = stimMat.shape[1]
        colors = stimCmap(   [ x/nBetas for x in range(nBetas) ]   )

        
        #---------------------
        if k == 3:
            stimScatter = ax.scatter( stimMat[0,:], stimMat[1,:], c=colors, marker='*', s=10 )
        else: 
            stimScatter = plt.scatter( stimMat[0,:], stimMat[1,:], c=colors, marker='*', s=10 )
        #---------------------
        
        
        stimScatter.set_cmap( stimCmap )
        # plt.colorbar( stimScatter, label='t', location='left', fraction=0.03, aspect=10 )
        plt.colorbar( stimScatter, label='t', location='left' )
        
        # stimCbar = plt.colorbar( stimScatter, label='t', location='left' ) 
        
    #---------------------------------------------------------
    
    
    
    return projection, fig 
    
    
    


def plotPCAOfStimLifecycle( model, modelData, modelInput=None, k=3, nBetasToPlot=3, 
                                           plotPhasesSeparately=True,
                                           cmap1=None, cmap2=None, 
                                           converged=False
                                           ):
    
    #-------------------------------------------------
    ''' Get the model state activity  '''
    #-------------------------------------------------
    epochNum = model.epochNum 
    if epochNum is None:
        raise Exception( '[plotPCAOfStimLifecycle] Cannot analyze training model'  )
    
    state = modelData[ epochNum ][ 'state' ]             
    
    # titleStr = 'Stim lifecycle (epochNum =' + str(epochNum) + ')'
    
    
    
    #-------------------------------------------------
    ''' Time indices for encoding/delay phases  '''
    #-------------------------------------------------
    nEncodSteps = modelData[ 'nEncodSteps' ]
    nRetenSteps = modelData[ 'nRetenSteps' ]
    nTotalEvolSteps = nEncodSteps + nRetenSteps
    
    
    
    #-------------------------------------------------
    ''' Colormaps '''
    #-------------------------------------------------
    if cmap1 is None:
        cmap1 = mpl.colormaps[ 'summer' ]
    if cmap2 is None:
        cmap2 = mpl.colormaps[ 'cool_r' ]
    
    colorsEncoding = cmap1(   [ x/nEncodSteps for x in range(nEncodSteps) ]   )         ## (  )
    colorsRetention = cmap2(   [ x/nRetenSteps for x in range(nRetenSteps) ]   )        ## (  )
    
    colors = list(colorsEncoding) + list(colorsRetention)
    
    

    #=====================================================================================
    figList = [ ]
    
    
    for betaInd in range( nBetasToPlot ):
        
        #-------------------------------------------
        if plotPhasesSeparately:
            [ fig, axs ] = plt.subplots(  1, 3,  subplot_kw={"projection": "3d"},  figsize=(10, 5)  ) 
        else: 
            fig = plt.figure( ) 
        #-------------------------------------------
    
    
        #-------------------------------------------
        ''' Current stim lifecycle '''
        #-------------------------------------------
        startTimeInd = (nTotalEvolSteps * betaInd) + 1            # 1 to account for initial state 
        endTimeInd = (nTotalEvolSteps * (betaInd+1)) + 1
        
        lifecycleTimeInds = list(  range(startTimeInd, endTimeInd)  )    
        
        lifecycleState = state[ :, lifecycleTimeInds ]           ## ( N, nTotalEvolSteps )
        encodingState = lifecycleState[  :,  0:nEncodSteps  ]
        retentionState = lifecycleState[  :,  nEncodSteps::  ]
        
        
        
        
        #-------------------------------------------
        ''' PCA '''
        #-------------------------------------------
        lifecycleProj = PCA( lifecycleState.T ).detach().numpy()               ## ( k, nTotalEvolSteps )
        # encodingProj = projection[  0:nEncodSteps,  :  ]
        # retentionProj = projection[  nEncodSteps::,  :  ]
        
        
        encodingProj = PCA( encodingState.T ).detach().numpy()              ## ( k, nEncodSteps )
        retentionProj = PCA( retentionState.T ).detach().numpy()            ## ( k, nRetenSteps )
        
        # projection = np.concatenate(  [ encodingProj, retentionProj ]  )    ## ( k, nTotalEvolSteps )
        

        if converged:
            convergedEncodProj = encodingProj[ :, -1 ]
            convergedRetenProj = retentionProj[ :, -1 ]
            # encodingProj = encodingProj[ :, -1 ]
            # retentionProj = retentionProj[ :, -1 ]



        #---------------------------------------------------------------------------------
        ''' PLOT the PCA '''
        #---------------------------------------------------------------------------------

        if k >= 3: 
            
            
            if plotPhasesSeparately:
                axs[0].set_title( 'Full life cycle' )
                scatterFig = axs[0].scatter( lifecycleProj[:,0], lifecycleProj[:,1], lifecycleProj[:,2], s=5, c=colors )
                
                axs[1].set_title( 'Encoding' )
                axs[2].set_title( 'Retention' )
                
                if converged:                 
                    encScatter = axs[1].scatter( encodingProj[-1,0], encodingProj[-1,1], encodingProj[-1,2], s=5, c=colorsEncoding[-1] )
                    retScatter = axs[2].scatter( retentionProj[-1,0], retentionProj[-1,1], retentionProj[-1,2], s=5, c=colorsRetention[-1] )

                else:            
                    encScatter = axs[1].scatter( encodingProj[:,0], encodingProj[:,1], encodingProj[:,2], s=5, c=colorsEncoding )
                    retScatter = axs[2].scatter( retentionProj[:,0], retentionProj[:,1], retentionProj[:,2], s=5, c=colorsRetention )
                        
                
                
            
                for ax in axs:
                    ax.set_xlabel( 'Component 1' )
                    ax.set_ylabel( 'Component 2' )
                axs[-1].set_zlabel( 'Component 3' )
                
                
            
            else: 
                scatterFig = plt.scatter( lifecycleProj[:,0], lifecycleProj[:,1], lifecycleProj[:,2], s=5, c=colors )

                plt.xlabel( 'Component 1' )
                plt.ylabel( 'Component 2' )
                plt.zlabel( 'Component 3' )

            
            
            if k > 3:
                print( 'PLOTTING ONLY THE FIRST 3 PRINCIPAL COMPONENTS' )
                
                
                
                
                
                
        elif (k == 2) or (k == 1):
            
            
            if plotPhasesSeparately:
                axs[0].set_title( 'Full life cycle' )
                scatterFig = axs[0].scatter( lifecycleProj[:,0], lifecycleProj[:,1], s=5, c=colors )
                
                axs[1].set_title( 'Encoding' )
                encScatter = axs[1].scatter( encodingProj[:,0], encodingProj[:,1], s=5, c=colorsEncoding )
                
                axs[2].set_title( 'Retention' )
                retScatter = axs[2].scatter( retentionProj[:,0], retentionProj[:,1], s=5, c=colorsRetention )
            
                for ax in axs:
                    ax.set_xlabel( 'Component 1' )
                axs[-1].set_ylabel( 'Component 2' )


            else:
                encScatter = plt.scatter( encodingProj[:,0], encodingProj[:,1], s=5, c=colorsEncoding )
                retScatter = plt.scatter( retentionProj[:,0], retentionProj[:,1], s=5, c=colorsRetention )

                plt.xlabel( 'Component 1' )
                plt.ylabel( 'Component 2' )            



            
        else:
            raise Exception( '[plotPCAOfStimLifecycle] Cannot plot for k less than 1.' )



        
        
        
        
        
        #---------------------------------------------------------------------------------
        ''' Colorbar '''
        #---------------------------------------------------------------------------------
        spiltRatio = nEncodSteps / nTotalEvolSteps
        fig = addDoubleColormap( fig, [cmap1, cmap2], spiltRatio, nTotalEvolSteps, label='t' )
    

        titleStr = 'Stim lifecycle: epoch {} (beta {})'.format( epochNum, betaInd )
        fig.suptitle( titleStr ) 
        fig.subplots_adjust(  top=0.9  )

    

        #---------------------------------------------------------------------------------
        ''' PLOT the goal stimulus '''
        #---------------------------------------------------------------------------------
        if modelInput is not None:
        
            x = modelInput.stimMat[ 0, betaInd ].detach().numpy()
            y = modelInput.stimMat[ 1, betaInd ].detach().numpy()
            
            for ax in axs[0:-1]:
                ax.scatter(  x,  y,  s=8,  c='r',  marker='*'  )
            # plt.scatter(  x,  y,  s=8,  c='r',  marker='*'  )
    
    
    
    
            
        #---------------------------------------------------------------------------------
        ''' Finalize Fig '''
        #---------------------------------------------------------------------------------
        # fig.tight_layout()
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        figList.append( fig )
        # fig.get_axes()[0].view_init(elev=45, azim=90, roll=270)

    


    return figList 
        
    
    
    
    
    


def plotPCAOfStimLifecycle_full( model, modelData, modelInput=None, k=3, nBetasToPlot=3, 
                                           plotPhasesSeparately=True,
                                           cmap1=None, cmap2=None, 
                                           converged=False
                                           ):
    
    #-------------------------------------------------
    ''' Get the model state activity  '''
    #-------------------------------------------------
    epochNum = model.epochNum 
    if epochNum is None:
        raise Exception( '[plotPCAOfStimLifecycle] Cannot analyze training model'  )
    
    state = modelData[ epochNum ][ 'state' ]             
    
    # titleStr = 'Stim lifecycle (epochNum =' + str(epochNum) + ')'
    
    
    
    #-------------------------------------------------
    ''' Time indices for encoding/delay phases  '''
    #-------------------------------------------------
    nEncodSteps = modelData[ 'nEncodSteps' ]
    nRetenSteps = modelData[ 'nRetenSteps' ]
    nTotalEvolSteps = nEncodSteps + nRetenSteps
    
    
    
    #-------------------------------------------------
    ''' Colormaps '''
    #-------------------------------------------------
    if cmap1 is None:
        cmap1 = mpl.colormaps[ 'summer' ]
    if cmap2 is None:
        cmap2 = mpl.colormaps[ 'cool_r' ]
    
    colorsEncoding = cmap1(   [ x/nEncodSteps for x in range(nEncodSteps) ]   )         ## (  )
    colorsRetention = cmap2(   [ x/nRetenSteps for x in range(nRetenSteps) ]   )        ## (  )
    
    colors = list(colorsEncoding) + list(colorsRetention)
    
    
    
    
    
    #=====================================================================================
    ''' PCA the full evolution '''
    #=====================================================================================
    stateTrimmed = state[ :, 1:: ]                                  ## to account for initial state (IC) 
    stateProj = PCA( stateTrimmed.T ).detach().numpy()              ## ( k, nTimes-1 )
    
    
    
    
    
    
    
    

    #=====================================================================================
    figList = [ ]
    
    
    
    #-------------------------------------------
    if converged:
        if plotPhasesSeparately:
            [ fig, axs ] = plt.subplots(  1, 3,  subplot_kw={"projection": "3d"},  figsize=(10, 5)  ) 
        else: 
            fig = plt.figure( ) 
    #-------------------------------------------
    
    
    
    
    for betaInd in range( nBetasToPlot ):
        
        #-------------------------------------------
        if not converged:
            if plotPhasesSeparately:
                [ fig, axs ] = plt.subplots(  1, 3,  subplot_kw={"projection": "3d"},  figsize=(10, 5)  ) 
            else: 
                fig = plt.figure( ) 
        #-------------------------------------------
    
    
        #-------------------------------------------
        ''' Current lifecycle PCA '''
        #-------------------------------------------
        startTimeInd = (nTotalEvolSteps * betaInd)            # 1 to account for initial state 
        endTimeInd = (nTotalEvolSteps * (betaInd+1))
        
        lifecycleTimeInds = list(  range(startTimeInd, endTimeInd)  )    
        
        lifecycleProj = stateProj[ lifecycleTimeInds, : ]                       ## ( N, nTimes )
        encodingProj = lifecycleProj[  0:nEncodSteps,  :  ]                     ## ( nEncodSteps, k )
        retentionProj = lifecycleProj[  nEncodSteps::,  :  ]                    ## ( nEncodSteps, k )
        
        

        if converged:
            convergedEncodProj = encodingProj[ -1, : ] 
            convergedRetenProj = retentionProj[ -1, : ] 
            # encodingProj = encodingProj[ :, -1 ]
            # retentionProj = retentionProj[ :, -1 ]



        #---------------------------------------------------------------------------------
        ''' PLOT the PCA '''
        #---------------------------------------------------------------------------------

        if k >= 3: 
            
            
            if plotPhasesSeparately:
                axs[0].set_title( 'Full life cycle' )
                scatterFig = axs[0].scatter( lifecycleProj[:,0], lifecycleProj[:,1], lifecycleProj[:,2], s=5, c=colors )
                
                axs[1].set_title( 'Encoding' )
                axs[2].set_title( 'Retention' )
                
                if converged:                 
                    encScatter = axs[1].scatter( encodingProj[-1,0], encodingProj[-1,1], encodingProj[-1,2], s=5, c=colorsEncoding[-1] )
                    retScatter = axs[2].scatter( retentionProj[-1,0], retentionProj[-1,1], retentionProj[-1,2], s=5, c=colorsRetention[-1] )

                else:            
                    encScatter = axs[1].scatter( encodingProj[:,0], encodingProj[:,1], encodingProj[:,2], s=5, c=colorsEncoding )
                    retScatter = axs[2].scatter( retentionProj[:,0], retentionProj[:,1], retentionProj[:,2], s=5, c=colorsRetention )
                        
                
                
            
                for ax in axs:
                    ax.set_xlabel( 'Component 1' )
                    ax.set_ylabel( 'Component 2' )
                axs[-1].set_zlabel( 'Component 3' )
                
                
            
            else: 
                scatterFig = plt.scatter( lifecycleProj[:,0], lifecycleProj[:,1], lifecycleProj[:,2], s=5, c=colors )

                plt.xlabel( 'Component 1' )
                plt.ylabel( 'Component 2' )
                plt.zlabel( 'Component 3' )

            
            
            if k > 3:
                print( 'PLOTTING ONLY THE FIRST 3 PRINCIPAL COMPONENTS' )
                
                
                
                
                
                
        elif (k == 2) or (k == 1):
            
            
            if plotPhasesSeparately:
                axs[0].set_title( 'Full life cycle' )
                scatterFig = axs[0].scatter( lifecycleProj[:,0], lifecycleProj[:,1], s=5, c=colors )
                
                axs[1].set_title( 'Encoding' )
                encScatter = axs[1].scatter( encodingProj[:,0], encodingProj[:,1], s=5, c=colorsEncoding )
                
                axs[2].set_title( 'Retention' )
                retScatter = axs[2].scatter( retentionProj[:,0], retentionProj[:,1], s=5, c=colorsRetention )
            
                for ax in axs:
                    ax.set_xlabel( 'Component 1' )
                axs[-1].set_ylabel( 'Component 2' )


            else:
                encScatter = plt.scatter( encodingProj[:,0], encodingProj[:,1], s=5, c=colorsEncoding )
                retScatter = plt.scatter( retentionProj[:,0], retentionProj[:,1], s=5, c=colorsRetention )

                plt.xlabel( 'Component 1' )
                plt.ylabel( 'Component 2' )            



            
        else:
            raise Exception( '[plotPCAOfStimLifecycle] Cannot plot for k less than 1.' )



        
        
        
        
        
        #---------------------------------------------------------------------------------
        ''' Colorbar '''
        #---------------------------------------------------------------------------------
        spiltRatio = nEncodSteps / nTotalEvolSteps
        fig = addDoubleColormap( fig, [cmap1, cmap2], spiltRatio, nTotalEvolSteps, label='t' )
    

        titleStr = 'Stim lifecycle: epoch {} (beta {})'.format( epochNum, betaInd )
        fig.suptitle( titleStr ) 
        fig.subplots_adjust(  top=0.9  )

    

        #---------------------------------------------------------------------------------
        ''' PLOT the goal stimulus '''
        #---------------------------------------------------------------------------------
        if modelInput is not None:
        
            x = modelInput.stimMat[ 0, betaInd ].detach().numpy()
            y = modelInput.stimMat[ 1, betaInd ].detach().numpy()
            
            for ax in axs[0:-1]:
                ax.scatter(  x,  y,  s=8,  c='r',  marker='*'  )
            # plt.scatter(  x,  y,  s=8,  c='r',  marker='*'  )
    
    
    
    
            
        #---------------------------------------------------------------------------------
        ''' Finalize Fig '''
        #---------------------------------------------------------------------------------
        # fig.tight_layout()
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        figList.append( fig )
        # fig.get_axes()[0].view_init(elev=45, azim=90, roll=270)

    


    return figList 
        
    
    
    
    
    
    
    
    
    
    
def addDoubleColormap( fig, cmapList, spiltRatio, totalPts, label='', tickLabels=[0,1] ):
    ## Much help from:  https://stackoverflow.com/questions/30082174/join-two-colormaps-in-imshow
    
    
    fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
    lastAxPos = fig.get_axes()[-1].get_position()
    cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, lastAxPos.height]  )
    # cbarAx = plt.axes(  [0.96, lastAxPos.y0, 0.02, lastAxPos.height]  )

    
    
    nPts1 = int( spiltRatio * totalPts )
    # img1 = np.linspace( 0, 1, nPts1 ).reshape( [nPts1, 1] )
    img1 = np.linspace( 0, 1, nPts1 ).reshape( [nPts1, 1] )[ : : -1 ] 
    
    nPts2 = int( (1-spiltRatio) * totalPts )
    # img2 = np.linspace( 0, 1, nPts2 ).reshape( [nPts2, 1] )
    img2 = np.linspace( 0, 1, nPts2 ).reshape( [nPts2, 1] )[ : : -1 ]
    
    
    spiltPt = (-1) + (2 * spiltRatio)
    
    
    cbarAx.imshow( img1, cmap=cmapList[0], extent=[0, 1, -1, spiltPt] ) 
    cbarAx.imshow( img2, cmap=cmapList[1], extent=[0, 1, spiltPt, 1] )  
    
    
    cbarAx.set_ylim( -1, 1 )
    cbarAx.set_aspect( 10 )
    cbarAx.set_ylabel( label )
    cbarAx.yaxis.set_label_position( "right" )
    cbarAx.set_xticks( [ ] )
    
    nTicks = len( tickLabels )
    cbarAx.set_yticks( np.linspace(-1,1,nTicks) )
    cbarAx.set_yticklabels( tickLabels ) 
    cbarAx.yaxis.tick_right( )



    return fig 








def binBetasByTheta( modelInput, nBins=None, binSize=None ):
    
    
    if (nBins is not None) and (binSize is not None):
        raise Exception( 'Must give EITHER nBins or binSize' )
    if (nBins is None) and (binSize is None):
        nBins = 6
    
    
    # [ counts, bin_edges ] = np.histogram( degrees, bins=nBins, range=(0,360) )
    
    
    
    if binSize is not None:
        nBins = int(  np.ceil( 360/binSize )  )
        intervalLen = int( binSize )

    if nBins is not None:
        intervalLen = int( 360/nBins )
        
    
    thetas = modelInput.thetas[0]
    degrees = thetas * ( 180 / math.pi )  
    
    
    
    indsBinned = { }
    
    
    
    for i in range( nBins ):
        start = int(  i * intervalLen  )
        end = int(  start + intervalLen  )
        
        key = str( start ) + '-' + str( end )
    
    
        indsForBin = np.where(   np.logical_and( degrees>=start, degrees<end )   )
        # indsBinned[ key ] = degrees[ indsForBin ]
        indsBinned[ key ] = indsForBin
    
    
    
    
    return indsBinned
    
    
    
    
    
# def plotPCA_2Colorbars(  ):
    
    
    
# encoding = False
# if nRetenSteps == 0:
#     encoding = True

        
# nBetasToPlot = 3
# # nBetasToPlot = nTestBetas



        

# if not plotOnlyRP:
#     # 

#     # [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds[0:nEncodingsToPlot] )

#     #--------------------------------------------------------
        
    
#     for epochNum in [ 0, nEpochs ]:
#         print( )
#         print( epochNum )
        
#         #--------------------------------------------------------
#         '''  '''
#         #--------------------------------------------------------
#         for k in range( nBetasToPlot ):
#             print( k )
            
#             fig = plt.figure( ) 
        
#             [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, [stimTimeInds[k]] )
            
            
#             if encoding:
#                 state = testingData[ epochNum ][ 'state' ][ :, encodingTimeInds ]
#                 titleStr = 'Encoding periods (epochNum =' + str(epochNum) + ')'
#             else: 
#                 state = testingData[ epochNum ][ 'state' ][ :, retentionTimeInds ]
#                 titleStr = 'Retention periods (epochNum =' + str(epochNum) + ')'

                
            
            
#             [ proj, fig ] = plotPCA( state.T, k=3, fig=fig )
#             fig.suptitle( titleStr )
#             # fig.get_axes()[0].view_init(elev=45, azim=90, roll=270)

        

    
    
    
    
    
    
    
#%%


def getDiffPeriodTimeInds( nEncodSteps, nRetenSteps, stimTimeInds, simOptions=None ):
    
    
    if simOptions is not None:
        nEncodSteps = simOptions['nEncodSteps_testing']
        nRetenSteps = simOptions['nRetenSteps']
        
        stimTimeInds = simOptions['stimTimeInds']
    
    
    
    encodingTimeInds = [  ]
    retentionTimeInds = [  ]
    
    if type(stimTimeInds) is not list:
        stimTimeInds = [stimTimeInds]
    
    
    for t in stimTimeInds:
        
        encodingTimes = [  t+i for i in range(nEncodSteps)  ]
        retentionTimes = [  (t+nEncodSteps)+i for i in range(nRetenSteps)  ]
        
        encodingTimeInds = encodingTimeInds + encodingTimes 
        retentionTimeInds = retentionTimeInds + retentionTimes
        
        
    return encodingTimeInds, retentionTimeInds





















#%%


# epochNum = nEpochs


# for statesToPCA in [  'state',  'convergedStates_encoding',  'convergedStates_retention'  ]:
# # statesToPCA = 'state'
# # statesToPCA = 'convergedStates_encoding'
# # statesToPCA = 'convergedStates_retention'



#     ## THE DATA 
#     A = testingData[ epochNum ][ statesToPCA ]
    
    
#     ## CENTER THE DATA (subtract the mean)
    
    
#     ## DO THE PCA AND THEN PLOT THE FIRST K COMPONENTS 
#     proj, fig = plotPCA( A )
#     fig.suptitle( statesToPCA )



# #%%

# fig = plt.figure() 


# startBetaInd = 200


# # for k in range( nTestBetas ):
# for k in range( 8 ):

    
#     startEncodingTimeInd = testingData[ 'stimTimeInds' ][ k ]
#     endEncodingTimeInd = testingData[ 'stimTimeInds' ][ k+1 ] - 1
    
#     totalLength = endEncodingTimeInd - startEncodingTimeInd
    
#     # encodingTimeInds = np.arange( startEncodingTimeInd, endEncodingTimeInd )  
    
    
#     A = testingData[ epochNum ][ 'state' ][ :, startEncodingTimeInd+(startBetaInd*totalLength):endEncodingTimeInd+(startBetaInd*totalLength) ]


#     [ U, S, V ] = torch.pca_lowrank( A )
    
#     plt.scatter(  )
    
    
#     proj = torch.matmul(  A,  V[:, :2]  ).detach().numpy()
#     plt.scatter(  proj[:,0],  proj[:,1],  c=colors[k]  )    


#     # proj, fig = plotPCA( A )
#     # fig.suptitle( k )
    
#     currStim = testingInput.stimMat[ :, k+startBetaInd ]
#     plt.scatter( currStim[0], currStim[1], c=colors[k], s=8 )
    









#%%
    




def plotCircularThetasHist( thetas, nBins=90 ):
## https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python


    bottom = 8
    max_height = 4
    
        
    nBetas = len( thetas )
    
    radii = max_height * np.random.rand( nBetas )
    width = (2*np.pi) / nBetas
    
    
    
    fig = plt.subplot( 111, polar=True )
    
    
    plt.hist( thetas, bins=nBins ) 
    
    
    [ counts, bins ] = np.histogram( thetas, bins=nBins )
    maxFreq = np.max( counts )
    # ticks = [  0,  int(np.round(maxFreq,-1)/2),  int(np.round(maxFreq,-1))  ]
    freqTicks = [  int(x) for x in np.linspace(0, np.round(maxFreq,-1), 5)  ]
    fig.set_rticks(  freqTicks  ) 
    
    
    fig.set_ymargin( 0.25 )



    return fig 



























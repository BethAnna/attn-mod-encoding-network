#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:14:54 2024

@author: bethannajones
"""





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:14:06 2024

@author: bethannajones
"""




'''

v17 - Training of the PyTorch model is all set. Working on testing and visualizing now. 

v19 - Final copy of v18 fixed an issue where as nEpochs was increasing, the RP of earlier 
        epochs did not agree with the RP when nEpochs was smaller. That is, for the same 
        inputs, d, N, nProxSteps, initial connections, etc, when we set nEpochs=600 and 
        nEpochs=30,000 in separate consoles, the line for epochNums={0,600} was different 
        in each console. I fixed this by changing how state was stored (created a dummy 
        variable to update for the current epochNum then assign that dummy variable to the 
        dictionary testingData at the end of the loop). 
            Now I am working on improving general performance through changing nEpcohs, 
        nProxSteps, and the learning rate of the optimizer, etc.  
       
        
        
v20 - The code is now able to train on a single stimulus (a single epoch) multiple times, 
        specified by maxIter, before moving on to a new stimulus/epoch.
            -Worked on Getting "good" RP for q=3 or more
        
        
v21- Switching to training D,H directly (instead of through W,Ms,Md)


v22 - Working out kinks in implementation of trainMatsDirectly 


v23 - It appears that the training of H and D directly is working. Now working on tuning 
        the hyperparameters, etc. Also made the edits so that:
            - the nProxSteps for training and testing can be independently specified (in 
            order to test cutting off convergence period during testing) 
            - there can be a period where the stimulus and/or reference state is/are no 
            longer available during the encoding period (proximal evolution) 


v24 - Change to model to include attentional parameter alpha so now 
        J_tot = alpha * J_enc  +  (1-alpha) * J_reten  +  || r(t) ||_1
    where alpha is in [0,1] and 
        J_enc = errW * J_err  +  hW * J_h 
            J_err = ||  x(t) - D*r(t)  ||_2^2
            J_h = ||  r(t-1) - H*r(t)  ||_2^2
        J_reten = rW * ||  r(t) - r(t-1)  ||_2^2.
    Note that when alpha=1, the model is equivalent to previous formulation. 


v25 - When alpha=0 and initalize with non-zero IC, the state is affected by the arrival of 
    a new stim, which shouldn't be the case. This is because we use refState, which is the 
    converged state and is updated at the arrival of every stim. Which is a little weird 
    for the retention mode. 
        Switching so that we remove the hsitory term and r(t-1) refers to the immediately 
    previous state instead of the previously converged state. This cahnges focuses on the 
    effect of the attentional parameter and retention of immediate info (as opposed to the
    recall of previous information). 



*  M. Kafashan and S. Ching, ‘Recurrent networks with soft-thresholding nonlinearities for 
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





from retenAnalysisFuncs import makeAlphaRefStr, getLongFilename











#=========================================================================================
#=========================================================================================
#% % chg CWD
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
#%% Functions
#=========================================================================================
#=========================================================================================





def extractRetenAndEncodFromAlphStr( alphaRefStr ):
    
    alphs = alphaRefStr.split( sep='_' )
    
    eAlp = alphs[0][  alphs[0].find('A')+1 : :  ]
    rAlp = alphs[1][  alphs[1].find('A')+1 : :  ]
    
    
    return eAlp, rAlp








# def makeAlphaRefStr( encodingAlpha, rententionAlpha ):
    
    
#     # alphaRefStr = 'rA' + str(rAlpha) + '_' + 'eA' + str(eAlpha)
#     alphaRefStr = 'eA' + str(encodingAlpha) + '_' + 'rA' + str(rententionAlpha)
    
#     return alphaRefStr 




# def makeAlphaRefStrs( rententionAlphaVals=None, encodingAlphaVals=None, 
#                                                              simOptions=None ):
def makeAlphaRefStrs( encodingAlphaVals=None, rententionAlphaVals=None, 
                                                             simOptions=None ):

    
    
    if (rententionAlphaVals is None) or (encodingAlphaVals is None):

        

        if (rententionAlphaVals is None) and (encodingAlphaVals is None):
            
            if simOptions is not None:
                rententionAlphaVals = [  simOptions['retentionAlpha']  ]
                encodingAlphaVals = [  simOptions['encodingAlpha']  ]
            else: 
                raise Exception( '[makeAlphaRefStrs] Need either i) lists of alpha values; or ii) simOptions' )
        
        
        elif (rententionAlphaVals is None):
            if simOptions is not None:
                rententionAlphaVals = [  simOptions['retentionAlpha']  ]
            else: 
                raise Exception( '[makeAlphaRefStrs] If rententionAlphaVals is not given, must provide simOptions ' )


        elif (encodingAlphaVals is None):
            if simOptions is not None:
                encodingAlphaVals = [  simOptions['encodingAlpha']  ]
            else: 
                raise Exception( '[makeAlphaRefStrs] If encodingAlphaVals is not given, must provide simOptions ' )

                
                


    alphaRefStrs = [ ]
    
    for rAlpha in rententionAlphaVals: 
        
        for eAlpha in encodingAlphaVals: 
            
            # alphaRefStr = 'rAlp' + str(rAlpha) + '_' + 'eAlp' + str(eAlpha)
            # alphaRefStr = 'rA' + str(rAlpha) + '_' + 'eA' + str(eAlpha)
            alphaRefStr = makeAlphaRefStr( eAlpha, rAlpha )
            alphaRefStrs.append( alphaRefStr )
    
    return alphaRefStrs
    




def loadInAlphaData( rententionAlphaVals, encodingAlphaVals, simOptions ):

    

    alphaTestingModelsDict = { }
    alphaTestingDataDict = { }
    alphaTestingInputDict = { }
    
    
    # alphaRefStrs = makeAlphaRefStrs( rententionAlphaVals, encodingAlphaVals, simOptions )
    alphaRefStrs = makeAlphaRefStrs( encodingAlphaVals, rententionAlphaVals, simOptions )
    
    
    for alphaRefStr in alphaRefStrs:
            
            
        #---------------------------------------------
        ''' Update the settings '''
        #---------------------------------------------
        currSimOptions = simOptions.copy() 
        
        # [ rAlp, eAlp ] = extractRetenAndEncodFromAlphStr( alphaRefStr )
        [ eAlp, rAlp ] = extractRetenAndEncodFromAlphStr( alphaRefStr )
    
        
        currSimOptions[ 'retentionAlpha' ] = rAlp
        currSimOptions[ 'encodingAlpha' ] = eAlp
    
        
        #---------------------------------------------
        ''' Find saved data & load it in '''
        #---------------------------------------------
        # saveDir = nameSaveDir( currSimOptions, weightCombos, nEpochs )
        # saveDir = nameSaveDir( currSimOptions, weightCombos )
        saveDir = nameSaveDir( currSimOptions )
    
        modelInfoDict = getModelInfo( saveDir )
    
        
        #---------------------------------------------
        ''' Copy to new dictionary '''
        #---------------------------------------------
        alphaTestingModelsDict[ alphaRefStr ] = modelInfoDict[ 'epochModels' ]
        
        alphaTestingDataDict[ alphaRefStr ] = modelInfoDict[ 'testingData' ]
        
        alphaTestingInputDict[ alphaRefStr ] = modelInfoDict[ 'testingInput' ] 


    return alphaTestingModelsDict, alphaTestingDataDict, alphaTestingInputDict











#=========================================================================================
#=========================================================================================



def loadInDataByVarVal( varValList, simOptions, varName='nRetenSteps', saveDirs=None ): 
    '''  
        Load in saved data based on the given simOptions, but the values of varType
    are varied based the different values in varValList. Here varName refers to the key of 
    simOptions the variable/parameter is stored under. 
    
    Loads in the data and stores them in dicts: testingModels, testingData, testingInput
    
    '''
    
    testingModelsDict = { }
    testingDataDict = { }
    testingInputDict = { }
    
    
    
    for varVal in varValList:
            
            
        #---------------------------------------------
        ''' Update the settings '''
        #---------------------------------------------
        currSimOptions = simOptions.copy() 
        currSimOptions[ varName ] = varVal            


    
        
        #---------------------------------------------
        ''' Find saved data & load it in '''
        #---------------------------------------------
        saveDir = nameSaveDir( currSimOptions )
        modelInfoDict = getModelInfo( saveDir )
    
        
        #---------------------------------------------
        ''' Copy to new dictionary '''
        #---------------------------------------------
        testingModelsDict[ varVal ] = modelInfoDict[ 'epochModels' ]
        
        testingDataDict[ varVal ] = modelInfoDict[ 'testingData' ]
        
        testingInputDict[ varVal ] = modelInfoDict[ 'testingInput' ] 

    
    
    return testingModelsDict, testingDataDict, testingInputDict





#=========================================================================================
#=========================================================================================



def analyzeVarValData( testingModelsDict, testingDataDict, testingInputDict, simOptions, 
                                      # varType='nRetenSteps',
                                       epochNum=None, FH=8, 
                                       compareToRefState=False ): 
    
    
    varValList = list( testingModelsDict.keys() )
    varValAnalysisDict = { }
    
    
    if epochNum is None:
        epochNum = simOptions['parameters']['nEpochs']
    
    
    for varVal in varValList: 
        
        varValAnalysisDict[ varVal ] = { }
        
        #-------------------------------------------------------------------------------------
        ''' Grab the data '''
        #-------------------------------------------------------------------------------------
        epochModels = testingModelsDict[ varVal ]
        testingData = testingDataDict[ varVal ]
        testingInput = testingInputDict[ varVal ]
        
        
        #-------------------------------------------------------------------------------------
        ''' Analysis '''
        #-------------------------------------------------------------------------------------
        # analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH )
        analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH, compareToRefState=compareToRefState )
    
    
        #-------------------------------------------------------------------------------------
        ''' Save to new dictionary '''
        #-------------------------------------------------------------------------------------
        varValAnalysisDict[ varVal ][ 'recovPerfDict' ] = analysisDict[ 'recovPerfDict' ][ epochNum ]
        varValAnalysisDict[ varVal ][ 'recovPerfRetenDict' ] = analysisDict[ 'recovPerfRetenDict' ][ epochNum ]
        varValAnalysisDict[ varVal ][ 'reconRDict' ] = analysisDict[ 'reconRDict' ][ epochNum ]
        # alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ] = analysisDict[ 'recovPerfDict' ]
    
    
    
    return varValAnalysisDict

#=========================================================================================
#=========================================================================================










#=========================================================================================
#=========================================================================================
#%% Alpha values
#=========================================================================================
#=========================================================================================



# simOptions['parameters']['maxIter'] = 25
# simOptions['parameters']['nEpochs'] = 5000
# simOptions['nRetenSteps'] = 1000





# # simOptions['parameters']['maxIter'] = 10
# # simOptions['maxIter'] = 10
# simOptions['parameters']['nEpochs'] = 5000
# # simOptions['nRetenSteps'] = 2000                    ## saveDir indicates TRAINING nRetenSteps
# # simOptions['nRetenSteps'] = 100                    ## saveDir indicates TRAINING nRetenSteps
# # simOptions['nEncodSteps'] = 50                    ## saveDir indicates TRAINING nRetenSteps




# simOptions['maxIter'] = 25
# simOptions['parameters']['nEpochs'] = 2000
# simOptions['nRetenSteps'] = 500                    ## saveDir indicates TRAINING nRetenSteps
# simOptions['nEncodSteps'] = 25                    ## saveDir indicates TRAINING nRetenSteps
# # simOptions['parameters']['nRetenSteps'] = simOptions['nRetenSteps']




# FH = analysisDict[ 'recovPerfDict' ][ epochNum ].shape[1] - 2
FH = 12



# rententionAlphaVals = [  0.1,  0.2  ]
rententionAlphaVals = [  0.1,  0.05  ]
# rententionAlphaVals = [  0.05,  0.1,  0.15,  0.2  ]
rententionAlphaVals = [  0.05,  0.1,  0.001  ]
# rententionAlphaVals = [  0.05  ]
# rententionAlphaVals = [  0.1  ]
rententionAlphaVals = [  0.001, 0.002, 0.003, 0.004  ]
# rententionAlphaVals = [  0.002, 0.004  ]
rententionAlphaVals = [  0.001  ]

# rententionAlphaVals = [  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7  ]








# encodingAlphaVals = [  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9  ]
# encodingAlphaVals = [  1,  0.95,  0.9,  0.85,  0.8,  0.75,  0.7 ]
# encodingAlphaVals = [  0.9,  0.85,  0.8,  0.75,  0.7 ]

# encodingAlphaVals = [  1,  0.9,  0.8,  0.7,  0.6,  0.5,  0.4  ]
encodingAlphaVals = [  0.9,  0.8,  0.7,  0.6,  0.5,  0.4  ]
# encodingAlphaVals = [  1,  0.8,  0.6,  0.4  ]
# encodingAlphaVals = [  0.9,  0.7,  0.5  ]

# encodingAlphaVals = [  0.9  ]







#=========================================================================================
''' Load in data & analyze '''
#=========================================================================================

alphaAnalysisDict = {  }






epochNum = nEpochs 
# 

# alphaRefStrs = makeAlphaRefStrs( rententionAlphaVals, encodingAlphaVals, simOptions )
alphaRefStrs = makeAlphaRefStrs( encodingAlphaVals, rententionAlphaVals, simOptions )
[ alphaTestingModelsDict, alphaTestingDataDict, alphaTestingInputDict ] = loadInAlphaData( rententionAlphaVals, encodingAlphaVals, simOptions )


for alphaRefStr in alphaRefStrs: 
    
    
    alphaAnalysisDict[ alphaRefStr ] = { }
    
    
    #-------------------------------------------------------------------------------------
    ''' Grab the data '''
    #-------------------------------------------------------------------------------------
    epochModels = alphaTestingModelsDict[ alphaRefStr ]
    testingData = alphaTestingDataDict[ alphaRefStr ]
    testingInput = alphaTestingInputDict[ alphaRefStr ]
    
    
    #-------------------------------------------------------------------------------------
    ''' Analysis '''
    #-------------------------------------------------------------------------------------
    analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                 # compareToRefState=RP_compareToRefState )
                                                     ) 

    #-------------------------------------------------------------------------------------
    ''' Save to new dictionary '''
    #-------------------------------------------------------------------------------------
    alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ] = analysisDict[ 'recovPerfDict' ][ epochNum ]
    alphaAnalysisDict[ alphaRefStr ][ 'reconRDict' ] = analysisDict[ 'reconRDict' ][ epochNum ]
    # alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ] = analysisDict[ 'recovPerfDict' ]



    del analysisDict



    # #-------------------------------------------------------------------------------------
    # '''  '''
    # #-------------------------------------------------------------------------------------
    # [ rAlp, eAlp ] = extractRetenAndEncodFromAlphStr( alphaRefStr )

    
    
    # #----------------------------------
    # ''' Recovery perfs '''
    # #----------------------------------
    # rAlp_recoveryPerfsDict[ rAlp ] = alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ]
    # eAlp_recoveryPerfsDict[ eAlp ] = alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ]

    # # recoveryPerfsDict[ rAlp ][ eAlp ] = 
















#=========================================================================================
''' Plot '''
#=========================================================================================


recoveryPerfsDict = { }

for alphaRefStr in alphaRefStrs:
    
    print(  )  
    print( alphaRefStr )
    
    
    #-------------------------------------------------------------------------------------
    '''  '''
    #-------------------------------------------------------------------------------------
    # [ rAlp, eAlp ] = extractRetenAndEncodFromAlphStr( alphaRefStr )
    [ eAlp, rAlp ] = extractRetenAndEncodFromAlphStr( alphaRefStr )

    print( rAlp, eAlp )

    
    rKey = 'r' + str(rAlp)
    
    if rKey not in recoveryPerfsDict.keys():
        recoveryPerfsDict[ rKey ] = { }
    
    recoveryPerfsDict[ rKey ][ eAlp ] = alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ]
    
    
    
    eKey = 'e' + str(eAlp)
    
    if eKey not in recoveryPerfsDict.keys():
        recoveryPerfsDict[ eKey ] = { }
        
    recoveryPerfsDict[ eKey ][ rAlp ] = alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ]

    




    
wcStr = weightComboToReadable( weightCombos[0] )
nBetas = testingInput.nBetas



#-----------------------------------------------------------------------------------------
## Fixed rAlpha
#-----------------------------------------------------------------------------------------

if len( encodingAlphaVals ) > 1:
    
    rRPFigList = [] 
    
    for rAlp in rententionAlphaVals:
        
        rKey = 'r' + str(rAlp)
        currDict = recoveryPerfsDict[ rKey ]
        
        
        rRPFig = plotRP( currDict, nBetas, colorbarLabel=r'Encoding $\alpha$', forwardHorizon=FH ) 
        
        titleStr = 'rAlp' + str(rAlp) + '  -  ' + wcStr
        rRPFig.suptitle(  titleStr  )
        
        
        rRPFigList.append( rRPFig )
    
    
#-----------------------------------------------------------------------------------------    
## Fixed eAlpha
#-----------------------------------------------------------------------------------------

if len( rententionAlphaVals ) > 1:

    eRPFigList = [] 
    
    for eAlp in encodingAlphaVals:
        
        eKey = 'e' + str(eAlp)
        currDict = recoveryPerfsDict[ eKey ]
    
    
        eRPFig = plotRP( currDict, nBetas, colorbarLabel=r'Retention $\alpha$', forwardHorizon=FH ) 

        titleStr = 'eAlp' + str(eAlp) + '  -  ' + wcStr
        eRPFig.suptitle(  titleStr  )


        eRPFigList.append( eRPFig )












#     rRPFig = plotRP( rAlp_recoveryPerfsDict, nBetas, colorbarLabel=r'Retention $\alpha$', forwardHorizon=FH ) 


    
#     # FH = analysisDict[ 'recovPerfDict' ][ epochNum ].shape[1] - 2
    
    
    
    
#     rRPFig = plotRP( rAlp_recoveryPerfsDict, nBetas, colorbarLabel=r'Retention $\alpha$', forwardHorizon=FH ) 
    
    
#     eRPFig = plotRP( eAlp_recoveryPerfsDict, nBetas, colorbarLabel=r'Encoding $\alpha$', forwardHorizon=FH ) 
    
    
    
#     wcStr = weightComboToReadable( weightCombos[0] )
#     [ rAlp, eAlp ] = extractRetenAndEncodFromAlphStr( alphaRefStr )
#     # titleStr = alphaRefStr + '  -  ' + wcStr
    
#     titleStr = 'eAlp' + eAlp + '  -  ' + wcStr
#     rRPFig.suptitle(  titleStr  )
    
#     titleStr = 'rAlp' + rAlp + '  -  ' + wcStr
#     eRPFig.suptitle(  titleStr  )


# #%%









def computeHessianEigvals( model ):
    
    D = model.D 
    H = model.H 
    
    
    rAlp = model.retentionAlpha
    eAlp = model.encodingAlpha
    eW = model.errorWeight
    hW = model.historyWeight
    
    term1 = eW * ( D.T @ D )
    term2 = hW * ( H.T @ D.T @ D @ H  )
    
    Hessian = term1 + term2 
    eigvals = torch.linalg.eig( Hessian )[ 0 ]
    
    return eigvals 
    
    
    # rHess = rAlp * Hessian
    # rEigvals = torch.linalg.eig( rHess )[ 0 ]
    
    # eHess = eAlp * Hessian
    # eEigvals = torch.linalg.eig( eHess )[ 0 ]

    # return eEigvals, rEigvals













#=========================================================================================
#=========================================================================================
#%% maxIter 
#=========================================================================================
#=========================================================================================

compareToRefState = True

# simOptions[ 'retentionAlpha' ] = 0.02
# simOptions[ 'encodingAlpha' ] = 0.9

# simOptions[ 'nEncodSteps' ] = simOptions[ 'nEncodSteps_testing' ] = 50 
# simOptions[ 'nRetenSteps' ] = simOptions[ 'nRetenSteps_ext' ] = 100 

# simOptions[ 'parameters' ][ 'nEpochs' ] = 2000


# simOptions[ 'trainBeforeDecay' ] = simOptions[ 'trainAfterDecay' ] = True


currSimOptions = simOptions.copy()


varType = varName = 'maxIter'
# currSimOptions['encodingAlpha'] = 0.9                     
varValList = [ 1, 5, 10, 15, 20,   30, 40, 50 ]





# currSimOptions['encodingAlpha'] = 0.9                     
# currSimOptions['retentionAlpha'] = 0.001                    



# currSimOptions['parameters']['nEpochs'] = 2000
# currSimOptions['nRetenSteps'] = 500                     ## saveDir indicates TRAINING nRetenSteps
# # currSimOptions['parameters']['nRetenSteps'] = currSimOptions['nRetenSteps']


# maxIterList = [  1,  5,  10,  15,  20,  25  ]
# # maxIterList = [  10,  20,  25,  50  ]
# maxIterList = [  10, 25,  50,  100,  200  ]
# # maxIterList = [  10,  20,  25  ]






#=========================================================================================
''' Load and Analyze '''
#========================================================================================= 
[ testingModelsDict, testingDataDict, testingInputDict ] = loadInDataByVarVal( varValList, currSimOptions, varName )


epochNum = simOptions['parameters']['nEpochs']
# epochNum = simOptions['epochNumsToTest'][-3]
varValAnalysisDict = analyzeVarValData( testingModelsDict, testingDataDict, testingInputDict, currSimOptions, 
                                                             epochNum=epochNum, 
                                                             FH=FH,
                                                             compareToRefState=compareToRefState, 
                                                             )


#% %
#=========================================================================================
''' Plot '''
#========================================================================================= 


''' Reorganize the dict '''
#-------------------------------------------------------------------
varValRecoveryPerfsDict = { 'recovPerfDict' : {},
                            'recovPerfRetenDict' : {}  
                           }

for varVal in varValList:
    # varValRecoveryPerfsDict[ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfDict' ]
    varValRecoveryPerfsDict['recovPerfDict'][ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfDict' ]
    varValRecoveryPerfsDict['recovPerfRetenDict'][ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfRetenDict' ]
    



''' Make the figure '''
#-------------------------------------------------------------------
nBetas = testingInputDict[ varValList[-1] ].nBetas
    


if varName == 'maxIter':
    colorbarLabel = r'Training iterations per $\beta$'
elif varName == 'nRetenSteps':
    colorbarLabel = r'Length of retention phase'
elif varName == 'nEncodSteps':
    colorbarLabel = r'Length of encoding phase'
elif varName == 'retentionAlpha':
    colorbarLabel = r'Retention $\alpha$'
    
    
#% %
    
varValFig_reten = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH, endOfEncoding=False ) 
varValFig = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH ) 
# varValFig = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH, compareToRefState=True ) 

alphaRefStr = makeAlphaRefStr(  simOptions['encodingAlpha'],  simOptions['retentionAlpha']  )

varValFig_reten.suptitle( varValFig_reten.get_suptitle() + '\n' + alphaRefStr )
varValFig.suptitle( varValFig.get_suptitle() + '\n' + alphaRefStr )





#%%

baseFilename = 'varyVar/RP_varying-' + varName + '_'

for fileType in [ '.jpeg', '.svg' ]:
    longFilename = getLongFilename(  simOptions,  baseFilename,  fileType  )
    varValFig.savefig( longFilename, bbox_inches='tight' )
    varValFig_reten.savefig(  longFilename.replace( 'RP_varying', 'RP-reten_varying' ), bbox_inches='tight'  )





# ''' Figure Title '''
# #-----------------------------------------
# wcStr = weightComboToReadable( weightCombos[0] )
# titleStr = wcStr

# alphaRefStrs = makeAlphaRefStrs( simOptions=currSimOptions )
# titleStr = alphaRefStrs[0]  + '  -  ' + wcStr
# # titleStr = 'epochNum={} \n'.format(epochNum)   +   titleStr


# varValFig.suptitle(  titleStr  )














#=========================================================================================
#=========================================================================================

def loadInMaxIterData( maxIterList, simOptions ): 
    
    
    testingModelsDict = { }
    testingDataDict = { }
    testingInputDict = { }
    
    
    
    for maxIter in maxIterList:
            
            
        #---------------------------------------------
        ''' Update the settings '''
        #---------------------------------------------
        currSimOptions = simOptions.copy() 
        # currSimOptions[ 'parameters' ][ 'maxIter' ] = maxIter
        currSimOptions[ 'maxIter' ] = maxIter
    
        
        #---------------------------------------------
        ''' Find saved data & load it in '''
        #---------------------------------------------
        # saveDir = nameSaveDir( currSimOptions, weightCombos )
        saveDir = nameSaveDir( currSimOptions )
    
        modelInfoDict = getModelInfo( saveDir )
    
        
        #---------------------------------------------
        ''' Copy to new dictionary '''
        #---------------------------------------------
        testingModelsDict[ maxIter ] = modelInfoDict[ 'epochModels' ]
        
        testingDataDict[ maxIter ] = modelInfoDict[ 'testingData' ]
        
        testingInputDict[ maxIter ] = modelInfoDict[ 'testingInput' ] 

    
    
    return testingModelsDict, testingDataDict, testingInputDict





#=========================================================================================
#=========================================================================================



def analyzeMaxIterData( testingModelsDict, testingDataDict, testingInputDict, simOptions, 
                                       epochNum=None, FH=8 ): 
    
    
    maxIters = list( testingModelsDict.keys() )
    
    
    maxIterAnalysisDict = { }
    
    if epochNum is None:
        epochNum = simOptions['parameters']['nEpochs']
        # epochNum = simOptions['epochNumsToTest'][-3]
    
    
    for maxIter in maxIters: 
        
        maxIterAnalysisDict[ maxIter ] = { }
        
        #-------------------------------------------------------------------------------------
        ''' Grab the data '''
        #-------------------------------------------------------------------------------------
        epochModels = testingModelsDict[ maxIter ]
        testingData = testingDataDict[ maxIter ]
        testingInput = testingInputDict[ maxIter ]
        
        
        #-------------------------------------------------------------------------------------
        ''' Analysis '''
        #-------------------------------------------------------------------------------------
        analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH )
    
    
        #-------------------------------------------------------------------------------------
        ''' Save to new dictionary '''
        #-------------------------------------------------------------------------------------
        maxIterAnalysisDict[ maxIter ][ 'recovPerfDict' ] = analysisDict[ 'recovPerfDict' ][ epochNum ]
        maxIterAnalysisDict[ maxIter ][ 'reconRDict' ] = analysisDict[ 'reconRDict' ][ epochNum ]
        # alphaAnalysisDict[ alphaRefStr ][ 'recovPerfDict' ] = analysisDict[ 'recovPerfDict' ]
    
    
    
    return maxIterAnalysisDict

#=========================================================================================
#=========================================================================================

















# #=========================================================================================
# ''' Load and Analyze '''
# #========================================================================================= 
# [ testingModelsDict, testingDataDict, testingInputDict ] = loadInMaxIterData( maxIterList, currSimOptions )


# epochNum = simOptions['parameters']['nEpochs']
# epochNum = simOptions['epochNumsToTest'][-3]
# maxIterAnalysisDict = analyzeMaxIterData( testingModelsDict, testingDataDict, testingInputDict, currSimOptions, 
#                                                              epochNum=epochNum )


# #% %
# #=========================================================================================
# ''' Plot '''
# #========================================================================================= 


# ''' Reorganize the dict '''
# #-------------------------------------------------------------------
# maxIterRecoveryPerfsDict = { }

# for maxIter in maxIterList:
    
#     maxIterRecoveryPerfsDict[ maxIter ] = maxIterAnalysisDict[ maxIter ][ 'recovPerfDict' ]
    



# ''' Make the figure '''
# #-------------------------------------------------------------------
# nBetas = testingInputDict[ maxIterList[-1] ].nBetas
    
# FH = 8
# colorbarLabel = r'Training iterations per $\beta$'
# maxIterFig = plotRP( maxIterRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH ) 



# ''' Figure Title '''
# #-----------------------------------------
# wcStr = weightComboToReadable( weightCombos[0] )
# titleStr = wcStr

# alphaRefStrs = makeAlphaRefStrs( simOptions=currSimOptions )
# titleStr = alphaRefStrs[0]  + '  -  ' + wcStr
# titleStr = 'epochNum={} \n'.format(epochNum)   +   titleStr


# maxIterFig.suptitle(  titleStr  )







#=========================================================================================
#=========================================================================================
#%% Retention length 
#=========================================================================================
#=========================================================================================

compareToRefState = True
# compareToRefState = False


FH = 12


currSimOptions = simOptions.copy() 


# currSimOptions['parameters']['nEpochs'] = 2000
# currSimOptions['nEncodSteps'] = 25                     
# # currSimOptions['nRetenSteps'] = 200                     
# currSimOptions['nRetenSteps'] = 50                     


# varName = 'nRetenSteps'
# varValList = [ 25, 50, 100, 200 ]


varName = 'retentionAlpha'
varValList = [ 0.002, 0.02, 0.2 ]
varValList = [  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09  ]



# varName = 'nEncodSteps'
# varValList = [ 25, 50, 100, 200, 300 ]
# # varValList = [ 5, 10, 15, 20, 25, 50, 100, 200, 300 ]
# # varValList = [ 100, 200, 300 ]




#% %

#=========================================================================================
''' Load and Analyze '''
#========================================================================================= 
[ testingModelsDict, testingDataDict, testingInputDict ] = loadInDataByVarVal( varValList, currSimOptions, varName )


epochNum = simOptions['parameters']['nEpochs']
# epochNum = simOptions['epochNumsToTest'][-3]
varValAnalysisDict = analyzeVarValData( testingModelsDict, testingDataDict, testingInputDict, currSimOptions, 
                                                             epochNum=epochNum, 
                                                             FH=FH,
                                                             compareToRefState=compareToRefState, 
                                                             )


#% %
#=========================================================================================
''' Plot '''
#========================================================================================= 


''' Reorganize the dict '''
#-------------------------------------------------------------------
varValRecoveryPerfsDict = { 'recovPerfDict' : {},
                            'recovPerfRetenDict' : {}  
                           }

for varVal in varValList:
    # varValRecoveryPerfsDict[ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfDict' ]
    varValRecoveryPerfsDict['recovPerfDict'][ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfDict' ]
    varValRecoveryPerfsDict['recovPerfRetenDict'][ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfRetenDict' ]
    



''' Make the figure '''
#-------------------------------------------------------------------
nBetas = testingInputDict[ varValList[-1] ].nBetas
    


if varName == 'maxIter':
    colorbarLabel = r'Training iterations per $\beta$'
elif varName == 'nRetenSteps':
    colorbarLabel = r'Length of retention phase'
elif varName == 'nEncodSteps':
    colorbarLabel = r'Length of encoding phase'
elif varName == 'retentionAlpha':
    colorbarLabel = r'Retention $\alpha$'
    
    
#% %
    
varValFig_reten = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH, endOfEncoding=False ) 
varValFig = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH ) 
# varValFig = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH, compareToRefState=True ) 

alphaRefStr = makeAlphaRefStr(  simOptions['encodingAlpha'],  simOptions['retentionAlpha']  )

varValFig_reten.suptitle( varValFig_reten.get_suptitle() + '\n' + alphaRefStr )
varValFig.suptitle( varValFig.get_suptitle() + '\n' + alphaRefStr )






baseFilename = 'varyVar/RP_varying-' + varName + '_'

for fileType in [ '.jpeg', '.svg' ]:
    longFilename = getLongFilename(  simOptions,  baseFilename,  fileType  )
    varValFig.savefig( longFilename, bbox_inches='tight' )
    varValFig_reten.savefig(  longFilename.replace( 'RP_varying', 'RP-reten_varying' ), bbox_inches='tight'  )




# ''' Figure Title '''
# #-----------------------------------------
# wcStr = weightComboToReadable( weightCombos[0] )
# titleStr = wcStr

# alphaRefStrs = makeAlphaRefStrs( simOptions=currSimOptions )
# titleStr = alphaRefStrs[0]  + '  -  ' + wcStr
# # titleStr = 'epochNum={} \n'.format(epochNum)   +   titleStr


# varValFig.suptitle(  titleStr  )










































#=========================================================================================
#=========================================================================================
#%% Encoding Alpha 
#=========================================================================================
#=========================================================================================

compareToRefState = True
# compareToRefState = False

FH = 12



currSimOptions = simOptions.copy() 


varType = varName = 'encodingAlpha'
# currSimOptions['maxIter'] = 50                   

varValList =  [  0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0  ]





#%%

#=========================================================================================
''' Load and Analyze '''
#========================================================================================= 
[ testingModelsDict, testingDataDict, testingInputDict ] = loadInDataByVarVal( varValList, currSimOptions, varName )


epochNum = simOptions['parameters']['nEpochs']
# epochNum = simOptions['epochNumsToTest'][-3]
varValAnalysisDict = analyzeVarValData( testingModelsDict, testingDataDict, testingInputDict, currSimOptions, 
                                                             epochNum=epochNum, 
                                                             FH=FH,
                                                             compareToRefState=compareToRefState, 
                                                             )


#% %
#=========================================================================================
''' Plot '''
#========================================================================================= 


''' Reorganize the dict '''
#-------------------------------------------------------------------
varValRecoveryPerfsDict = { 'recovPerfDict' : {},
                            'recovPerfRetenDict' : {}  
                           }

for varVal in varValList:
    # varValRecoveryPerfsDict[ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfDict' ]
    varValRecoveryPerfsDict['recovPerfDict'][ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfDict' ]
    varValRecoveryPerfsDict['recovPerfRetenDict'][ varVal ] = varValAnalysisDict[ varVal ][ 'recovPerfRetenDict' ]
    



''' Make the figure '''
#-------------------------------------------------------------------
nBetas = testingInputDict[ varValList[-1] ].nBetas
    



alphaRefStr = makeAlphaRefStr(  simOptions['encodingAlpha'],  simOptions['retentionAlpha']  )
suptitleAdd = '\n' + alphaRefStr


cmap = mpl.colormaps['hsv']


if varName == 'maxIter':
    colorbarLabel = r'Training iterations per $\beta$'
    cmap = mpl.colormaps['hsv']

    
elif varName == 'nRetenSteps':
    colorbarLabel = r'Length of retention phase'
    # cmap = mpl.colormaps['Oranges']
    cmap = mpl.colormaps['spring']
elif varName == 'nEncodSteps':
    colorbarLabel = r'Length of encoding phase'
    # cmap = mpl.colormaps['Purples']
    cmap = mpl.colormaps['cool']
    
elif varName == 'retentionAlpha':
    colorbarLabel = r'Retention $\alpha$'
    suptitleAdd = '\n e' +  str(simOptions['encodingAlpha'] )
    # cmap = mpl.colormaps['Reds']
    cmap = mpl.colormaps['autumn']
elif varName == 'encodingAlpha':
    colorbarLabel = r'Encoding $\alpha$'
    suptitleAdd = '\n r' +  str(simOptions['retentionAlpha'] )
    # cmap = mpl.colormaps['Blues']
    cmap = mpl.colormaps['winter']
    

nColors = len( varValList )
nToSample = nColors 
# nToSample = nColors + 2
# nToSample = nColors + 5
# nToSample = nColors * 2
cmap = cmap.resampled(  nToSample  )            ## + to avoid the too-light colors at the end 

# colors = [  cmap(nColors-i) for i in range( nColors )  ]    
colors = [  cmap(nToSample-i) for i in range( nColors )  ]
# colors = [  cmap(i/nColors) for i in range( nColors )  ]    
# colors = [  cmap(i/nToSample) for i in range( nColors )  ]    
# colors = [  cmap(i) for i in range( nColors )  ]    
# 

# colors.reverse()    
    


#% %  
# varValFig_reten = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH, endOfEncoding=False ) 
# varValFig = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH ) 

varValFig_reten = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH, endOfEncoding=False, colors=colors ) 
varValFig = plotRP( varValRecoveryPerfsDict, nBetas, colorbarLabel=colorbarLabel, forwardHorizon=FH, colors=colors ) 


varValFig_reten.suptitle(  varValFig_reten.get_suptitle() + suptitleAdd  )





baseFilename = 'varyVar/RP_varying-' + varName + '_'

for fileType in [ '.jpeg', '.svg' ]:
    longFilename = getLongFilename(  simOptions,  baseFilename,  fileType  )
    varValFig.savefig( longFilename, bbox_inches='tight' )
    varValFig_reten.savefig(  longFilename.replace( 'RP_varying', 'RP-reten_varying' ), bbox_inches='tight'  )




# ''' Figure Title '''
# #-----------------------------------------
# wcStr = weightComboToReadable( weightCombos[0] )
# titleStr = wcStr

# alphaRefStrs = makeAlphaRefStrs( simOptions=currSimOptions )
# titleStr = alphaRefStrs[0]  + '  -  ' + wcStr
# # titleStr = 'epochNum={} \n'.format(epochNum)   +   titleStr


# varValFig.suptitle(  titleStr  )
























#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:01:06 2024

@author: bethannajones
"""


#=========================================================================================
#=========================================================================================
#%% IMPORT
#=========================================================================================
#=========================================================================================

import numpy as np
import torch

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from functools import partial


from retenAnalysisFuncs import * 



#----------------------------------------
''' Rainbow color dict '''
#----------------------------------------

colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]



colorDict = { }

for i in range( len(colors) ):
    colorDict[ i ] = colors[ i ] 



#% %

# #----------------------------------------
# ''' Base simOptions '''
# #----------------------------------------

# simOptions[ 'encodingAlpha' ] = 0.9
# simOptions[ 'retentionAlpha' ] = 0.1
# simOptions[ 'parameters' ][ 'nEpochs' ] = 2000


# initState = torch.tensor([[0.4581, 0.4829, 0.3125, 0.6150, 0.2139, 0.4118, 0.6938, 0.9693, 0.6178,
#           0.3304, 0.5479, 0.4440, 0.7041, 0.5573, 0.6959]])
# simOptions[ 'parameters' ][ 'initState' ] = initState


# simOptions[ 'maxIter' ] = 50







#=========================================================================================
#=========================================================================================
#%% 
#=========================================================================================
#=========================================================================================


simOptions[ 'encodingAlpha' ] = 1


varType = 'retentionAlpha'
# varType = 'encodingAlpha' 

varValList = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] 
varValList = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] 
varValList = [ 0.1, 0.2 ] 



# iterateOverSimParam( simOptions, varValList, varType=varType,
#                                     # trainAndTest=False, autoSaveCheck=True, 
#                                     testOnly=False, autoSave=True, 
#                                     nTestBetas=50 )





testOnly = False
autoSave = True
nTestBetas = 50 


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
    
    nEpochs = simOptions['parameters']['nEpochs'] 
    nEpochsKeyBool = nEpochs in testingData.keys()

    
    if nEpochsKeyBool:
        
        ## Play a sound when saving
        sys.stdout.write('\a')
        sys.stdout.flush()
        
        
        # saveDir = saveBasedOnSimOptions( simOptions, locals() )
        runcell('SAVE, #1', '/Users/bethannajones/Desktop/PGM/PGM_main.py')   
        # continue
    
    else: 
        sendEmailWithStatus( subject='Did not save: e{}r{}'.format( simOptions['encodingAlpha'], simOptions['retentionAlpha'] ), body=saveDir )
    
    print(  '----------------------------------------------'  )






#%%

saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# saveDir = '/Users/bethannajones/Desktop/PGM/varsAndFigs/retrainOnStim50/eAlp1rAlp0.9/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# saveDir = 'varsAndFigs/retrainOnStim50/eAlp1rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'


    
simOptions = getModelInfo( saveDir )[ 'simOptions' ]


epochNumsToTest = simOptions['epochNumsToTest']
epochModels = getModelInfo( saveDir )[ 'epochModels' ]

testingWasInterrupted = False


trainingModel = getModelInfo( saveDir )[ 'trainingModel' ]
trainingInput = getModelInfo( saveDir )[ 'trainingInput' ]
trainingData = getModelInfo( saveDir )[ 'trainingData' ]

epochNum = nEpochs = 2000




#%% Test: long delay for different rA


''' The data directory '''
#----------------------------------------------
saveDir = nameSaveDir( simOptions )


# saveDir = 'varsAndFigs/retrainOnStim50/eAlp1rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'


saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
alphaRefStr = makeAlphaRefStr( 0.9, 0.1 )
origRetenSteps = 100



# saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.02/eSteps50rSteps0/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# alpahRefStr = makeAlphaRefStr( 0.9, 0.02 )
# origRetenSteps = 0





retrainRefExample = True

if retrainRefExample:
    # alphaRefStr = makeAlphaRefStr(  )
    currAlpDir = 'eA{}_extDelay'.format(simOptions['encodingAlpha'])  +  '_retest-' + alphaRefStr.replace('_', '-')  +  '-delay{}'.format(origRetenSteps)
else:
    currAlpDir = 'eA{}_extDelay'.format(simOptions['encodingAlpha'])


if not os.path.exists( currAlpDir ):
    os.makedirs( currAlpDir )


makeGIFs = True
makeGIFs = False


encodPCs = True
# encodPCs = False


analyzeOnly = True
# analyzeOnly = False



#% %

''' Set length of extended delay period '''
#----------------------------------------------
totalDelayLen = 1000
totalDelayLen = 700
# totalDelayLen = 500


nTestingEncodSteps = 150
nTestingEncodSteps = 50



''' Stimuli '''
#----------------------------------------------
nTestBetas = 200
nTestBetas = 100
# nTestBetas = 400
# nTestBetas = 300
inputSaveName = 'testingInput_{}'.format( nTestBetas )

testingInput = getModelInfo( 'currTestingInfo', varNames=['testingInput'], saveNames=[ inputSaveName ] )




# ''' Principal components from initial sim '''
# #----------------------------------------------
# if simOptions['nRetenSteps'] == simOptions['nRetenSteps_ext']:
# #     shortSimState = testingData[ nEpochs ][ 'state' ]
#     shortSimState = getModelInfo( saveDir )[ 'testingData' ][ nEpochs ][ 'state' ]
#     PCs = torch.pca_lowrank( shortSimState.T )[2]
# else: 
#     raise Exception( 'Current loaded testing data does not satisfy nRetenSteps = nRetenSteps_ext' )


#%%


varType = 'retentionAlpha'
varValList = [  0.1,  0.2  ]
# varValList = [  0.2, 0.3, 0.4, 0.5, 0.6, 0.7  ]
varValList = [  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7  ]
# varValList = [  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09 ]
# varValList = [  0.04, 0.05, 0.06, 0.07, 0.08, 0.09 ]
# varValList = [  0.03 ]
# varValList = [  0.1  ]
varValList = [  0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29  ]
# varValList = [  0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29  ]




#% %
#=========================================================================================
''' Run a long test for each value of rA '''
#=========================================================================================
for varVal in varValList:
    
    print( '----' )
    print( varType, ' - ', varVal  )
    print( '----' )

    
    
    if varType == 'retentionAlpha':
        varValStr = 'rA' + str( varVal )
    
    
    if retrainRefExample:
        simOptions = getModelInfo( saveDir )[ 'simOptions' ]

        newSimOptions = simOptions.copy()
        newSimOptions[ varType ] = varVal
        # newSimOptions[ 'nEncodSteps_testing' ] = nTestingEncodSteps

        saveDir_new = nameSaveDir( newSimOptions )
        saveDir_testing = saveDir_new + '/testing_rSteps' + str(totalDelayLen) 
        saveDir_testing = saveDir_testing + 'eSteps' + str(nTestingEncodSteps) 
        
        print( '\nWill load...', saveDir )
        print( 'Will edit + save to...', saveDir_testing )
        
        # simOptions[ 'nEncodSteps' ] = newSimOptions[ 'nEncodSteps_testing' ] = 150


        # simOptions[ varType ] = varVal

    else:
        simOptions[ varType ] = varVal
        saveDir = nameSaveDir( simOptions )
        saveDir_testing = saveDir + '/testing_rSteps' + str(totalDelayLen)
        saveDir_testing = saveDir_testing + 'eSteps' + str(nTestingEncodSteps) 

        simOptions[ 'nEncodSteps_testing' ] = nTestingEncodSteps



    # ''' Principal components from initial sim '''
    # #----------------------------------------------
    # if simOptions['nRetenSteps'] == simOptions['nRetenSteps_ext']:
        
    #     # shortSimState = getModelInfo( saveDir )[ 'testingData' ][ nEpochs ][ 'state' ]
    #     origSimState = testingData[ nEpochs ][ 'state' ]
        
    #     if encodPCs:
    #         stimTimeInds = testingData['stimTimeInds']        
    #         [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( testingData['nEncodSteps'], simOptions['nRetenSteps'], stimTimeInds   )
    #         encodingStates = origSimState[ :, encodingTimeInds ]
    #         PCs = torch.pca_lowrank( encodingStates.T )[2] 
    #     else:
    #         PCs = torch.pca_lowrank( shortSimState.T )[2]
        
    # else: 
    #     raise Exception( 'Current loaded testing data does not satisfy nRetenSteps = nRetenSteps_ext' )
    
    
    
    ''' Principal components from initial sim '''
    #----------------------------------------------
    origTestingData = getModelInfo( saveDir )[ 'testingData' ]
    origSimState = origTestingData[ nEpochs ][ 'state' ] 
    # origSimState = testingData[ nEpochs ][ 'state' ]
    
    if encodPCs:
        stimTimeInds = origTestingData['stimTimeInds']        
        [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( origTestingData['nEncodSteps'], simOptions['nRetenSteps'],  stimTimeInds   )
        encodingStates = origSimState[ :, encodingTimeInds ]
        PCs = torch.pca_lowrank( encodingStates.T )[2] 
    else:
        PCs = torch.pca_lowrank( origSimState.T )[2]
        


    #=====================================================================================
    
    
    if not analyzeOnly:
        
        ''' Load trained data dir '''
        #----------------------------------------------
        runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
        
        
        ''' Test it with a long delay time '''
        #----------------------------------------------
        simOptions['nRetenSteps_ext'] = totalDelayLen
        
        if retrainRefExample:
            simOptions[ varType ] = varVal
            
            simOptions[ 'nEncodSteps_testing' ] = nTestingEncodSteps

            # simOptions[ 'nEncodSteps' ] = simOptions[ 'nEncodSteps_testing' ] = 150
            # simOptions[ 'nEncodSteps_testing' ] = 150

    
        testingInput = getModelInfo( 'currTestingInfo', varNames=['testingInput'], saveNames=[ inputSaveName ] )
        
        simOptions[ 'loadPrevData' ] = True
        runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
        simOptions[ 'loadPrevData' ] = False
    
    
    
        ''' Save the data '''
        #----------------------------------------------
        nEpochs = simOptions['parameters']['nEpochs'] 
        nEpochsKeyBool = nEpochs in testingData.keys() 
    
        if nEpochsKeyBool:
            filenames = saveModelInfo(   locals(),   saveDir_testing   ) 
        
        else: 
            continue
        
        
        
        
        
        
    else: 
        
        ''' Load extended delay data dir '''
        #----------------------------------------------
        print( '\nLoading...', saveDir_testing )
        modelInfoDict = getModelInfo( saveDir_testing )                 ## Load the data 
        
        simOptions = modelInfoDict[ 'simOptions' ]
        testingData = modelInfoDict[ 'testingData' ]
        testingInput = modelInfoDict[ 'testingInput' ]
        
        
        # simOptions['nRetenSteps_ext'] = totalDelayLen
        
    

    #=====================================================================================
    
    
    
    
    #% %
    #=====================================================================================
    ''' PCA of state activity '''
    #=====================================================================================
    
    # extFilenameBase = 'ext'   +   str( varVal - simOptions['nRetenSteps'] )
    extFilenameBase = 'ext'   +   str(simOptions['nRetenSteps_ext'])
    
    
    if makeGIFs:
        
        ''' Make and save a GIF '''
        #----------------------------------------------
        # saveDirList = [ saveDir_testing, 'extendedRetention' ]
        saveDirList = [ saveDir_testing, 'extendedRetention', currAlpDir ]
        
        ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                        # trimRetentionLen=trimAnimationLen, 
                                                        saveDir=saveDirList,
                                                        filenameBase = extFilenameBase,
                                                        )
    
    
    
    ''' Grab the final states (end of encoding, end of delay) '''
    #----------------------------------------------
    timePtTypes = [ 'endOfEncoding', 'endOfRetention_trained', 'endOfRetention_extended' ]
    figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
                                             sameFig=False, timePtTypes=timePtTypes )
    
    
    
    for i in range( len(timePtTypes) ):
        fig = figList[ i ]
        # fig.savefig( extDir + timePtTypes[i] + '.jpg' ) 
        # fig.savefig( extFilenameBase + timePtTypes[i] + '.jpg' ) 
        
        fig.savefig( saveDir_testing + '/' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 
        fig.savefig( currAlpDir + '/' + varValStr + '_' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 
        # fig.savefig( saveDir_testing + '/' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 
        # fig.savefig( currAlpDir + '/' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 

    


    
    

    ''' Apply short test PCA to ext test '''
    #----------------------------------------------
    # extFilenameBase = 'ext'   +    str( varVal - simOptions['nRetenSteps'] )   +   '-origPCA_'
    
    if encodPCs:
        extFilenameBase = 'ext'   +   str(simOptions['nRetenSteps_ext'])   +   '-encodingPCA_'
    else:
        extFilenameBase = 'ext'   +   str(simOptions['nRetenSteps_ext'])   +   '-origPCA_'
    
    
    if makeGIFs:
        ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                        
                                                        PCsToUse=PCs,
                                                    
                                                        # trimRetentionLen=trimAnimationLen, 
                                                        saveDir=saveDirList,
                                                        filenameBase=extFilenameBase,
                                                        )
    
    
    figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
                                             sameFig=False, timePtTypes=timePtTypes, 
                                             PCsToUse=PCs,
                                             )
    
    for i in range( len(timePtTypes) ):
        fig = figList[ i ]
        fig.savefig( extFilenameBase + timePtTypes[i] + '.jpg' ) 
        # fig.savefig( saveDir_testing + extFilenameBase + timePtTypes[i] + '.jpg' ) 
        # fig.savefig( currAlpDir + extFilenameBase + timePtTypes[i] + '.jpg' ) 
        
        fig.savefig( saveDir_testing + '/' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 
        fig.savefig( currAlpDir + '/' + varValStr + '_' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 
        # fig.savefig( saveDir_testing + '/' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 
        # fig.savefig( currAlpDir + '/' + extFilenameBase + '_' + timePtTypes[i] + '.jpg' ) 

    
    
    
    

    '''  Plot converged states -- at origin? '''
    #----------------------------------------------
    fig, axs = plt.subplots( 1, 1 ) 


    convergedReten = testingData[ epochNum ][ 'convergedStates_retention' ].detach().numpy()
    [ N, nStates ] = convergedReten.shape
    
    
    xPts = [  list(range(nTestBetas))  ] * N
    
    axs.scatter( xPts, convergedReten[:,1::], s=8 )
    axs.set_ylabel( r'$\mathbf{r}(t_k + \epsilon + \delta)$' )
    axs.set_xlabel( r'Stimulus index $k$' )
    
    fig.suptitle( 'Extended delay state values' )
    
    
    filename = getLongFilename( simOptions, 'convergedStateVals_', '.jpg' )
    fig.savefig(  currAlpDir + '/' + varValStr + '_' + extFilenameBase + filename  ) 

    
    
    

    print(  '----------------------------------------------'  )
    
    
    
    



#%% Test: long delay (single exmaple)


''' The data directory '''
#----------------------------------------------
# saveDir = nameSaveDir( simOptions )
# currAlpDir = 'eA{}_extDelay'.format(simOptions['encodingAlpha'])


# saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
saveDir = 'varsAndFigs/retrainOnStim50/eAlp1rAlp0.2/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
simOptions = getModelInfo( saveDir )[ 'simOptions' ]
runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

currAlpDir = 'eA{}_extDelay'.format(simOptions['encodingAlpha'])




#% %

''' Set length of extended delay period '''
#----------------------------------------------
totalDelayLen = 1000
totalDelayLen = 700
totalDelayLen = 500



''' Stimuli '''
#----------------------------------------------
nTestBetas = 200
inputSaveName = 'testingInput_{}'.format( nTestBetas )

testingInput = getModelInfo( 'currTestingInfo', varNames=['testingInput'], saveNames=[ inputSaveName ] )




# ''' Principal components from initial sim '''
# #----------------------------------------------
# if simOptions['nRetenSteps'] == simOptions['nRetenSteps_ext']:
#     shortSimState = testingData[ nEpochs ][ 'state' ]
#     PCs = torch.pca_lowrank( shortSimState.T )[2]
# else: 
#     raise Exception( 'Current loaded testing data does not satisfy nRetenSteps = nRetenSteps_ext' )



varValList = [  0.1,  0.2  ]
varValList = [  0.2  ]


#% %
#=========================================================================================
''' Run a long test for each value of rA '''
#=========================================================================================
for varVal in varValList:
    
    
    # ''' Load trained data dir '''
    # #----------------------------------------------
    # simOptions[ 'retentionAlpha' ] = varVal
    # runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
    
    
    ''' Update parameters '''
    #----------------------------------------------
    simOptions[ 'retentionAlpha' ] = varVal
    
    
    
    ''' Test it with a long delay time '''
    #----------------------------------------------
    simOptions['nRetenSteps_ext'] = totalDelayLen

    testingInput = getModelInfo( 'currTestingInfo', varNames=['testingInput'], saveNames=[ inputSaveName ] )
    
    simOptions[ 'loadPrevData' ] = True
    runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    simOptions[ 'loadPrevData' ] = False



    ''' Save the data '''
    #----------------------------------------------
    nEpochs = simOptions['parameters']['nEpochs'] 
    nEpochsKeyBool = nEpochs in testingData.keys()

    if nEpochsKeyBool:
        print( )
        saveDir_testing = saveDir + '/testing_rSteps' + str(varVal)
        fullExtTestingDir = os.path.join( currDir, saveDir_testing ) 
    
        filenames = saveModelInfo(   locals(),   fullExtTestingDir   )
    
    else: 
        continue
    
    #=====================================================================================
    
    #% %
    ''' Make and save a GIF '''
    #----------------------------------------------
    # extFilenameBase = 'ext'   +   str( varVal - simOptions['nRetenSteps'] )
    extFilenameBase = 'ext'   +   str(simOptions['nRetenSteps_ext'])
    # saveDirList = [ saveDir_testing, 'extendedRetention' ]
    saveDirList = [ saveDir_testing, 'extendedRetention', currAlpDir ]
    
    ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                    # trimRetentionLen=trimAnimationLen, 
                                                    saveDir=saveDirList,
                                                    filenameBase = extFilenameBase,
                                                    )
    
    
    
    ''' Grab the final states (end of encoding, end of delay) '''
    #----------------------------------------------
    timePtTypes = [ 'endOfEncoding', 'endOfRetention_trained', 'endOfRetention_extended' ]
    figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
                                             sameFig=sameFig, timePtTypes=timePtTypes )
    
    for i in range( len(timePtTypes) ):
        fig = figList[ i ]
        # fig.savefig( extDir + timePtTypes[i] + '.jpg' ) 
        fig.savefig( extFilenameBase + timePtTypes[i] + '.jpg' ) 
        fig.savefig( saveDir_testing + extFilenameBase + timePtTypes[i] + '.jpg' ) 
        fig.savefig( currAlpDir + extFilenameBase + timePtTypes[i] + '.jpg' ) 

    


    
    

    ''' Apply short test PCA to ext test '''
    #----------------------------------------------
    # extFilenameBase = 'ext'   +    str( varVal - simOptions['nRetenSteps'] )   +   '-origPCA_'
    extFilenameBase = extFilenameBase   +   '-origPCA_'
    
    ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                    
                                                    PCsToUse=PCs,
                                                
                                                    # trimRetentionLen=trimAnimationLen, 
                                                    saveDir=saveDirList,
                                                    filenameBase=extFilenameBase,
                                                    )
    
    
    figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
                                             sameFig=sameFig, timePtTypes=timePtTypes, 
                                             PCsToUse=PCs,
                                             )
    
    for i in range( len(timePtTypes) ):
        fig = figList[ i ]
        fig.savefig( extFilenameBase + timePtTypes[i] + '.jpg' ) 
        fig.savefig( saveDir_testing + extFilenameBase + timePtTypes[i] + '.jpg' ) 
        fig.savefig( currAlpDir + extFilenameBase + timePtTypes[i] + '.jpg' ) 

    
    
    
    

    '''  Plot converged states -- at origin? '''
    #----------------------------------------------
    fig, axs = plt.subplots( 1, 1 ) 


    convergedReten = testingData[ epochNum ][ 'convergedStates_retention' ].detach().numpy()
    [ N, nStates ] = convergedReten.shape
    
    
    xPts = [  list(range(nTestBetas))  ] * N
    
    axs.scatter( xPts, convergedReten[:,1::], s=8 )
    axs.set_ylabel( r'$\mathbf{r}(t_k + \epsilon + \delta)$' )
    axs.set_xlabel( r'Stimulus index $k$' )
    
    
    filename = getLongFilename( simOptions, 'convergedStates', '.jpg' )
    fig.savefig(  currAlpDir + '/convergedStateVals' + filename  ) 

    
    
    

    print(  '----------------------------------------------'  )
    
    
    
    
    
    
    
    
    


#%%


simOptions['encodingAlpha'] = 0.9


varType = 'retentionAlpha'
varValList = [  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8  ] 
varValList = [  0.5,  0.6,  0.7,  0.8  ] 


testingDataDict = { }


for varVal in varValList: 
    
    simOptions[ varType ] = varVal
    
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    testingDataDict[ varVal ] = testingData
    
    
    
plotAvgTimeToConvergePerVarVal( testingDataDict, simOptions, nBetas, phase='retention',
                                              epochNum=nEpochs, testing=True, tol=1e-4 )





#%% rename save folders 


varType = 'maxIter'
varValList = [ 1, 5, 10, 15, 20, 30, 40, 50 ]


dataFolder = 'varsAndFigs/initState_8067'

# newInitState = torch.tensor([[0.4581, 0.4829, 0.3125, 0.6150, 0.2139, 0.4118, 0.6938, 0.9693, 0.6178,
#          0.3304, 0.5479, 0.4440, 0.7041, 0.5573, 0.6959]])


for varVal in varValList: 
    
    simOptions[ varType ] = varVal
    
    oldSaveDir = nameSaveDir( simOptions )
    
    
    
    saveDir = nameSaveDir( simOptions ) 
    newSaveDir = saveDir.replace( 'varsAndFigs', dataFolder )
    
    
    copySaveDir( oldSaveDir, newDir=newSaveDir )



#%%

varType = 'retentionAlpha'

dataFolder = 'varsAndFigs/initState_8067'


for varVal in varValList: 
    
    simOptions[ varType ] = varVal
    
    oldSaveDir = nameSaveDir( simOptions )
    # newSaveDir = nameSaveDir( simOptions )
    
    copySaveDir( oldSaveDir, dataFolder)




#%% 

varType = 'retentionAlpha'

varType = 'encodingAlpha' 


saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
simOptions = getModelInfo( saveDir )[ 'simOptions' ]


varValList = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] 



iterateOverSimParam( simOptions, varValList, varType=varType,
                                    # trainAndTest=False, autoSaveCheck=True, 
                                    testOnly=True, autoSave=True, 
                                    nTestBetas=50 )

    
    
    




#%% Test for an extended retention time


# saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.7rAlp0.02/eSteps50rSteps0/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.02/eSteps50rSteps0/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'

# saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'


# # saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.1/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.02/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.05/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'

simOptions = getModelInfo( saveDir )[ 'simOptions' ] 


runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')




varType = 'nRetenSteps_ext'
# extendLength = [ 50, 100, 200, 300, 400, 500, 600, 700, 800, 900 ] 
extendLength = [ 200, 400, 600, 800, 1000 ] 
extendLength = [ 1000 ] 

# varValList = [ 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ] 
varValList = [  simOptions['nRetenSteps'] + T  for T in extendLength  ]
# varValList = [ 150, 200, 300, 400, 500 ]
# # varValList = [ 150, 200 ]
# varValList = [ 150, 300, 400, 500, 600 ]
# # varValList = [ 700, 800, 900, 1000 ]
# varValList = [ 900, 1000 ]

# varValList = [ simOptions['nRetenSteps'] ]



inputSaveName = 'testingInput_{}'.format(nTestBetas)
testingInput = getModelInfo( referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )

#% %

for varVal in varValList:
    
    simOptions[ varType ] = varVal
    
    
    simOptions[ 'loadPrevData' ] = True
    runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    simOptions[ 'loadPrevData' ] = False

    print( )
    saveDir_testing = saveDir + '/testing_rSteps' + str(varVal)
    fullExtTestingDir = os.path.join( currDir, saveDir_testing ) 

    filenames = saveModelInfo(   locals(),   fullExtTestingDir   )
    
    
    
    #----------------------
    
    # extDir = 'extendedRetention/ext'   +   str( varVal - simOptions['nRetenSteps'] )
    # saveDirList = [ saveDir_testing, extDir ]
    
    
    
    # ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
    #                                                 # trimRetentionLen=trimAnimationLen, 
    #                                                 saveDir=saveDirList,
    #                                                 )
    
    
    # timePtTypes = [ 'endOfEncoding', 'endOfRetention_trained', 'endOfRetention_extended' ]
    # figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
    #                                          sameFig=sameFig, timePtTypes=timePtTypes )
    
    # for i in range( len(timePtTypes) ):
    #     fig = figList[ i ]
    #     fig.savefig( extDir + timePtTypes[i] + '.jpg' ) 

    
    

    print(  '----------------------------------------------'  )
    
    

#%% Load and analyze extended tests

saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.02/eSteps50rSteps0/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'

# saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.02/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'

simOptions = getModelInfo( saveDir )[ 'simOptions' ] 
testingData = getModelInfo( saveDir )[ 'testingData' ] 


#------------------------------------------------------------------------------
''' 
    Use the prin. compon. from  nRetenSteps = nRetenSteps_ext test 
(testing length = training length)
'''
#------------------------------------------------------------------------------
if simOptions['nRetenSteps'] == simOptions['nRetenSteps_ext']:
    shortSimState = testingData[ nEpochs ][ 'state' ]
    PCs = torch.pca_lowrank( shortSimState.T )[2]
else: 
    raise Exception( 'Current loaded testing data does not satisfy nRetenSteps = nRetenSteps_ext' )



varType = varName = 'nRetenSteps_ext'
varValList = [ 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ]
varValList = [ 900 ]
# varValList = [ 150, 300, 400, 500, 700, 800, 900 ]
# varValList = [ 200, 600, 1000 ]
# varValList = [ simOptions['nRetenSteps'] ]


for varVal in varValList:
    
    
    simOptions[ varType ] = varVal
    
    
    
    ''' Load from extTesting dir '''
    #----------------------------------------------
    print( )
    saveDir_testing = saveDir + '/testing_rSteps' + str(varVal)
    fullExtTestingDir = os.path.join( currDir, saveDir_testing ) 
    
    modelInfoDict = getModelInfo( fullExtTestingDir )                 ## Load the data 
    simOptions = modelInfoDict[ 'simOptions' ]
    testingData = modelInfoDict[ 'testingData' ]
    testingInput = modelInfoDict[ 'testingInput' ]
    
    
    
    ''' Animation '''
    #----------------------------------------------
    extFilenameBase = 'ext'   +   str( varVal - simOptions['nRetenSteps'] )
    saveDirList = [ saveDir_testing, 'extendedRetention' ]
    
    ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                    # trimRetentionLen=trimAnimationLen, 
                                                    saveDir=saveDirList,
                                                    filenameBase = extFilenameBase,
                                                    )
    
    
    ''' Phase time points '''
    #----------------------------------------------

    timePtTypes = [ 'endOfEncoding', 'endOfRetention_trained', 'endOfRetention_extended' ]
    figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
                                             sameFig=sameFig, timePtTypes=timePtTypes )
    
    for i in range( len(timePtTypes) ):
        fig = figList[ i ]
        # fig.savefig( extDir + timePtTypes[i] + '.jpg' ) 
        fig.savefig( extFilenameBase + timePtTypes[i] + '.jpg' ) 
        fig.savefig( saveDir_testing + extFilenameBase + timePtTypes[i] + '.jpg' ) 

    


    

    ''' Apply short test PCA to ext test '''
    #----------------------------------------------
    extFilenameBase = 'ext'   +    str( varVal - simOptions['nRetenSteps'] )   +   '-origPCA_'
    
    ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                    
                                                    PCsToUse=PCs,
                                                
                                                    # trimRetentionLen=trimAnimationLen, 
                                                    saveDir=saveDirList,
                                                    filenameBase=extFilenameBase,
                                                    )
    
    
    figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
                                             sameFig=sameFig, timePtTypes=timePtTypes, 
                                             PCsToUse=PCs,
                                             )
    
    for i in range( len(timePtTypes) ):
        fig = figList[ i ]
        fig.savefig( extFilenameBase + timePtTypes[i] + '.jpg' ) 
        fig.savefig( saveDir_testing + extFilenameBase + timePtTypes[i] + '.jpg' ) 

    



#%% apply nRetenSteps PCA to long testing period ??








#%%

# simOptions['maxIter'] = 10
# simOptions['parameters']['nEpochs'] = 2000



runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
runcell('Single sim', '/Users/bethannajones/Desktop/PGM/retentionAnalysis.py')



# nTestBetas = 75 
# simOptions[ 'nEncodSteps' ] = simOptions[ 'nEncodSteps_testing' ] = 50
# simOptions[ 'nRetenSteps' ] = simOptions[ 'nRetenSteps_ext' ] = 100



# runcell('Training', '/Users/bethannajones/Desktop/PGM/PGM_main.py') 
# runcell('Create testing models', '/Users/bethannajones/Desktop/PGM/PGM_main.py')


# runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')



# runcell('Single sim', '/Users/bethannajones/Desktop/PGM/retentionAnalysis.py')


#%%


initFolder = 'varsAndFigs/initState_8067'
# initFolder = 'varsAndFigs/initState_4581'

saveDir = initFolder + '/retrainOnStim10/eAlp0.9rAlp0.01/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'


simOptions = getModelInfo( saveDir )[ 'simOptions' ] 


runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')


ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                trimRetentionLen=trimAnimationLen, 
                                                saveDir=[saveDir, initFolder],
                                                # saveDir='varsAndFigs',
                                                )

#%% update initStates 

# varType = 'retentionAlpha'
# # varValList = [  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09  ]
# varValList = [  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09  ]



varType = 'maxIter'
varValList = [ 1, 5, 10, 15, 20, 30, 40, 50 ]


dataFolder = 'varsAndFigs/initState_8067'





# # saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.07/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# saveDir = 'varsAndFigs/retrainOnStim10/eAlp0.9rAlp0.07/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# simOptions = getModelInfo( saveDir )[ 'simOptions' ] 


newInitState = torch.tensor([[0.4581, 0.4829, 0.3125, 0.6150, 0.2139, 0.4118, 0.6938, 0.9693, 0.6178,
         0.3304, 0.5479, 0.4440, 0.7041, 0.5573, 0.6959]])




for varVal in varValList:
    
    simOptions[ varType ] = varVal

    saveDir = nameSaveDir( simOptions )
    oldSaveDir = saveDir.replace( 'varsAndFigs', dataFolder )
    
    simOptions = getModelInfo( oldSaveDir )[ 'simOptions' ] 


    ''' Update initState '''
    #---------------------------------------------------------
    simOptions[ 'parameters' ][ 'initState' ] = newInitState
    # print( simOptions[ 'parameters' ][ 'initState' ] )
    
    # print( )
    # print( oldSaveDir )
    # print( saveDir )
    
    #% %
    
    ''' Resave '''
    #---------------------------------------------------------
    copySaveDir( oldSaveDir, newDir=saveDir )



#%%
    
''' Retrain, test, and save '''
#-----------------------------------------------------------------------------------------
iterateOverSimParam( simOptions, varValList, varType='maxIter',
                                    testOnly=False, autoSave=True, 
                                    nTestBetas=200 )
    
    

#%%
    
    initState = testingData[ nEpochs ]['state'][ :, 0 ]
    # initState = simOptions['parameters']['initState']
    print( initState.T )
    
    
#% %
    
    simOptions['parameters']['initState'] = newInitState
    
    
    
    
    runcell('Training', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    runcell('Create testing models', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    
    
    testingInput = getModelInfo( referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )

    simOptions[ 'loadPrevData' ] = True
    runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    simOptions[ 'loadPrevData' ] = False
    
    
    
    
    
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




#%% Check PCA 


saveDir = 'varsAndFigs/retrainOnStim50/eAlp0.9rAlp0.07/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
# saveDir = 'varsAndFigs/retrainOnStim10/eAlp0.9rAlp0.07/eSteps50rSteps100/trainBeforeAfterDecay_WCerr4-eff0.02-s0.01-h10-r1-f0-p0/Epochs2000'
simOptions = getModelInfo( saveDir )[ 'simOptions' ] 
runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

print( simOptions['parameters']['initState'] )

# runcell('Analyze single sim', '/Users/bethannajones/Desktop/PGM/retentionAnalysis.py')



# trimLen = min(  simOptions['nRetenSteps_ext'],  simOptions['nRetenSteps']+100  )
# trimAmt = trimLen - simOptions['nRetenSteps_ext']


# ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
#                                                         trimRetentionLen=trimAmt,
#                                                         saveDir=[ 'checkPCA' ], 
#                                                         useLongFilename=True, 
#                                                         )





# #% %


# varValList = [  simOptions['retentionAlpha']  ] 

# simOptions['nEncodSteps_testing'] = simOptions['nEncodSteps'] + 25


# testingInput = getModelInfo( referenceTestingFolder, varNames=['testingInput'], saveNames=[ inputSaveName ] )

# simOptions[ 'loadPrevData' ] = True
# runcell('Testing', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
# simOptions[ 'loadPrevData' ] = False





# ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
#                                                         trimRetentionLen=trimAmt,
#                                                         saveDir=[ 'checkPCA' ], 
#                                                         useLongFilename=True, 
#                                                         )


#=========================================================================================
#=========================================================================================
#%% Analyze single sim
#=========================================================================================
#=========================================================================================


epochNum = nEpochs



printSimOptions2( simOptions )
# runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')




#-----------------------------------------------------------------------------------------
''' Analysis dictionary '''
#-----------------------------------------------------------------------------------------
FH = 12

analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                     compareToRefState=True,
                                                     # compareToRefState=False, 
                                                     )



#-----------------------------------------------------------------------------------------
''' Reconstruction performance '''
#-----------------------------------------------------------------------------------------
alphaRefStr = makeAlphaRefStr(  simOptions['encodingAlpha'],  simOptions['retentionAlpha']  )

rpFig_reten = plotRP( analysisDict, nTestBetas, FH, colors=colorDict, endOfEncoding=False  )
rpFig_reten.suptitle( rpFig_reten.get_suptitle() + '\n' + alphaRefStr )

rpFig = plotRP( analysisDict, nTestBetas, FH, colors=colorDict  )
rpFig.suptitle( rpFig.get_suptitle() + '\n' + alphaRefStr )





#% %
#-----------------------------------------------------------------------------------------
''' PCA of state evolution '''
#-----------------------------------------------------------------------------------------
sameFig = True
sameFig = False

figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, sameFig=sameFig, timePtTypes=['endOfEncoding'] )
# figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, sameFig=sameFig )


if sameFig:
    fig = figList
    
    
     
#% %
#-----------------------------------------------------------------------------------------
''' Encoding accuracy '''
#-----------------------------------------------------------------------------------------
epochModel = epochModels[ epochNum ]
fig = plotModelEncodingAccur( epochModel, testingInput, testingData, z=5 )


# fig = plotConvergedEncodingAccur( epochModel, testingInput, testingData )



#%%

#-----------------------------------------------------------------------------------------
''' Animated -- PCA of state evolution '''
#-----------------------------------------------------------------------------------------
trimLen = min(  simOptions['nRetenSteps_ext'],  simOptions['nRetenSteps']+100  )
trimAmt = trimLen - simOptions['nRetenSteps_ext']

# saveDirList = [ saveDir, 'stimLifecyclePCA', 'varsAndFigs' ]
saveDirList = [ saveDir, 'varsAndFigs/initState_4581' ]

ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                        trimRetentionLen=trimAmt,
                                                        saveDir=saveDirList, 
                                                        useLongFilename=True, 
                                                        )

 






#=========================================================================================
#=========================================================================================
#%% plot across alphas
#=========================================================================================
#=========================================================================================


# simOptions[ 'encodingAlpha' ] = 0.9



rententionAlphaVals = [  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7  ]
rententionAlphaVals = [  0.7  ]
# simOptions[ 'nRetenSteps' ] = nRetenSteps = 1000


rententionAlphaVals = [  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09  ]
rententionAlphaVals = [  0.06,  0.07,  0.08,  0.09  ]
# rententionAlphaVals = [  0.02, 0.04, 0.06, 0.08  ]
# rententionAlphaVals = [  0.08  ]
# rententionAlphaVals = [  0.06  ]
# # rententionAlphaVals = [  0.02, 0.04, 0.06  ]
# simOptions[ 'nRetenSteps' ] = nRetenSteps = 100








# simOptions['parameters']['nEpochs'] = nEpochs = 5000
# simOptions['parameters']['nEpochs'] = nEpochs = 2000
# # simOptions['parameters']['nEpochs'] = nEpochs = 3000
# # simOptions['parameters']['maxIter'] = maxIter = 10
# simOptions['maxIter'] = maxIter = 10


# epochNum = nEpochs 
# epochNum = 2000 






#%%
#-----------------------------------------------------------------------------------------
''' Stim lifecycles (color-coordinated phases) '''
#-----------------------------------------------------------------------------------------

nEncodSteps = simOptions[ 'nEncodSteps' ]

nRetenSteps = simOptions[ 'nRetenSteps_ext' ]
# nRetenSteps_training = simOptions[ 'nRetenSteps' ]



cmap1 = cmap = mpl.colormaps[ 'winter_r' ]
colors1 = colors = cmap(   [ x/nEncodSteps for x in range(nEncodSteps) ]   )


cmap2 = mpl.colormaps[ 'autumn_r' ]
colors2 = cmap2(   [ x/nRetenSteps for x in range(nRetenSteps) ]   )




for rAlph in rententionAlphaVals:

    simOptions[ 'retentionAlpha' ] = rAlph
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    
    nBetasToPlot = 20
    nBetasToPlot = int( testingInput.nBetas / 3 )
    nBetasToPlot = 3
    # nBetasToPlot = testingInput.nBetas
    nBetasToPlot = int( testingInput.nBetas / 2 )


    
    # runcell('All phases, color-coordinated', '/Users/bethannajones/Desktop/PGM/retentionAnalysis.py')
    fig = plotColorCoorPhases( epochNum, testingData, testingInput, nBetasToPlot=nBetasToPlot, cmap1=cmap1, cmap2=cmap2 )

    
    currTitle = fig.get_suptitle()
    alphaRefStr = 'rA' + str(rAlph) + '_' + 'eA' + str(simOptions['encodingAlpha'])
        # 
    fig.suptitle( currTitle + '\n' + alphaRefStr )






#-----------------------------------------------------------------------------------------
#%%  GIF of stim lifecycles 
#-----------------------------------------------------------------------------------------

nStepsPast = 100

desiredSteps = simOptions[ 'nRetenSteps' ] + nStepsPast
trimAnimationLen = max(  0,   simOptions[ 'nRetenSteps_ext' ] - desiredSteps   )
trimAnimationLen = 0





# varType = 'nEncodSteps'
# varValList = [ 25, 50, 100, 200, 300 ]
# # varValList = [ 100, 200, 300 ]
# # varValList = [ 200, 300 ]
# simOptions[ 'retentionAlpha' ] = 0.02
# simOptions[ 'nRetenSteps' ] = 200



# varType = 'retentionAlpha'
# varValList = [  0.002,  0.004,  0.006,  0.008  ]
# varValList = [  0.002,  0.02,  0.2  ]
# # varValList = [  0.002,  0.008  ]

# varValList = [  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09  ]
# # varValList = [  0.01,  0.02,  0.03,  0.04  ]
# # simOptions[ 'nEncodSteps' ] = 25
# # simOptions[ 'nRetenSteps' ] = 50
# # varValList = [  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7  ] 



varType = 'encodingAlpha'
varValList =  [  0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0  ]






# for rAlph in rententionAlphaVals:
for varVal in varValList:

    simOptions[ varType ] = varVal
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
    
    saveDirList = [ saveDir, 'stimLifecyclePCA' ]
    # saveDirList = [ saveDir ]
    # saveDirList = [ 'varsAndFigs/initState_8067' ] 
    # saveDirList = [ 'stimLifecyclePCA' ]

    

    # trimAnimationLen = max(  0,   simOptions[ 'nRetenSteps_ext' ] - desiredSteps   )
    trimAnimationLen = 0
    
    
    # print( testingData[ epochNum ]['state'][:,0]  )
    # continue
    
    
    ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, 
                                                    trimRetentionLen=trimAnimationLen, 
                                                    saveDir=saveDirList,
                                                    # saveDir='varsAndFigs',
                                                    )
    
    # figList = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum, 
    #                                              sameFig=False, timePtTypes=['endOfEncoding'] )

    



#-----------------------------------------------------------------------------------------
#%%  RP plots 
#-----------------------------------------------------------------------------------------



for varVal in varValList:

    simOptions[ varType ] = varVal
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
    
    
    FH = 12
    
    analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                     compareToRefState=True,
                                                     )
    
    rpFig = plotRP( analysisDict, nTestBetas, FH, colors=colorDict  )
    rpFig_reten = plotRP( analysisDict, nTestBetas, FH, colors=colorDict, endOfEncoding=False  )


    alphaRefStr = makeAlphaRefStr(  simOptions['encodingAlpha'],  simOptions['retentionAlpha']  )
    rpFig_reten.suptitle( rpFig_reten.get_suptitle() + '\n' + alphaRefStr )
    rpFig.suptitle( rpFig.get_suptitle() + '\n' + alphaRefStr )
















# #-----------------------------------------------------------------------------------------
# ''' Frozen phase time points (PCA) '''
# #-----------------------------------------------------------------------------------------

# for rAlph in rententionAlphaVals:

#     simOptions[ 'retentionAlpha' ] = rAlph
#     runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')


#     fig = plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum=epochNum )


   


#%%
#-----------------------------------------------------------------------------------------
''' end of encoding - PCA on entire state or converged pts '''
#-----------------------------------------------------------------------------------------

for rAlph in rententionAlphaVals:

    simOptions[ 'retentionAlpha' ] = rAlph
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
    fig = plotStateVsConvergedPCA( testingData,  testingInput, simOptions )

 



#%%
#-----------------------------------------------------------------------------------------
''' PCA on state, then plot converged states --- color-coordinated by degree '''
#-----------------------------------------------------------------------------------------



for rAlph in rententionAlphaVals:

    simOptions[ 'retentionAlpha' ] = rAlph
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
        

    
    fig, projPlot = plotConvergedStatesByDegree( epochNum, testingData, testingInput, PCAtype='state' )
    
    currTitle = fig.get_suptitle()
    alphaRefStr = 'rA' + str(rAlph) + '_' + 'eA' + str(simOptions['encodingAlpha'])
    fig.suptitle( currTitle + '\n' + alphaRefStr )




#%% 

''' Compare PCs to the rows of D, H '''


D = epochModels[ nEpochs ].D
H = epochModels[ nEpochs ].H


A = testingData[ nEpochs ][ 'state' ].T
[ U, S, V ] = torch.pca_lowrank( A )  

k = 3      
PCs = V[:, :k] 
projection = torch.matmul(  A,  PCs  )  


DHV = torch.cat(   [ torch.tensor(D), torch.tensor(H), torch.tensor(PCs.T) ]   )            ## ( d + N + k )
corrs = torch.corrcoef( DHV )
# corrs = torch.corrcoef( DHV.T )


plt.imshow( corrs, vmin=-1, vmax=1 )
plt.colorbar( )




#%%
#-----------------------------------------------------------------------------------------
''' Raw converged states, sorted by degree '''
#-----------------------------------------------------------------------------------------

for rAlph in rententionAlphaVals:

    simOptions[ 'retentionAlpha' ] = rAlph
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    

    model = epochModels[ epochNum ]
    fig = plotConvergedRetentionByDegree( epochNum, model, testingInput, testingData, plotConvergence=True )
    
    
    currTitle = fig.get_suptitle()
    alphaRefStr = 'rA' + str(rAlph) + '_' + 'eA' + str(simOptions['encodingAlpha'])
    
    fig.suptitle( fig.get_suptitle() + '\n' + alphaRefStr )




#% %
#-----------------------------------------------------------------------------------------
''' Raw activity (partial) '''
#-----------------------------------------------------------------------------------------

for rAlph in rententionAlphaVals:

    simOptions[ 'retentionAlpha' ] = rAlph
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
    
    runcell('plot activity', '/Users/bethannajones/Desktop/PGM/retentionAnalysis.py')

    


    
    
    
#%%   

#-----------------------------------------------------------------------------------------
''' recovery performance '''
#-----------------------------------------------------------------------------------------
    
FH = 12
    
    

for rAlph in rententionAlphaVals:

    simOptions[ 'retentionAlpha' ] = rAlph
    runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
    alphaRefStr = makeAlphaRefStr(  simOptions[ 'encodingAlpha' ],  simOptions[ 'retentionAlpha' ]  )
    
    
    
    analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                     compareToRefState=False )
    # rpFig = plotRP( analysisDict['recovPerfDict'], testingInput.nBetas, FH, colors=colorDict )
    rpFig = plotRP( analysisDict, testingInput.nBetas, FH, colors=colorDict )
    rpFig.suptitle(  rpFig.get_suptitle() + '\n' + alphaRefStr  )
    
    # newYLabel = rpFig.get_axes()[0].get_ylabel(  ).replace( '\\mathbf{u}(t_i-q)', '\\mathbf{x}(t_i-q)' )
    # newYLabel = newYLabel.replace( '\\mathbf{H^q}\\mathbf{r}(t_i)', '\\mathbf{DH^q}\\mathbf{r}(t_i)' )
    
    # rpFig.get_axes()[0].set_ylabel( newYLabel )
    
    
    
    
    analysisDict_refState = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                     compareToRefState=True )
    # rpFig_refState = plotRP( analysisDict_refState['recovPerfDict'], testingInput.nBetas, FH, colors=colorDict )
    rpFig_refState = plotRP( analysisDict_refState, testingInput.nBetas, FH, colors=colorDict )
    rpFig_refState.suptitle(  rpFig.get_suptitle() + '\n' + alphaRefStr  )
    
    
    # newYLabel = rpFig_refState.get_axes()[0].get_ylabel(  ).replace( '\\mathbf{r}(t_i-q)', '\\mathbf{r}(t_i-q)' )
    # rpFig_refState.get_axes()[0].set_ylabel( newYLabel )
    
        





    
    
    
    
    
    
    
    
    
#%%   
    


#------------------------------
''' eAlp0.9rAlp0.001, eSteps25rSteps500 '''
#------------------------------
simOptions[ 'encodingAlpha' ] = 0.9
simOptions[ 'retentionAlpha' ] = 0.001

simOptions['parameters']['nEpochs'] = nEpochs = 2000


simOptions['nEncodSteps'] = 25
simOptions['nRetenSteps'] = 500

runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

simOptions['nEncodSteps'] = 25
simOptions['nRetenSteps'] = 500
simOptions['nRetenSteps_ext'] = simOptions['nRetenSteps']

# plt.plot( testingInput.stimMat.T )

fig = plotStateVsConvergedPCA( testingData,  testingInput, simOptions )
# ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, trimRetentionLen=trimAnimationLen )

print( testingData[nEpochs]['state'][:,0] )



#% %
FH = 12
analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                 compareToRefState=False )
# rpFig = plotRP( analysisDict['recovPerfDict'], testingInput.nBetas, FH, colors=colorDict )
rpFig = plotRP( analysisDict, testingInput.nBetas, FH, colors=colorDict )
rpFig.suptitle(  rpFig.get_suptitle() + '\n' + alphaRefStr  )




analysisDict_refState = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                 compareToRefState=True )
# rpFig_refState = plotRP( analysisDict_refState['recovPerfDict'], testingInput.nBetas, FH, colors=colorDict )
rpFig_refState = plotRP( analysisDict_refState, testingInput.nBetas, FH, colors=colorDict )
rpFig_refState.suptitle(  rpFig.get_suptitle() + '\n' + alphaRefStr  )



#%%
#------------------------------
''' eAlp0.9rAlp0.07, eSteps25rSteps500  '''
#------------------------------
simOptions[ 'encodingAlpha' ] = 0.9
simOptions[ 'retentionAlpha' ] = 0.07

simOptions['parameters']['nEpochs'] = nEpochs = 2000


simOptions['nEncodSteps'] = 25
simOptions['nRetenSteps'] = 500

runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

simOptions['nEncodSteps'] = simOptions['nEncodSteps_testing']= 25
simOptions['nRetenSteps'] = 500
simOptions['nRetenSteps_ext'] = simOptions['nRetenSteps']


fig = plotStateVsConvergedPCA( testingData,  testingInput, simOptions )

# ani, aniFilename = animatedStimLifecyclePCA( simOptions, testingData, testingInput, trimRetentionLen=trimAnimationLen )
print( testingData[nEpochs]['state'][:,0] )



# simOptions_backup = simOptions.copy()

# testingData_backup = testingData.copy()
# testingInput_backup = testingInput.copy()

# trainingData_backup = trainingData.copy()
# trainingInputbackup = trainingInput.copy()

# epochModels_backup = epochModels.copy()





#% %

alphaRefStr = makeAlphaRefStr( simOptions['encodingAlpha'], simOptions['retentionAlpha'] )

FH = 12
analysisDict = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                 compareToRefState=False )
# rpFig = plotRP( analysisDict['recovPerfDict'], testingInput.nBetas, FH, colors=colorDict )
rpFig = plotRP( analysisDict, testingInput.nBetas, FH, colors=colorDict )
rpFig.suptitle(  rpFig.get_suptitle() + '\n' + alphaRefStr  )




analysisDict_refState = analyzeTestingPerf( epochModels, testingData, testingInput, simOptions, FH=FH,
                                                 compareToRefState=True )
# rpFig_refState = plotRP( analysisDict_refState['recovPerfDict'], testingInput.nBetas, FH, colors=colorDict )
rpFig_refState = plotRP( analysisDict_refState, testingInput.nBetas, FH, colors=colorDict )
rpFig_refState.suptitle(  rpFig.get_suptitle() + '\n' + alphaRefStr  )

# newYLabel = rpFig.get_axes()[0].get_ylabel(  ) 



# plt.plot( testingInput.stimMat.T )


#%%





rpFig = plotRP( analysisDict, nTestBetas, FH, colors=colorDict  )
rpFig_reten = plotRP( analysisDict, nTestBetas, FH, colors=colorDict, endOfEncoding=False  )




#%%
#------------------------------
'''  '''
#------------------------------
simOptions[ 'encodingAlpha' ] = 0.9
simOptions[ 'retentionAlpha' ] = 0.01

simOptions['parameters']['nEpochs'] = nEpochs = 2000


simOptions['nEncodSteps'] = simOptions['nEncodSteps_testing'] = 100
simOptions['nRetenSteps'] = 200



runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')



#%%


#------------------------------
'''  '''
#------------------------------
simOptions[ 'encodingAlpha' ] = 0.9
simOptions[ 'retentionAlpha' ] = 0.06

simOptions['parameters']['nEpochs'] = nEpochs = 2000

# simOptions['parameters']['maxIter'] = 10
simOptions['maxIter'] = 10



simOptions['nEncodSteps'] = simOptions['nEncodSteps_testing'] = 25
simOptions['nRetenSteps'] = 500

simOptions['weightCombos'] = weightCombos


runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    

#%%

#------------------------------
'''  '''
#------------------------------
simOptions[ 'encodingAlpha' ] = 0.9
simOptions[ 'retentionAlpha' ] = 0.2

simOptions['parameters']['nEpochs'] = nEpochs = 2000

# simOptions['parameters']['maxIter'] = 10
simOptions['maxIter'] = 10



simOptions['nEncodSteps'] = simOptions['nEncodSteps_testing'] = 25
simOptions['nRetenSteps'] = 200
simOptions['nRetenSteps_testing'] = 300

simOptions['weightCombos'] = weightCombos


runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')

    
    
    
    
    
    
    
    
    
    
    

    
#-----------------------------------------------------------------------------------------
#%% Extract the phases + PCA them 
#-----------------------------------------------------------------------------------------


nBetasToPlot = 3


for k in range( nBetasToPlot ):
    
    
    stimTimeInd = [ stimTimeInds[k] ]
    [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInd )
    
    
    
    # encodingPhases = state[ :, encodingTimeInds ]
    # retentionPhases = state[ :, retentionTimeInds ]
    
    
    retentionProj = stateProj[ retentionTimeInds ]                      # (  (nRetenSteps * nTestBetas),  k  )
    encodingProj = stateProj[ encodingTimeInds ]                      # (  (nEncodSteps * nTestBetas),  k  )
    
    
    fig = plt.figure( ) 
    ax = fig.add_subplot( projection='3d' )

    
    
    
    # retenFig = ax.scatter( retentionProj[:,0], retentionProj[:,1], retentionProj[:,2], s=5, c=colors[binNum], marker='o' )
    retenFig = ax.scatter( retentionProj[:,0], retentionProj[:,1], retentionProj[:,2], s=5, marker='o' )
    
    # # encodFig = ax.scatter( encodingProj[:,0], encodingProj[:,1], encodingProj[:,2], s=5, c=colors[binNum] )
    # encodFig = ax.scatter( encodingProj[:,0], encodingProj[:,1], encodingProj[:,2], s=5  )








#-----------------------------------------------------------------------------------------
#%%
#-----------------------------------------------------------------------------------------









#-----------------------------------------------------------------------------------------
#%%
#-----------------------------------------------------------------------------------------










#-----------------------------------------------------------------------------------------
#%% 
#-----------------------------------------------------------------------------------------


    for epochNum in [ 0, nEpochs ]:
        print( )
        print( epochNum )
        

        #--------------------------------------------------------
        '''  '''
        #--------------------------------------------------------
        for k in range( nBetasToPlot ):
            print( k )
            
            fig = plt.figure( ) 
        
            [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, [stimTimeInds[k]] )
            
            
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

#%%





#%% plot activity



# def plotRetentionActivity( epochNum, testingData, testingInput, nBetasToPlot=5 )

    

nBetasToPlot = nTestBetas
nBetasToPlot = int( nTestBetas/4 )
nBetasToPlot = 5

#--------------------------------------

state = testingData[ epochNum ][ 'state' ].detach().numpy()


# nEncodSteps = nEncodSteps_testing = testingData['nEncodSteps']
# nRetenSteps = testingData['nRetenSteps']
nEncodSteps = nEncodSteps_testing = simOptions['nEncodSteps_testing']
nRetenSteps = simOptions['nRetenSteps_ext']

stimTimeInds = testingData[ 'stimTimeInds' ]
[ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds[0:nBetasToPlot] )


retentionStates = state[ :, retentionTimeInds ]
encodingStates = state[ :, encodingTimeInds ]





#-----------------------------------------------------------------------------------------

fig, axs = plt.subplots( 2,1 )


#--------------------------------------

axs[0].plot( retentionStates.T )
axs[0].set_ylabel( r'$\mathbf{r}(t)$' ) 

#--------------------------------------
    
D = epochModels[ epochNum ].D.detach().numpy()
decodedRetentionPhase = D @ retentionStates

axs[1].plot( decodedRetentionPhase.T )
axs[1].set_ylim( [-1.25, 1.25] )

axs[1].set_ylabel( r'$\mathbf{Dr}(t)$' ) 

#--------------------------------------


d = testingInput.signalDim


for k in range( nBetasToPlot ):
    axs[0].axvline( k*nRetenSteps, c='gray', linewidth=1, linestyle='--' )
    axs[1].axvline( k*nRetenSteps, c='gray', linewidth=1, linestyle='--' )
    
    axs[1].scatter( [k*nRetenSteps + 1]*d, testingInput.stimMat[:,k], c='red', marker='*', s=4 )


plt.suptitle( 'Retention phase activity for betas 0-{}'.format(nBetasToPlot) )
plt.xlabel( 'Timestep' )







fig, axs = plt.subplots( 2,1 )


#--------------------------------------

axs[0].plot( encodingStates.T )
axs[0].set_ylabel( r'$\mathbf{r}(t)$' ) 

#--------------------------------------
    
D = epochModels[ epochNum ].D.detach().numpy()
decodedEncodingPhase = D @ encodingStates

axs[1].plot( decodedEncodingPhase.T )
axs[1].set_ylim( [-1.25, 1.25] )

axs[1].set_ylabel( r'$\mathbf{Dr}(t)$' ) 

#--------------------------------------


for k in range( nBetasToPlot ):
    axs[0].axvline( k*nEncodSteps, c='gray', linewidth=1, linestyle='--' )
    axs[1].axvline( k*nEncodSteps, c='gray', linewidth=1, linestyle='--' )
    
    axs[1].scatter( [k*nEncodSteps + 1]*d, testingInput.stimMat[:,k], c='red', marker='*', s=4 )


plt.suptitle( 'Encoding phase activity for betas 0-{}'.format(nBetasToPlot) )
plt.xlabel( 'Timestep' )








currTitle = fig.get_suptitle()
alphaRefStr = 'rA' + str(simOptions['retentionAlpha']) + '_' + 'eA' + str(simOptions['encodingAlpha'])

fig.suptitle( fig.get_suptitle() + '\n' + alphaRefStr )
    
    
    
    # return fig







#%%














#%%















#%%

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




nBetasToPlot = nTestBetas


nBins = 6
nBins = 12
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
    [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds_curr )
    
    
    
    
    
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
    
    
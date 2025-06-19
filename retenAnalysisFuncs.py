#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:06:41 2024

@author: bethannajones
"""


#-----------------------------------------------------------------------------------------
#%% IMPORT
#-----------------------------------------------------------------------------------------


import numpy as np
import torch

import os

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial




#----------------------------------------
''' Encoding/retention alphas '''
#----------------------------------------

def makeAlphaRefStr( encodingAlpha, rententionAlpha ):
    
    alphaRefStr = 'eA' + str(encodingAlpha) + '_' + 'rA' + str(rententionAlpha)
    
    return alphaRefStr 





def getLongFilename( simOptions, baseFilename, fileType='' ):
    

    # aniFilename = 'stimLifecyclePCA_'  +  alphaRefStr.replace( '_', '-' )
    # longFilename = 'maxIter' + str( simOptions['parameters']['maxIter'] ) 
    
    alphaRefStr = makeAlphaRefStr( simOptions['encodingAlpha'], simOptions['retentionAlpha'] )
    
    
    longFilename = baseFilename
    
    longFilename = longFilename + 'maxIter' + str( simOptions['maxIter'] ) 
    longFilename = longFilename + '_' + alphaRefStr.replace( '_', '-' )
    longFilename = longFilename + '_nEn' + str( simOptions['nEncodSteps'] ) + 'nRe' + str( simOptions['nRetenSteps'] ) 
    longFilename = longFilename + '_WC' + simOptions['weightCombos'][0].replace( '_', '-' )
    longFilename = longFilename + '_nEpochs' + str( simOptions['parameters']['nEpochs'] ) 
    
    longFilename = longFilename + fileType   
    
    
    return longFilename



#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% All phases, color-coordinated
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


def plotColorCoorPhases( epochNum, testingData, testingInput, nBetasToPlot=20, 
                                        cmap1=mpl.colormaps['winter_r'], cmap2=mpl.colormaps['autumn_r'] ):

    nRetenSteps = testingData[ 'nRetenSteps' ]
    # nRetenSteps = testingData[ 'nRetenSteps_ext' ]
    nEncodSteps = testingData[ 'nEncodSteps' ]
    stimTimeInds = testingData[ 'stimTimeInds' ]

    # [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds[0:nBetasToPlot] )
    [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds[1:nBetasToPlot+1] )
    
    state = testingData[ epochNum ][ 'state' ]
    
    # state = torch.nn.functional.normalize( state.T ).T                  #################################3
    stateProj = PCA( state.T ).detach().numpy()
    
    
    
    stateProj_reten = stateProj[ retentionTimeInds ]                        ##  ( nTotalCycleSteps,  k )
    stateProj_encod = stateProj[ encodingTimeInds ]                         ##  ( nTotalCycleSteps,  k )
    
    
    colors1 = colors = cmap(   [ x/nEncodSteps for x in range(nEncodSteps) ]   )
    colors2 = cmap2(   [ x/nRetenSteps for x in range(nRetenSteps) ]   )

    # colors3 = np.concatenate(  [ colors1, colors2 ]  )
    
    
    
    
    
    fig = plt.figure( figsize=(6, 6) )
    ax = fig.add_subplot( projection='3d' )
    
    
    
    # for k in range( nBetasToPlot ):
    for k in range( 1, nBetasToPlot+1 ):
        
        # fig, axs = plt.subplots(  2, 3,  subplot_kw={"projection": "3d"},  figsize=(10, 5)  )
        
        # start = k * nRetenSteps
        start = (k-1) * nRetenSteps
        retenTimeInds = list(  range(start, start+nRetenSteps)  )
        
        # start = k * nEncodSteps
        start = (k-1) * nEncodSteps
        encodTimeInds = list(  range(start, start+nEncodSteps)  ) 
        
        
            
        encod = ax.scatter(  stateProj_encod[encodTimeInds,0],  stateProj_encod[encodTimeInds,1],  stateProj_encod[encodTimeInds,2],  c=colors1, s=5  )
        reten = ax.scatter(  stateProj_reten[retenTimeInds,0],  stateProj_reten[retenTimeInds,1],  stateProj_reten[retenTimeInds,2],  c=colors2, s=5  )
        # ax.set_title( 'PCA on state first' )
    
    
        ax.set_xlabel( 'Component 1' )
        ax.set_ylabel( 'Component 2' )
        ax.set_zlabel( 'Component 3' )
    
    
    
    
    
    
    encod.cmap = cmap1
    # cbarLabel1 = 'Encoding phase t'
    cbarLabel1 = 'Individual encoding phase t'
    cbar1 = fig.colorbar( encod, location='left', label=cbarLabel1 )
    
    
    
    reten.cmap = cmap2
    
    fig.subplots_adjust( right=0.9 )    
    lastAxPos = fig.get_axes()[-1].get_position()
    cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, lastAxPos.height]  )                                    ## create space on the right hand side
    
    # cbarLabel2 = 'Retention phase t'
    cbarLabel2 = 'Individual retention phase t'
    fig.colorbar( reten, cax=cbarAx, label=cbarLabel2 )
    
    
    titleStr = 'Stim lifecycles: betas 0-{}'.format(nBetasToPlot) 
    
    plt.suptitle( titleStr )    



    return fig 




#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% Degree bins 
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------



def plotRetentionPCAsByDegreeBins( epochNum, modelData, modelInput, nBins=4, cmap=mpl.colormaps['viridis'], phasesPCA=False ):


    
    state = modelData[ epochNum ][ 'state' ]



    #-------------------------------------------------------------------------------------
    ''' Make sure we can go ahead '''
    #-------------------------------------------------------------------------------------
    possibleBinNums = [ 4, 8, 12 ]
    
    if nBins not in possibleBinNums:
        raise Exception(  '[plotRetentionTrajsByDegreeBins] This function is currently only able to handle nBins={0}.'.format( possibleBinNums )  ) 
    
    


    #-------------------------------------------------------------------------------------
    ''' Bin the circular stimuli by their degree '''
    #-------------------------------------------------------------------------------------
    binnedBetaInds = binBetasByTheta( modelInput, nBins=nBins )
    binKeys = list( binnedBetaInds.keys() )
    
    
    #-------------------------------------------------------------------------------------
    ''' Set up the figure '''
    #-------------------------------------------------------------------------------------
    colors = cmap(   [ x/nRetenSteps for x in range(nRetenSteps) ]   )
    
    #------------------------------------
    
    if nBins == 4:
        
        [fig, axs] = plt.subplots( 2, 2, figsize=(6, 6), subplot_kw={"projection": "3d"} )

        axIndDict = { 0  :  [ 0, 1 ],       #  0 - 90
                      1  :  [ 0, 0 ],       #  91 - 180
                      2  :  [ 1, 0 ],       #  181 - 270
                      3  :  [ 1, 1 ],       #  271 - 360
                      }
        
    #------------------------------------
    
    elif nBins == 8:
        
        [fig, axs] = plt.subplots( 4, 2, figsize=(6, 6), subplot_kw={"projection": "3d"} )

        axIndDict = { 0  :  [ 0, 0 ],       #  0 - 45           ## Q1
                      1  :  [ 0, 1 ],       #  46 - 90
                      
                      2  :  [ 1, 0 ],       #  91 - 135         ## Q2
                      3  :  [ 1, 1 ],       #  136 - 180
                      
                      4  :  [ 2, 0 ],       #  181 - 225        ## Q3
                      5  :  [ 2, 1 ],       #  226 - 270
                      
                      6  :  [ 3, 0 ],       #  271 - 315        ## Q4
                      7  :  [ 3, 1 ],       #  316 - 360
                      }
        
    #------------------------------------
    
    elif nBins == 12:
        
        [fig, axs] = plt.subplots( 4, 3, figsize=(6, 6), subplot_kw={"projection": "3d"} )

        axIndDict = { 0  :  [ 0, 0 ],       #  0 - 30           ## Q1
                      1  :  [ 0, 1 ],       #  31 - 60
                      2  :  [ 0, 2 ],       #  61 - 90
                      
                      3  :  [ 1, 0 ],       #  91 - 120         ## Q2
                      4  :  [ 1, 1 ],       #  121 - 150
                      5  :  [ 1, 2 ],       #  151 - 180
                      
                      6  :  [ 2, 0 ],       #  181 - 210        ## Q3
                      7  :  [ 2, 1 ],       #  211 - 240
                      8  :  [ 2, 2 ],       #  241 - 270
                      
                      9  :  [ 3, 0 ],       #  271 - 300        ## Q4
                      10  :  [ 3, 1 ],       #  301 - 330
                      11  :  [ 3, 2 ],       #  331 - 360
                      }
        
        #------------------------------------
        
        # elif nBins == 16:
            
        #     [fig, axs] = plt.subplots( 4, 4, figsize=(6, 6), subplot_kw={"projection": "3d"} )

        #     axIndDict = { 0  :  [ 0, 0 ],       #  0 - 45           ## Q1
        #                   1  :  [ 0, 1 ],       #  46 - 90
        #                   2  :  [ 1, 1 ],       #  136 - 180
        #                   3  :  [ 1, 1 ],       #  136 - 180
                          
        #                   2  :  [ 1, 0 ],       #  91 - 135         ## Q2
        #                   3  :  [ 1, 1 ],       #  136 - 180
        #                   2  :  [ 1, 1 ],       #  136 - 180
        #                   3  :  [ 1, 1 ],       #  136 - 180
                          
        #                   4  :  [ 2, 0 ],       #  181 - 225        ## Q3
        #                   5  :  [ 2, 1 ],       #  226 - 270
        #                   2  :  [ 1, 1 ],       #  136 - 180
        #                   3  :  [ 1, 1 ],       #  136 - 180
                          
        #                   6  :  [ 3, 0 ],       #  271 - 315        ## Q4
        #                   7  :  [ 3, 1 ],       #  316 - 360
        #                   2  :  [ 1, 1 ],       #  136 - 180
        #                   3  :  [ 1, 1 ],       #  136 - 180
        #                   }
        
    
    
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Plot the PCA for each bin '''
    #-------------------------------------------------------------------------------------
    if phasesPCA:
        [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds )
    
        retentionStates = state[ :, retentionTimeInds ]
        retentionProj = PCA( retentionStates.T ).detach().numpy()
    
    else: 
        stateProj = PCA( state.T ).detach().numpy()                         # ( nTimes, k )

    

    stimTimeInds = modelData[ 'stimTimeInds' ]


    
    for binNum in range( nBins ):
    
    
        ''' The current bin '''
        #--------------------------------------
        axInds = axIndDict[ binNum ]
        ax = axs[ axInds[0], axInds[1] ]
        
        ax.set_title( binKeys[binNum] )

        
        ''' Stimuli in current bin '''
        #--------------------------------------
        betaInds = binnedBetaInds[   binKeys[ binNum ]   ][ 0 ] 
        
        if len(betaInds) == 0:
            continue
    
        
        ''' PCA of current trajectories '''
        #--------------------------------------
        if not phasesPCA:
            stimTimeInds_curr = [  stimTimeInds[k] for k in betaInds  ]
            [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps_testing, nRetenSteps, stimTimeInds_curr )
            
            stateProj_reten = stateProj[ retentionTimeInds ]                      # (  (nRetenSteps * nTestBetas),  k  )
        
        
        # currRetenStates = state[ :, retentionTimeInds ]
        # currRetenProj = PCA( currRetenStates.T ).detach().numpy()
        
        
        
        
        ''' Plot the indiv. trajectories '''
        #--------------------------------------
        for k in range( len(betaInds) ):
            start = k * nRetenSteps
            
            if phasesPCA:
                currToPlot = retentionProj[ start : start+nRetenSteps ]
            else: 
                currToPlot = stateProj_reten[ start : start+nRetenSteps ]
            # currToPlot = currRetenProj
        
        
            retenFig = ax.scatter( currToPlot[:,0], currToPlot[:,1], currToPlot[:,2], s=5, c=colors, marker='o' )



        # retenFig = ax.scatter( stateProj_reten[:,0], stateProj_reten[:,1], stateProj_reten[:,2], s=5, c=colors, marker='o' )
    
    
            
            
            
    
    #-------------------------------------------------------------------------------------
    ''' Finish the figure '''
    #-------------------------------------------------------------------------------------        
    fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
    lastAxPos = fig.get_axes()[-1].get_position()
    # cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, lastAxPos.height]  )
    cbarAx = plt.axes(  [0.97, 0, 0.02, 1]  )
    
    retenFig.cmap = cmap
    
    fig.colorbar( retenFig, cax=cbarAx, label='Individual retention phase t' )
    
    # plt.colorbar( fig, label='Retention t', pad=0.15 )

        
    # ax.set_xlabel( 'Component 1' )
    # ax.set_ylabel( 'Component 2' )
    # ax.set_zlabel( 'Component 3' )
    plt.suptitle( 'Retention Phase PCA' )
    # 
    
    
    fig.tight_layout() 
    # fig.subplots_adjust(hspace=0.7)


        
        
        
        
    return fig 
        

        






def plotConvergedStatesByDegree( epochNum, modelData, modelInput, PCAtype='state',
                                        cmap=mpl.colormaps['viridis'], fig=None ):
    ''' Plot the converged retention states where the colors show degree value '''
    
    
    
    thetas = modelInput.thetas[0]
    degrees = thetas * ( 180 / math.pi ) 
    
    nPts = len( degrees )
        
    
    colors = cmap(   [ x/nPts for x in range(nPts) ]   )
    
    
    
    
    ''' PCA '''
    #-----------------------------
    if PCAtype == 'state':
        stimTimeInds = modelData['stimTimeInds']
        
        
        states = modelData[ epochNum ][ 'state' ]                           ## ( N, nBetas+1 )
        
        state = torch.nn.functional.normalize( states.T )                   ###################################
        
        stateProj = PCA( states.T )
    
        convergedTimeIndsReten = [  t-1 for t in stimTimeInds  ]
        stateProj_convergReten = stateProj[ convergedTimeIndsReten ] 
    
        proj = stateProj_convergReten.detach().numpy()
    
    
    else:
        convergedStates = modelData[ epochNum ][ 'convergedStates_retention' ][:,1::]        ## ( N, nBetas+1 )
        
        convergedStates = torch.nn.functional.normalize( convergedStates.T )                   ###################################

        proj = PCA( convergedStates.T ).detach().numpy()
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Plot '''
    #-------------------------------------------------------------------------------------
    if fig is None:
        fig, ax = plt.subplots( figsize=(6, 6), subplot_kw={'projection': '3d'}  )
        # ax = fig.add_subplot( projection='3d' ) 
    
    projPlot = ax.scatter( proj[:,0], proj[:,1], proj[:,2], c=colors, s=8 )    
    projPlot.cmap = cmap
    
    
    
    # if plotConvergence:
    #     ax = fig.add_subplot( 212, projection='3d' )     
    #     fig = plotAverageStateChangeForConvergence( epochNum, modelData, modelInput, fig=fig )
        

    
    
    if fig is not None:
        
        
        ''' Colorbar '''
        #-----------------------------
    
        nTicks = 7
        tickLocs = np.linspace(0,1,nTicks) 
        
        fig.subplots_adjust( right=0.8 )   
        lastAxPos = fig.get_axes()[-1].get_position()
        cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.04, lastAxPos.height]  ) 
        # cbarAx = 
        cbar = plt.colorbar( projPlot, label='Stimulus degree', ticks=tickLocs, cax=cbarAx )
        # cbar = plt.colorbar( projPlot, label='Degree', ticks=tickLocs )
        
        tickLabels = [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]
        cbar.ax.set_yticklabels( tickLabels )
        
    
    
    
        ''' Labels '''
        #-----------------------------
        ax.set_xlabel( 'Component 1' )
        ax.set_ylabel( 'Component 2' )
        ax.set_zlabel( 'Component 3' )
        
        if PCAtype == 'state':
            fig.suptitle( 'State PCA: converged retention states' )
        else:
            fig.suptitle( 'PCA of converged retention states' )
    
    
    
    
    return fig, projPlot
    
    
    
    
    
    




def plotConvergedStatesByDegree_multipleEpochs( epochNums, modelData, modelInput, cmapList, PCAtype='state' ):
    ''' Plot the converged retention states where the colors show degree value '''
    
    
    
    thetas = modelInput.thetas[0]
    degrees = thetas * ( 180 / math.pi ) 
    
    nPts = len( degrees )
        
    
        
    fig = plt.figure( figsize=(6, 6) )
    ax = fig.add_subplot( projection='3d' ) 
    
    
    for i in range( len(epochNums) ):
        epochNum = epochNums[ i ]
        cmap = cmapList[ i ]
    
    
        [fig, projPlot] = plotConvergedStatesByDegree( epochNum, testingData, testingInput, cmap=cmapList[i] ) 
    
    
    
        
        ''' Colorbar '''
        #-----------------------------
        nTicks = 7
        tickLocs = np.linspace(0,1,nTicks) 
        
        fig.subplots_adjust( right=0.8 )   
        lastAxPos = fig.get_axes()[-1].get_position()
        cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.04, lastAxPos.height]  ) 

        cbar = plt.colorbar( projPlot, location='left', label='Stimulus degree', ticks=tickLocs, cax=cbarAx )
        
        tickLabels = [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]
        cbar.ax.set_yticklabels( tickLabels )
        
    
    
    
        ''' Labels '''
        #-----------------------------
        ax.set_xlabel( 'Component 1' )
        ax.set_ylabel( 'Component 2' )
        ax.set_zlabel( 'Component 3' )
        
        if PCAtype == 'state':
            fig.suptitle( 'State PCA: converged retention states' )
        else:
            fig.suptitle( 'PCA of converged retention states' )
    
    


    ''' Overall colorbar '''
    #-----------------------------
    colors = [ cmap(1) for cmap in cmapList ]
    # colors = [ 'red', 'orange', 'gold', 'yellowgreen', 'green', 'royalblue', 'blue', 'purple', 'black' ]
    nColors = len( colors )
    
    colorDict = { }
    for i in range( nColors ):
        colorDict[ i ] = colors[ len(epochNums) - i - 1 ]
    
    colorDict = updateColorDictLabels( colorDict, epochNums )
    
    cbarMain = discreteColorbar( colorDict, fig, colorbarLabel='Epoch' )

        
    
    
    return fig
    
    
    
    
def plotConvergedRetentionByDegree( epochNum, model, modelInput, modelData, 
                                       plotConvergence=False, nStepsBack=5 ):
    
    
    thetas = modelInput.thetas[0]
    degrees = thetas * ( 180 / math.pi ) 
    
    nPts = len( degrees )
        
    
    sortingInds = np.argsort( degrees )
    
    
    convergedStates = modelData[ epochNum ][ 'convergedStates_retention' ][:,1::]        ## ( N, nBetas+1 )
    sortedStates = convergedStates[ :, sortingInds ]
    
    
    N = sortedStates.shape[0]
    
    cmap = mpl.colormaps[ 'hsv' ].resampled( N+2 ) 
    colors = cmap( [ x for x in range(N) ] )
    
    
    
    #-------------------------------------------------------------------------------------
    ''' Plot '''
    #-------------------------------------------------------------------------------------
    if plotConvergence:
        fig, axs = plt.subplots( 3, 1, figsize=(6, 6) )
    else:
        fig, axs = plt.subplots( 2, 1, figsize=(6, 6) )
    # fig = plt.figure( figsize=(6, 6) )
    
    
    
    # projPlot = axs[0].scatter( [degrees[sortingInds]]*N, sortedStates.detach().numpy().T, s=5, c=colors )    
    projPlot = axs[0].scatter( [degrees[sortingInds]]*N, sortedStates.detach().numpy().T, s=5 )    
    projPlot.cmap = cmap
    axs[0].set_ylabel( r'$\mathbf{r}(t^+)$' )

    
    
    decoded = model.D @ sortedStates
    d = decoded.shape[ 0 ]
    decodedPlot = axs[1].scatter( [degrees[sortingInds]]*d, decoded.detach().numpy().T, s=5 )    
    axs[1].set_ylim( [-1.25, 1.25] )

    
    axs[1].set_ylabel( r'$\mathbf{Dr}(t^+)$' )

    
    
    
    if plotConvergence:
        avgDiffNorms = averageStateChangeForConvergence( epochNum, modelData, modelInput, nStepsBack=5 )
    
        axs[2].scatter(  degrees[sortingInds],  avgDiffNorms[ sortingInds ], s=5  )
        axs[2].set_ylabel( 'Avg. end-phase change \n n={}'.format(nStepsBack) )
    
    
    
    axs[-1].set_xlabel( 'Stimulus degree' )    

    
    
    
    
    
    ''' Colorbar '''
    #-----------------------------
    # cbar = discreteColorbar( colors, fig, colorbarLabel='Subnetwork index' )
    
    
    
    
    
    
    # ''' Colorbar '''
    # #-----------------------------

    # nTicks = 7
    # tickLocs = np.linspace(0,1,nTicks) 
    
    # fig.subplots_adjust( right=0.8 )   
    # lastAxPos = fig.get_axes()[-1].get_position()
    # cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.04, lastAxPos.height]  ) 
    # # cbarAx = 
    # cbar = plt.colorbar( projPlot, label='Stimulus degree', ticks=tickLocs, cax=cbarAx )
    # # cbar = plt.colorbar( projPlot, label='Degree', ticks=tickLocs )
    
    # tickLabels = [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]
    # cbar.ax.set_yticklabels( tickLabels )
    



    # ''' Labels '''
    # #-----------------------------
    # ax.set_xlabel( 'Degree' )
    # ax.set_ylabel( r'$\mathbf{r}(t)$' )
    
    fig.suptitle( 'Converged retention states' )


    
    
    return fig 
    
    

        
      
        
       
def plotAverageStateChangeForConvergence( epochNum, modelData, modelInput, nStepsBack=5, fig=None ):
    
    
    avgDiffNorms = averageStateChangeForConvergence( epochNum, modelData, modelInput, nStepsBack )
    
    
    
    ''' Organize by degree '''
    #----------------------------------    
    thetas = modelInput.thetas[0]
    degrees = thetas * ( 180 / math.pi )         
    
    
    sortingInds = np.argsort( degrees )
    sortedDiffNorms = avgDiffNorms[ sortingInds ]
    
    
    
    
    ''' Plot'''
    #----------------------------------   
    if fig is None:
        fig = plt.figure(  )
        
    
    plt.scatter( degrees[sortingInds], sortedDiffNorms )
    
    plt.suptitle( 'End-phase convergence (Retention)' )
    plt.xlabel( 'Stimulus degree' )
    plt.ylabel( 'Norm of average state change' )
    
    
    
    return fig
        




       
def averageStateChangeForConvergence( epochNum, modelData, modelInput, nStepsBack=5 ):
    
    
    ''' Useful parameters '''
    #----------------------------------
    nBetas = modelInput.nBetas
    
    stimTimeInds = modelData[ 'stimTimeInds' ]
    nTotalCycleSteps = modelData[ 'nEncodSteps' ] + modelData[ 'nRetenSteps' ]
    
    # nextStimTimeInds = [  (stimTimeInds[k] + nTotalCycleSteps) for k in range(nBetas)  ]
    nextStimTimeInds = stimTimeInds[1::]  +  [ stimTimeInds[-1] + nTotalCycleSteps ]
    
    
    ''' Get the end phase states '''
    #----------------------------------
    state = modelData[ epochNum ][ 'state' ] .detach().numpy()
    
    endPhaseStates = np.zeros( [nBetas, N, nStepsBack] )
    
    for k in range( nBetas ):
        nextStimTimeInd = nextStimTimeInds[ k ]
        endPhaseTimeInds = list(   range( nextStimTimeInd-nStepsBack, nextStimTimeInd )   )
        currPhaseStates = state[ :, endPhaseTimeInds ]                                           ## ( N, nStepsBack )
        stateDiffs = [ currPhaseStates[:,i] - currPhaseStates[:,i-1] for i in range(1,nStepsBack)  ]      
        
        endPhaseStates[ k ] = currPhaseStates 
        
        
        
    
    ''' Average the state change '''
    #----------------------------------
    stateDiffs = [ endPhaseStates[:,i] - endPhaseStates[:,i-1] for i in range(1,nStepsBack)  ]      ## ( nStepsBack-1, nBetas, N )
    
    avgDiffs = np.mean(  np.array(stateDiffs),  axis=0  )           ## ( nBetas, N )
    avgDiffNorms = np.linalg.norm( avgDiffs, axis=1 )               ## ( nBetas, )
    
    

    
    # ''' Organize by degree '''
    # #----------------------------------    
    # thetas = modelInput.thetas[0]
    # degrees = thetas * ( 180 / math.pi )         
    
    
    # sortingInds = np.argsort( degrees )
    # sortedDiffNorms = avgDiffNorms[ sortingInds ]
    
    
    
    
    
    return avgDiffNorms
        










# plotRetentionTrajsByDegreeBins( state, testingInput )


    
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% Encoding of input + input history performance '''
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------



def encodingPerf( epochModels, testingData, testingInput, simOptions, epochNum=None ):
    ''' 
    Computes the error between 
            - Stimulus train and the decoded state          --->    stimPerf
            - Converged states and the reconstructed state  --->    refPerf
            - Stimulus train and the reconstructed state    --->    historyPerf
            
    '''
    

    if epochNum is None:
        epochNum = simOptions[ 'parameters' ][ 'nEpochs' ]
        
        
    state = testingData[ epochNum ][ 'state' ]                      ## ( N, nTimes )
    N, nTimes = state.shape
        
    model = epochModels[ epochNum ]
    D = model.D                                                     ## ( d, N )
    H = model.H                                                     ## ( N, N )
    
    decoded = D @ state                                             ## ( d, nTimes )
    reconstructed = H @ state                                       ## ( N, nTimes )
    refStates = testingData[ epochNum ][ 'convergedStates_retention' ]
        
    
    stimMat = testingInput.stimMat                                  ## ( d, nBetas )
    d, nBetas = stimMat.shape
    
    cycleLength = int(  (nTimes-1) /  nBetas  )         
    
    stimMat_fullCycle = torch.zeros( [ d, cycleLength*nBetas ] )    ## ( d, nTimes-1 )
    refStates_fullCycle = torch.zeros( [ N, cycleLength*nBetas ] )    ## ( d, nTimes-1 )
    
    
    for k in range( nBetas ):
        
        start = cycleLength * k
        end = start + cycleLength
        
        currStim = stimMat[ :, k ].reshape( d, 1 )
        currRef = refStates[ :, k+1 ].reshape( N, 1 )
        
        stimMat_fullCycle[ :, start:end ] = torch.tile(  currStim,  (1,cycleLength)  )
        refStates_fullCycle[ :, start:end ] = torch.tile(  currRef,  (1,cycleLength)  )
     
        
     
        
    # stimDiff = normError( decoded[:,1::], stimMat_fullCycle, normAxis=1 )
    stimDiff = decoded[:,1::] - stimMat_fullCycle                   ## ( d, nTimes-1 )
    stimDiffNorm = torch.linalg.norm(  stimDiff,  dim=0  )              ## ( nTimes-1, )
    stimPerf = stimDiffNorm.detach().numpy()
    
    denom = torch.norm(  stimMat_fullCycle, p=2, dim=0  )
    stimPerf = stimPerf / denom 

    

    # refDiff = reconstructed[:,(cycleLength+1)::] - refStates_fullCycle[:,0:-cycleLength]                   ## ( N,  )
    refDiff = reconstructed[:,1::] - refStates_fullCycle                   ## ( N,  )
    refDiffNorm = torch.linalg.norm(  refDiff,  dim=0  )                ## ( nTimes-1, )
    refPerf = refDiffNorm.detach().numpy()
    
    
    histDiff = (D @ reconstructed[:,1::]) - stimMat_fullCycle                    ## ( N, nTimes-1 )
    historyDiffNorm = torch.linalg.norm(  histDiff,  dim=0  )           ## ( nTimes-1, )
    historyPerf = historyDiffNorm.detach().numpy()
    
    
    

    return stimPerf, refPerf, historyPerf 




def plotEncodingPerf( epochModels, testingData, testingInput, simOptions, epochNum=None, maxBetaInd=None ):
    
    
    
    nBetas = testingInput.nBetas
    if maxBetaInd is None:
        maxBetaInd = nBetas
    
    stimPerf, refPerf, historyPerf = encodingPerf( epochModels, testingData, testingInput, simOptions, epochNum)
    
    
    fig, axs = plt.subplots( 3, 1 )
    
    
    nTimePts = stimPerf.shape[0] 
    cycleLength = int(  nTimePts /  nBetas  )         

    stimPerf = stimPerf[  0 : (maxBetaInd*cycleLength) ]
    refPerf = refPerf[  0 : (maxBetaInd*cycleLength) ]
    historyPerf = historyPerf[  0 : (maxBetaInd*cycleLength) ]
    
    

    nPts = stimPerf.shape[0] 
    xPts = range( 0, nPts )
    
    
    axs[0].scatter( xPts, stimPerf, s=5 )
    axs[0].set_ylabel( r'$\|  \beta_k - \mathbf{Dr}(t)  \|$' )
    
    
    axs[1].scatter( xPts, refPerf, s=5 )
    axs[1].set_ylabel( r'$\|  \mathbf{r}(t-1) - \mathbf{Hr}(t)  \|$' )
    
    
    axs[2].scatter( xPts, historyPerf, s=5 )
    axs[2].set_ylabel( r'$\|  \beta_k - \mathbf{DHr}(t)  \|$' )
    
    
    axs[2].set_xlabel( 't' )
    
    

    for k in range( 1, maxBetaInd ):
        arrivalTime = cycleLength * k 
        axs[0].axvline( arrivalTime, c='gray', linestyle='--', linewidth=2 )
        axs[1].axvline( arrivalTime, c='gray', linestyle='--', linewidth=2 )
        axs[2].axvline( arrivalTime, c='gray', linestyle='--', linewidth=2 )

    
    fig.suptitle( 'Encoding performance' )
    
    
    return fig 





def plotConvergedEncodingAccur( epochModels, testingData, testingInput, simOptions, epochNum=None, maxBetaInd=None ):
    
    
    
    nBetas = testingInput.nBetas
    if maxBetaInd is None:
        maxBetaInd = nBetas
    
    stimPerf, refPerf, historyPerf = encodingPerf( epochModels, testingData, testingInput, simOptions, epochNum)
    
    
    convergedTimeInds = testingData['convergedTimeInds'][ 0:maxBetaInd ]
    convergedStimPerfs = stimPerf[ convergedTimeInds ] 
    
    
    xPts = range( 0, maxBetaInd )
    
    
    fig = plt.figure(  )

    plt.scatter( xPts, convergedStimPerfs, s=5 )
    plt.ylabel( r'$\|  \beta_k - \mathbf{Dr}(t)  \|$' )
    
    plt.xlabel( 'Stimulus index k' )
    
    
    
    fig.suptitle( 'Encoding Accuracy' )
    
    
    return fig 






def extendedDelayConvergence( saveDir, epochNum=None ):

    fig, axs = plt.subplots( 2, 1 ) 
    

    
    modelInfoDict = getModelInfo( saveDir )
    
    if epochNum is None:
        epochNum = modelInfoDict[ 'simOptions' ][ 'parameters' ][ 'nEpochs' ]
    
    
    convergedReten = modelInfoDict[ 'testingData' ][ epochNum ][ 'convergedStates_retention' ].detach().numpy()
    [ N, nStates ] = convergedReten.shape
    nBetas = nStates - 1
    
    
    xPts = [  list(range(nBetas))  ] * N
    
    axs[0].scatter( xPts, convergedReten[:,1::], s=8 )
    axs[0].set_ylabel( r'$\mathbf{r}(t_k + \epsilon + \delta)$' )
    
    
    
    # stateVariance = stateVarianceInEndDelayPhase( modelInfoDict[ 'testingData' ], epochNum=epochNum )
    
    # state = modelInfoDict[ 'testingData' ][ epochNum ][ 'state' ].detach().numpy()
    
    
    # axs[1].scatter( xPts, stateVariance[:,1::], s=8 )
    # axs[1].set_ylabel( r'$\mathbf{r}(t_k + \epsilon + \delta)$' )
    # axs[1].set_xlabel( r'Stimulus index $k$' )


    return fig





def stateVarianceInEndDelayPhase( testingData, epochNum=None ):
    
    state = testingData[ epochNum ][ 'state' ].detach().numpy()
    
    
    
    
    
    return 







#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% PCA on state then plot covnerged vs PCA on converged directly 
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------




def PCA( A, k=3, returnPCs=False ):
    ''' 
        ** 
        Assumes that if A is an m-by-n matrix, there are m samples of n variables each. 
        That is, the rows of A are the observations and the columns are the variables. 
        **    
            
    '''
    
    [ U, S, V ] = torch.pca_lowrank( A )         
    projection = torch.matmul(  A,  V[:, :k]  )         # project data to first k principal components 
    
    if returnPCs:
        return projection, V
    
    else:
        return projection




# rententionAlphaVals = [ 0.001 ] 

# for rAlph in rententionAlphaVals:

#     simOptions[ 'retentionAlpha' ] = rAlph
#     runcell('LOAD using simOptions', '/Users/bethannajones/Desktop/PGM/PGM_main.py')
    
#     fig = plotStateVsConvergedPCA( testingData,  testingInput, simOptions )



def plotStateVsConvergedPCA( testingData, testingInput, simOptions, epochNum=None, 
                                        # cmap = mpl.colormaps[ 'cool_r' ],
                                        cmap = mpl.colormaps[ 'viridis' ],
                                        ): 
    
    
    if epochNum is None:
        epochNum = simOptions['parameters']['nEpochs'] 
    
    
    state = testingData[ epochNum ][ 'state' ]
    
    nBetas = testingInput.nBetas
    
    
    phaseTimePtDict = phaseTimePts( simOptions, nBetas )
    convergedTimeInds = phaseTimePtDict[ 'endOfEncoding' ]
    # convergedTimeInds = phaseTimePtDict[ 'endOfRetention_trained' ]
    
    
        
    [ fig, axs ] = plt.subplots(  1, 2,  subplot_kw={"projection": "3d"},  figsize=(10, 5)  ) 
    colors = cmap(   [ x/nBetas for x in range(nBetas) ]   )
    
    
    
    for i in range(2):
        
        PCAstateFirst = [ True, False ][ i ]
    
        if PCAstateFirst:    
            stateProj = PCA( state.T ).detach().numpy()
            projToPlot = stateProj[ convergedTimeInds, : ]
            
        else:
            converged = state[ :, convergedTimeInds ]
            projToPlot = PCA( converged.T ).detach().numpy()
            
        
        ax = axs[i]
        
        scatterFig = ax.scatter3D( projToPlot[:,0], projToPlot[:,1], projToPlot[:,2],
                                                 s=10, c=colors )
        
        scatterFig.cmap = cmap
        
        
        
        ax.set_xlabel( 'Component 1' )
        ax.set_ylabel( 'Component 2' )
        ax.set_zlabel( 'Component 3' )
        

        if PCAstateFirst:
            ax.set_title( 'PCA on state first' )
        else:
            ax.set_title( 'PCA on converged directly' )
        
        
        
        
    fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
    axPos = ax.get_position()
    cbarAx = plt.axes(  [0.95, axPos.y0, 0.02, axPos.height]  )
    cbar = plt.colorbar( scatterFig, label='Stimulus degree', cax=cbarAx ) 
    # cbar.set_ticks(  )
    
    eAlph = simOptions[ 'encodingAlpha' ]
    rAlph = simOptions[ 'retentionAlpha' ]
    alphaRefStr = makeAlphaRefStr( eAlph, rAlph )
    fig.suptitle( 'End of encoding' + '\n' + alphaRefStr)
        
        
    return fig 




    
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% Phase time points 
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
    




# def phaseTimePts( stimTimeInds, nEncodSteps, nRetenSteps_training, nRetenSteps_testing ):
def phaseTimePts( simOptions, nBetas ):
    
    
    # stimTimeInds = [  (nEncodSteps_training*i)+1  for i in range(nTrainingBetas)  ]



    # nEncodSteps_testing = nEncodSteps = simOptions[ 'nEncodSteps_testing' ]
    nEncodSteps_testing = nEncodSteps = simOptions[ 'nEncodSteps' ]
    nRetenSteps_training = simOptions[ 'nRetenSteps' ]
    nRetenSteps = nRetenSteps_ext = simOptions[ 'nRetenSteps_ext' ]
    
    totalCycleLength = nRetenSteps_ext + nEncodSteps_testing
    
    
    stimTimeInds = list(  np.arange( 1, (totalCycleLength*nBetas + 1), totalCycleLength )  )
    stimTimeInds = np.array( stimTimeInds )
    
    
    phaseTimePtDict = { 'endOfEncoding' :  stimTimeInds + nEncodSteps,
                        'beginningOfDelay' :  stimTimeInds + nEncodSteps + 1,
                        'endOfRetention_trained' :  stimTimeInds + nRetenSteps_training,
                        'endOfRetention_extended' :  stimTimeInds + nRetenSteps_ext,
                       } 
    
    
    # timePts = np.zeros(  [ nBetas, 4 ]  )
    
    
    # timePts[ :, 0 ] = stimTimeInds + nEncodSteps                ## end of encoding phase
    # timePts[ :, 1 ] = stimTimeInds + nEncodSteps + 1            ## beginning of delay
    # timePts[ :, 2 ] = stimTimeInds + nRetenSteps_training       ## end of trained delay
    # timePts[ :, 3 ] = stimTimeInds + nRetenSteps_testing        ## end of extended delay 

    
    return phaseTimePtDict








def plotPCAForPhaseTimePts( simOptions, testingData, testingInput, epochNum=None, 
                                   # cmap=mpl.colormaps['viridis'], normalizeData=False ):
                                   cmap=mpl.colormaps['viridis'], sameFig=True,
                                   timePtTypes=None,
                                   PCsToUse=None, 
                                   ):
    '''  '''
    
    
    
    #-----------------------------------------
    ''' PCA the overall state space '''
    #------------------------------------------
    if epochNum is None:
        epochNum = simOptions[ 'parameters' ][ 'nEpochs' ]

    state = testingData[ epochNum ][ 'state' ]                  ## ( N, nTimes+1 )
    
    # if normalizeData:
    #     state = torch.nn.functional.normalize( state.T )        ## ( nTimes+1, N )
    #     state = state.T                                         ## For consistency 
    
    
    
    # stateProj = PCA( state.T ).detach().numpy()                   ## ( nTimes+1, k )
    
    
    if PCsToUse is None:
        stateProj = PCA( state.T ).detach().numpy()         ## ( nTimes, k )
    
    else: 
        if PCsToUse.shape[0] == state.shape[0]:
            stateProj = torch.matmul(  state.T,  PCsToUse[:, :3]  ).detach().numpy()         
        else:
            raise Exception( 'Shape of given principal components are not compatible' )
                  
    
    
    
    
    
    #------------------------------------------
    ''' The time pts + corres. states '''
    #------------------------------------------
    stimTimeInds = testingData[ 'stimTimeInds' ] 
    # phaseTimePtDict = phaseTimePts( simOptions, stimTimeInds )
    phaseTimePtDict = phaseTimePts( simOptions, testingInput.nBetas )
    possibleTimePtTypes = list( phaseTimePtDict.keys() )
    
    if timePtTypes is None:
        timePtTypes = possibleTimePtTypes
    else:
        for timePtType in timePtTypes:
            if timePtType not in possibleTimePtTypes:
                raise Exception( 'Did not understand given timePtTypes' )
        
    
    stateProjDict = {  }
    
    for key in timePtTypes:
        timePts = phaseTimePtDict[ key ]
        # stateProjDict[ key ] = stateProj[ timePts ].detach().numpy() 
        stateProjDict[ key ] = stateProj[ timePts ]
    
    
    nSubFigs = len( timePtTypes )   
    
    
    #------------------------------------------
    ''' We color pts by the stimulus degree '''
    #------------------------------------------
    thetas = testingInput.thetas[0]
    degrees = thetas * ( 180 / math.pi ) 
    
    nPts = len( degrees )
    colors = cmap(   [ x/nPts for x in range(nPts) ]   )
    
    sortingInds = np.argsort( degrees )

    
    
    #------------------------------------------
    ''' PLOT '''
    #------------------------------------------
    if sameFig:
        fig, axs = plt.subplots( 2, 2, subplot_kw={'projection':'3d'} )
        
        axIndsDict = {  0 : [ 0, 0 ],
                        1 : [ 0, 1 ], 
                        2 : [ 1, 0 ], 
                        3 : [ 1, 1 ]  }
    else:
        figList = [  ] 

        for i in range( nSubFigs ):
            fig, axs = plt.subplots( subplot_kw={'projection':'3d'} )
            figList.append( fig )
    
    
    
    # projToPlot = [  stateProjDict[key] for key in timePtTypes  ]
    # maxVals = np.max( np.max( projToPlot, axis=0 ), axis=0 )
    # minVals = np.min( np.min( projToPlot, axis=0 ), axis=0 )
    
    maxVals = np.max( stateProj, axis=0 )
    minVals = np.min( stateProj, axis=0 )
    
    ## Assumed to be 4
    for i in range( nSubFigs ):
        
        
        if sameFig:
            axInds = axIndsDict[ i ]
            ax = axs[ axInds[0], axInds[1] ]
        else:
            fig = figList[ i ]
            ax = fig.get_axes()[0]
        
        currStates = stateProjDict[ timePtTypes[i] ]
        sortedStates = currStates[ sortingInds ]
        
        
        
        scatterPlot = ax.scatter(  sortedStates[:,0],  sortedStates[:,1],  sortedStates[:,2],  s=5,  c=colors  )
        
        if sameFig:
            ax.set_title( timePtTypes[i] )
        
        ax.set_xlabel( 'PC 1' )
        ax.set_ylabel( 'PC 2' )
        ax.set_zlabel( 'PC 3' )
        
        
        ax.set_xlim3d(  [ minVals[0], maxVals[0] ]  )
        ax.set_ylim3d(  [ minVals[1], maxVals[1] ]  )
        ax.set_zlim3d(  [ minVals[2], maxVals[2] ]  )
        
        
        
            
            
        ''' Colorbar '''
        #------------------------------------------
        fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
        
        nTicks = 7
        tickLocs = np.linspace( 0, 1, nTicks ) 
        tickLabels = [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]
        
        scatterPlot.cmap = cmap
        
        # if sameFig:
        #     lastAxPos = fig.get_axes()[-1].get_position()
        #     cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, 0.8]  )
            
        #     cbar = fig.colorbar( scatterPlot, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )

        # else:
        #     cbar = fig.colorbar( scatterPlot, label='Stimulus degree', ticks=tickLocs )
        
        
        
        if sameFig:
            lastAxPos = fig.get_axes()[-1].get_position()
            cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, 0.8]  )
        else:
            lastAxPos = ax.get_position()
            cbarAx = fig.add_axes([0.9, lastAxPos.y0, 0.02, 0.8])            
            # cbarAx = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
            
        cbar = fig.colorbar( scatterPlot, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )

        
        # mappable = fig.get_axes()[0].collections[0]
        # mappable = scatterPlot
        # mappable.cmap = cmap
        # cbar = fig.colorbar( mappable, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )
        
        cbar.ax.set_yticklabels( tickLabels )
            
            
        
        if not sameFig:
            figList[ i ] = fig
            
            
    
    
    
    
    if sameFig:
        figList = [ fig ]
        
        
    for i in range( len(figList) ):
        
        fig = figList[ i ]
        
        if sameFig:
            titleStr = 'PCA: phase time points' 
        else:
            titleStr = 'PCA: ' + timePtTypes[i]
            
        
        alphaRefStr = 'eA' + str(simOptions['encodingAlpha']) + '_' + 'rA' + str(simOptions['retentionAlpha'])
        fig.suptitle( titleStr + '\n' + alphaRefStr ) 
    
        
        # ''' Colorbar '''
        # #------------------------------------------
        # fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
        # lastAxPos = fig.get_axes()[-1].get_position()
        # cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, 0.8]  )
        
        # # scatterPlot.cmap = cmap
    
        
        # nTicks = 7
        # tickLocs = np.linspace( 0, 1, nTicks ) 
        # tickLabels = [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]
        
        # mappable = fig.get_axes()[0].collections[0]
        # # mappable = scatterPlot
        # mappable.cmap = cmap

        # # cbar = fig.colorbar( scatterPlot, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )
        # cbar = fig.colorbar( mappable, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )
        # cbar.ax.set_yticklabels( tickLabels )
            

    
        # fig.subplots_adjust( hspace=0.6, wspace=0.3 )
        fig.subplots_adjust( hspace=0.6, wspace=0.3 )

        figList[i] = fig


    if sameFig:
        return fig 
    else: 
        return figList
    
    
    
    

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% Manifolds for different alphas
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------



# def convergedForDifferentAlpha( testingDataDict, testingInputDict, alphaType='retentionAlpha'
        
#         # simOptions, testingData, testingInput, epochNum=None, 
#         #                            # cmap=mpl.colormaps['viridis'], normalizeData=False ):
#         #                            cmap=mpl.colormaps['viridis'], sameFig=True,
#         #                            timePtTypes=None,
#         #                            PCsToUse=None, 
#         #                            ):
#     '''  '''
    
#     alphaVals = list(  testingDataDict.keys()  )
    
    
#     if len(alphaVals) > 6:
#         raise Exception( 'Currently cannot handle more than 7 alpha values ' )
    
#     cmapNames = [  'Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Grays'  ]
#     # cmapList = [  mpl.colormaps( name )  for name in cmapNames  ]  
#     cmapList = [  mpl.colormaps( name ).resampled(400)  for name in cmapNames  ]  
    
    
    
    
    
    
    
    
#     #-----------------------------------------
#     ''' PCA the overall state space '''
#     #------------------------------------------
#     if epochNum is None:
#         epochNum = simOptions[ 'parameters' ][ 'nEpochs' ]

#     state = testingData[ epochNum ][ 'state' ]                  ## ( N, nTimes+1 )
    
#     # if normalizeData:
#     #     state = torch.nn.functional.normalize( state.T )        ## ( nTimes+1, N )
#     #     state = state.T                                         ## For consistency 
    
    
    
#     # stateProj = PCA( state.T ).detach().numpy()                   ## ( nTimes+1, k )
    
    
#     if PCsToUse is None:
#         stateProj = PCA( state.T ).detach().numpy()         ## ( nTimes, k )
    
#     else: 
#         if PCsToUse.shape[0] == state.shape[0]:
#             stateProj = torch.matmul(  state.T,  PCsToUse[:, :3]  ).detach().numpy()         
#         else:
#             raise Exception( 'Shape of given principal components are not compatible' )
                  
    
    
    
    
    
#     #------------------------------------------
#     ''' The time pts + corres. states '''
#     #------------------------------------------
#     stimTimeInds = testingData[ 'stimTimeInds' ] 
#     # phaseTimePtDict = phaseTimePts( simOptions, stimTimeInds )
#     phaseTimePtDict = phaseTimePts( simOptions, testingInput.nBetas )
#     possibleTimePtTypes = list( phaseTimePtDict.keys() )
    
#     if timePtTypes is None:
#         timePtTypes = possibleTimePtTypes
#     else:
#         for timePtType in timePtTypes:
#             if timePtType not in possibleTimePtTypes:
#                 raise Exception( 'Did not understand given timePtTypes' )
        
    
#     stateProjDict = {  }
    
#     for key in timePtTypes:
#         timePts = phaseTimePtDict[ key ]
#         # stateProjDict[ key ] = stateProj[ timePts ].detach().numpy() 
#         stateProjDict[ key ] = stateProj[ timePts ]
    
    
#     nSubFigs = len( timePtTypes )   
    
    
#     #------------------------------------------
#     ''' We color pts by the stimulus degree '''
#     #------------------------------------------
#     thetas = testingInput.thetas[0]
#     degrees = thetas * ( 180 / math.pi ) 
    
#     nPts = len( degrees )
#     colors = cmap(   [ x/nPts for x in range(nPts) ]   )
    
#     sortingInds = np.argsort( degrees )

    
    
#     #------------------------------------------
#     ''' PLOT '''
#     #------------------------------------------
#     if sameFig:
#         fig, axs = plt.subplots( 2, 2, subplot_kw={'projection':'3d'} )
        
#         axIndsDict = {  0 : [ 0, 0 ],
#                         1 : [ 0, 1 ], 
#                         2 : [ 1, 0 ], 
#                         3 : [ 1, 1 ]  }
#     else:
#         figList = [  ] 

#         for i in range( nSubFigs ):
#             fig, axs = plt.subplots( subplot_kw={'projection':'3d'} )
#             figList.append( fig )
    
    
    
#     # projToPlot = [  stateProjDict[key] for key in timePtTypes  ]
#     # maxVals = np.max( np.max( projToPlot, axis=0 ), axis=0 )
#     # minVals = np.min( np.min( projToPlot, axis=0 ), axis=0 )
    
#     maxVals = np.max( stateProj, axis=0 )
#     minVals = np.min( stateProj, axis=0 )
    
#     ## Assumed to be 4
#     for i in range( nSubFigs ):
        
        
#         if sameFig:
#             axInds = axIndsDict[ i ]
#             ax = axs[ axInds[0], axInds[1] ]
#         else:
#             fig = figList[ i ]
#             ax = fig.get_axes()[0]
        
#         currStates = stateProjDict[ timePtTypes[i] ]
#         sortedStates = currStates[ sortingInds ]
        
        
        
#         scatterPlot = ax.scatter(  sortedStates[:,0],  sortedStates[:,1],  sortedStates[:,2],  s=5,  c=colors  )
        
#         if sameFig:
#             ax.set_title( timePtTypes[i] )
        
#         ax.set_xlabel( 'PC 1' )
#         ax.set_ylabel( 'PC 2' )
#         ax.set_zlabel( 'PC 3' )
        
        
#         ax.set_xlim3d(  [ minVals[0], maxVals[0] ]  )
#         ax.set_ylim3d(  [ minVals[1], maxVals[1] ]  )
#         ax.set_zlim3d(  [ minVals[2], maxVals[2] ]  )
        
        
        
            
            
#         ''' Colorbar '''
#         #------------------------------------------
#         fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
        
#         nTicks = 7
#         tickLocs = np.linspace( 0, 1, nTicks ) 
#         tickLabels = [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]
        
#         scatterPlot.cmap = cmap
        
#         # if sameFig:
#         #     lastAxPos = fig.get_axes()[-1].get_position()
#         #     cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, 0.8]  )
            
#         #     cbar = fig.colorbar( scatterPlot, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )

#         # else:
#         #     cbar = fig.colorbar( scatterPlot, label='Stimulus degree', ticks=tickLocs )
        
        
        
#         if sameFig:
#             lastAxPos = fig.get_axes()[-1].get_position()
#             cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, 0.8]  )
#         else:
#             lastAxPos = ax.get_position()
#             cbarAx = fig.add_axes([0.9, lastAxPos.y0, 0.02, 0.8])            
#             # cbarAx = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
            
#         cbar = fig.colorbar( scatterPlot, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )

        
#         # mappable = fig.get_axes()[0].collections[0]
#         # mappable = scatterPlot
#         # mappable.cmap = cmap
#         # cbar = fig.colorbar( mappable, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )
        
#         cbar.ax.set_yticklabels( tickLabels )
            
            
        
#         if not sameFig:
#             figList[ i ] = fig
            
            
    
    
    
    
#     if sameFig:
#         figList = [ fig ]
        
        
#     for i in range( len(figList) ):
        
#         fig = figList[ i ]
        
#         if sameFig:
#             titleStr = 'PCA: phase time points' 
#         else:
#             titleStr = 'PCA: ' + timePtTypes[i]
            
        
#         alphaRefStr = 'eA' + str(simOptions['encodingAlpha']) + '_' + 'rA' + str(simOptions['retentionAlpha'])
#         fig.suptitle( titleStr + '\n' + alphaRefStr ) 
    
        
#         # ''' Colorbar '''
#         # #------------------------------------------
#         # fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
#         # lastAxPos = fig.get_axes()[-1].get_position()
#         # cbarAx = plt.axes(  [0.97, lastAxPos.y0, 0.02, 0.8]  )
        
#         # # scatterPlot.cmap = cmap
    
        
#         # nTicks = 7
#         # tickLocs = np.linspace( 0, 1, nTicks ) 
#         # tickLabels = [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]
        
#         # mappable = fig.get_axes()[0].collections[0]
#         # # mappable = scatterPlot
#         # mappable.cmap = cmap

#         # # cbar = fig.colorbar( scatterPlot, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )
#         # cbar = fig.colorbar( mappable, cax=cbarAx, label='Stimulus degree', ticks=tickLocs )
#         # cbar.ax.set_yticklabels( tickLabels )
            

    
#         # fig.subplots_adjust( hspace=0.6, wspace=0.3 )
#         fig.subplots_adjust( hspace=0.6, wspace=0.3 )

#         figList[i] = fig


#     if sameFig:
#         return fig 
#     else: 
#         return figList
    
    
    
    
    
def timeToConverge( testingData, simOptions, nBetas, epochNum=None, testing=True, tol=1e-4 ):
    
    if epochNum is None:
        epochNum = simOptions[ 'parameters' ][ 'nEpochs' ]
    
    
    
    ''' Time inds '''
    #-----------------------------------------------------------------
    if testing:
        nEncodSteps = simOptions[ 'nEncodSteps_testing' ]
        nRetenSteps = simOptions[ 'nRetenSteps_ext' ]
    else: 
        nEncodSteps = simOptions[ 'nEncodSteps_training' ]
        nRetenSteps = simOptions[ 'nRetenSteps_training' ]
        
    nTotalCycleSteps = nEncodSteps + nRetenSteps
    
    stimTimeInds = [  (k*nTotalCycleSteps)+1 for k in range(nBetas)  ]
    # [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps, nRetenSteps, stimTimeInds )


    
    ''' The state values '''
    #-----------------------------------------------------------------
    state = testingData[ epochNum ][ 'state' ]
    
    
    encodConvergTimes = np.zeros( [nBetas, 1] )
    retenConvergeTimes = np.zeros( [nBetas, 1] )
    
    
    for k in range(nBetas):
    
        [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps, nRetenSteps, stimTimeInds[k] )

        encodingStates = state[ :, encodingTimeInds ]
        retentionStates = state[ :, retentionTimeInds ]
    
        diffNorms_encod = torch.linalg.norm(  encodingStates[:,1::] -  encodingStates[:,0:-1],  axis=0  )
        diffNorms_reten = torch.linalg.norm(  retentionStates[:,1::] -  retentionStates[:,0:-1],  axis=0  )
    
        # encodConvergTimes[ k ] = torch.where( diffNorms_encod < tol )[0][0]
        # retenConvergeTimes[ k ] = torch.where( diffNorms_reten < tol )[0][0]
        print( torch.where( diffNorms_encod < tol ) )
        print( torch.where( diffNorms_reten < tol ) )
        
        encodConvergTimes[ k ] = torch.where( diffNorms_encod < tol )[0]
        retenConvergeTimes[ k ] = torch.where( diffNorms_reten < tol )[0]
    
    
    # ''' The state values '''
    # #-----------------------------------------------------------------
    
    # diffNorms_encod = torch.linalg.norm(  encodingStates[:,1::] -  encodingStates[:,0:-1],  axis=0  )
    # diffNorms_reten = torch.linalg.norm(  retentionStates[:,1::] -  retentionStates[:,0:-1],  axis=0  )
    
  
    
    return encodConvergTimes, retenConvergeTimes



def plotAvgTimeToConvergePerVarVal( testingDataDict, simOptions, nBetas, phase='retention',
                                              epochNum=None, testing=True, tol=1e-4 ):
    
    
    varVals = list(  testingDataDict.keys()  )
    
    if phase == 'both':
        avgTimes = np.zeros( [nBetas,2] )
        xPts = varVals
        colors = [ 'red' ]

    else:
        avgTimes = np.zeros( [nBetas,1] )
        xPts = [ varVals ] * 2
        colors = [ 'red', 'blue' ]

    
    for i in range( len(varVals) ):
        
        alphaVal = varVals[ i ]
        
        testingData = testingDataDict[ alphaVal ]
    
        [ encodConvergTimes, retenConvergeTimes ] = timeToConverge( testingData, simOptions, nBetas, 
                                                                   epochNum=epochNum, testing=testing, tol=tol )
    
        if phase == 'retention':
            avgTimes[ i ] = torch.mean( encodConvergTimes ) 
        elif phase == 'encoding':
            avgTimes[ i ] = torch.mean( retenConvergeTimes ) 
        elif phase =='both':
            avgTimes[ i, 0 ] = torch.mean( encodConvergTimes ) 
            avgTimes[ i, 1 ] = torch.mean( retenConvergeTimes ) 
            
            
            
    fig = plt.figure( )
    plt.scatter( xPts, avgTimes, c=colors )
    
    plt.ylabel( 'Average time to converge' )
    
    
    plt.xlabel( r'$\alpha$' )
    
    
    
    return fig 


    
    





def plotBifurcationOfStateOverAlpha( testingDataDict, simOptions, nBetas, betaInd = 0,
                                              epochNum=None,
                                              ):
    
    

    if epochNum is None:
        epochNum = simOptions[ 'parameters' ][ 'nEpochs' ]
    
    
    
    
    stateVals = np.zeros( [N, ] )
    
    
    for i in range( len(varVals) ):
        
        alphaVal = varVals[ i ]
        
        testingData = testingDataDict[ alphaVal ]
    
    
        state = testingData[ epochNum ][ 'state' ]
    
    
    
    
    return 
    



def plotBifurcationOfStateOverTheta( testingData, testingInput, simOptions,
                                              epochNum=None,
                                              ):
    
    
    if epochNum is None:
        epochNum = simOptions[ 'parameters' ][ 'nEpochs' ]
    
    
    state = testingData[ epochNum ][ 'state' ]

    N = state.shape[0]
    
    
    thetas = testingInput.thetas[0].detach().numpy()
    degrees = thetas * ( 180 / math.pi ) 
    # nStims = thetas.shape[ -1 ]

    sortingInds = np.argsort( thetas )
    
    convergedStates = testingData[ epochNum ]['convergedStates_retention'][:,1::].detach().numpy()
    convergedStates_sorted = convergedStates[ :, sortingInds ]
    
    
    
    
    fig, axs = plt.subplots( 15, 1 )

    
    
    for i in range( N ):
        print(degrees.shape)
        print(convergedStates[i].shape)
        
        axs[i].scatter( degrees, convergedStates[i], s=2 )
        
        axs[i].xaxis.set_visible( False )
        
        for spine in ['top', 'right', 'left', 'bottom']:
            axs[i].spines[spine].set_visible(False)
        
        axs[i].set_yticks( [] )
        
        axs[i].set_ylabel( i+1 )
        
        
    # axs[-1].set_xt( 'Stimulus angle' )
    axs[-1].spines['bottom'].set_visible(True)
    plt.suptitle( '' )
    fig.supylabel( 'Subnetwork index' )
    fig.supxlabel( 'Stimulus angle' )


    # axs[-1].set_xticklabels(  )
    
    
    return fig 
    
    
    
    
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% ANIMATED stim lifecycle (PCA)
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


    
'''
Source:  https://matplotlib.org/stable/gallery/animation/simple_scatter.html


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.set_xlim([0, 10])

scat = ax.scatter(1, 0)
x = np.linspace(0, 10)


def animate(i):
    scat.set_offsets((x[i], 0))
    return scat,

ani = animation.FuncAnimation(fig, animate, repeat=True,
                                    frames=len(x) - 1, interval=50)

# To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('scatter.gif', writer=writer)

plt.show()
'''
  
    
  
    
  
    
  
import matplotlib.animation as animation

from functools import partial

import shutil



  
    
def getStimLifecycleTimeInds( simOptions, stimInd, testing=True ):
    ''' Assumes stimulus indexing starts with 0. '''
    
    
    if testing:
        nTotalCycleSteps = simOptions[ 'nEncodSteps_testing' ] + simOptions[ 'nRetenSteps_ext' ]
    else: 
        nTotalCycleSteps = simOptions[ 'nEncodSteps_training' ] + simOptions[ 'nRetenSteps_training' ]
        

        
    startInd = (stimInd * nTotalCycleSteps) + 1          ## Account for IC with the +1 
    endInd = startInd + nTotalCycleSteps
    
    lifecycleTimeInds = list(  range( startInd, endInd )  )
    
    
    return lifecycleTimeInds 






def getTimeIndsForLifecyclePt( simOptions, cycleTimePt, nBetas, extendedReten=True, testing=True ):
    
    
    if extendedReten:
        nRetenSteps = simOptions[ 'nRetenSteps_ext' ]
    else: 
        nRetenSteps = simOptions[ 'nRetenSteps' ]


    if testing:
        nEncodSteps = simOptions[ 'nEncodSteps_testing' ]
    else:
        nEncodSteps = simOptions[ 'parameters' ][ 'nEncodSteps_training' ]

    
    nTotalCycleSteps = nEncodSteps + nRetenSteps
    
    start = cycleTimePt                             ## timePt for first stim
    end = start + (nBetas * nTotalCycleSteps)       ## timePt for last stim
    


    timeInds = list(  np.arange( start, end, nTotalCycleSteps)  )

    
    return timeInds




def reorganizeStateProjIntoCycles( stateProj, simOptions, nBetas, PCAk=3, cycleLength=None ):
    
    if cycleLength is None:
        nTimes = stateProj.shape[0]
        cycleLength = int(  (nTimes-1) / nBetas  )              ## Account for IC via the -1
    
    
    cycleData = np.zeros(  [ cycleLength, nBetas, PCAk ]  )


    for cycleTimePt in range( cycleLength ):
        # currTimeInds = getTimeIndsForLifecyclePt( simOptions, cycleTimePt, nBetas )
        
        start = cycleTimePt + 1
        end = cycleTimePt + ((nBetas-1) * cycleLength) + 1                         # timePt for last stim
        # timeInds = list(  np.arange( start, end, cycleLength)  )
        timeInds = list(  np.arange( start, nTimes, cycleLength)  )
        
        cycleData[ cycleTimePt ] = stateProj[ timeInds, : ]
        

    
    return cycleData









def animatedStimLifecyclePCA( simOptions, testingData, testingInput, epochNum=None, 
                                         trimRetentionLen=0,
                                         
                                         PCsToUse=None,
                                         
                                         cmap=mpl.colormaps['viridis'], 
                                         
                                         save=True, fileType='.gif', 
                                         saveDir=None, useLongFilename=True, filenameBase='',
                                         
                                         includeTimeBar=True, 
                                         PCAonCurrStates=False,
                                          ):
    
    ''' Color-coordinated to stimulus degree ''' 
    
    
    

    
    
    ''' PCA the state '''
    #-------------------------------------
    if epochNum is None:
        epochNum = simOptions[ 'epochNumsToTest' ][ -1 ]    ## nEpcohs
    state = testingData[ epochNum ][ 'state' ]              ## ( N, nTimes )
    
    
    # if not PCAonCurrStates:
    if PCsToUse is None:
        stateProj = PCA( state.T ).detach().numpy()         ## ( nTimes, k )
    
    else: 
        if PCsToUse.shape[0] == state.shape[0]:
            stateProj = torch.matmul(  state.T,  PCsToUse[:, :3]  ).detach().numpy()         
        else:
            raise Exception( 'Shape of given principal components are not compatible' )
                  
    
    
    ''' Stimulus info '''
    #-------------------------------------
    thetas = testingInput.thetas[0]
    degrees = thetas * ( 180 / math.pi ) 
    
    sortingInds = np.argsort( degrees )
    
    nPts = nStims = len( degrees )                                   ## nPts = nBetas = nStims
    colors = cmap(   [ x/nPts for x in range(nPts) ]   )
    
    
    
    
    
    
    ''' Cycle length '''
    #-------------------------------------    
    nRetenSteps_ext = nRetenSteps = simOptions[ 'nRetenSteps_ext' ]    
    nEncodSteps = simOptions[ 'nEncodSteps' ]
        
    trainedDelay = nEncodSteps + simOptions[ 'nRetenSteps' ] 

    # nTimes = stateProj.shape[0]
    # cycleLength = int(  (nTimes-1) / nStims  ) 
    cycleLength = nEncodSteps + nRetenSteps_ext
    

        

    ''' Reorganize the data '''
    #-------------------------------------
    # if PCAonCurrStates:
        
        
    toPlot = reorganizeStateProjIntoCycles( stateProj, simOptions, nStims )     ## ( cycleLength, nBetas, k )
    toPlot_sorted = toPlot[ :, sortingInds ]                                    ## ( cycleLength, nBetas, k )

    
    if trimRetentionLen > 0:
        cycleLength = cycleLength - trimRetentionLen
        toPlot_sorted = toPlot_sorted[ 0:-trimRetentionLen ]        ## ( cycleLength_trimmed,  nBetas,  k  )

    


    
    
    ''' Set up the fig '''
    #-------------------------------------
    fig, ax = plt.subplots(  figsize=(6, 6),  subplot_kw={'projection': '3d'}  )
    # fig, ax = plt.subplots(  figsize=(6, 6),  subplot_kw={'projection': '3d'},  bbox_inches='tight' )
    
    
    # fig, axs = plt.subplots( 2, 1, figsize=(6, 6),  subplot_kw={'projection': '3d'}, height_ratios=[4, 1] )

    # ax = axs[0]
    # timeAx = axs[1]
    
    # newBbox = mpl.transforms.Bbox.from_bounds(  0.2, 0.2, 0.8, 0.8  )
    # ax.bbox = newBbox
    
    # ax = plt.axes(  [0.2, 0.2, 0.8, 0.8]  )         ## (x0, y0, width, height)


    ax.set_xlabel( 'PC 1' )
    ax.set_ylabel( 'PC 2' )
    ax.set_zlabel( 'PC 3' )
    
    
    titleStr = 'Evolution of stimulus lifecycles'
    alphaRefStr = 'eA' + str(simOptions['encodingAlpha']) + '_' + 'rA' + str(simOptions['retentionAlpha'])
    fig.suptitle( titleStr + '\n' + alphaRefStr ) 
    
    
    maxVals = np.max( stateProj, axis=0 )
    minVals = np.min( stateProj, axis=0 )
    ax.set_xlim(  [ minVals[0], maxVals[0] ]  )
    ax.set_ylim(  [ minVals[1], maxVals[1] ]  )
    ax.set_zlim(  [ minVals[2], maxVals[2] ]  )

    

    scatPCA = ax.scatter(  toPlot_sorted[0,:,0],  toPlot_sorted[0,:,1],  toPlot_sorted[0,:,2],  s=15,  c=colors  )

    

    
    
    
    ''' Add a time bar '''
    #-------------------------------------
    if includeTimeBar:
        fig.subplots_adjust( bottom=0.2 )          
        timeAx = plt.axes(  [0.1, 0.11, 0.9, 0.03]  )      ## (x0, y0, width, height)
        
        
        # timeAx.scatter( 0,0 )
        timeAx.axvline( 0, c='red' ) 
        
                
        timeAx.set_xlim( [0,cycleLength] )
        # timeAx.set_xticks( [ nEncodSteps ],  [ 'Delay begins' ]  )
        timeAx.set_xticks( [ nEncodSteps, trainedDelay ],  [ 'Delay \nbegins', 'Trained \ndelay' ]  )
        # timeAx.axvline( nEncodSteps, c='red' )
        
        timeAx.set_xlabel( 'Cycle length' )
        
        timeAx.set_yticks( [] )
        # timeAx.spines['top'].set_visible(False)
        # timeAx.spines['right'].set_visible(False)
        # timeAx.spines['bottom'].set_visible(False)
        # timeAx.spines['left'].set_visible(False)
        
        
                
    

    # return fig



    ''' Colorbar '''
    #-------------------------------------
    fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side
    # fig.subplots_adjust( right=0.7 )                                        ## create space on the right hand side
    
    axPos = ax.get_position()
    
    # cbarAx = plt.axes(  [0.97, axPos.y0, 0.02, axPos.height]  )
    cbarAx = plt.axes(  [0.95, axPos.y0, 0.02, axPos.height]  )
    # cbarAx = plt.axes(  [0.75, axPos.y0, 0.02, axPos.height]  )


    cbar = plt.colorbar( scatPCA, label='Stimulus degree', cax=cbarAx ) 
    
    nTicks = 7
    cbar.set_ticks(  np.linspace( 0, 1, nTicks )   )
    cbar.set_ticklabels(   [ int(x) for x in  np.linspace( 0, 360, nTicks ) ]   )

    

    

    ''' Animate: plot each cycleTimePt '''
    #-------------------------------------
    
    # ani = animation.FuncAnimation( fig, animatePCA, repeat=True,
    #                                     frames=cycleLength-1, interval=50 )
    # ani = animation.FuncAnimation( fig, partial(animateData, pathCollObj=scatPCA, data=toPlot_sorted), 
    #                                       repeat=True, frames=cycleLength-1, interval=50 )
    
    interval = 500
    interval = 1000
    # interval = 50
    
    if includeTimeBar:
        ani = animation.FuncAnimation( fig, partial(animateDataWithTimeBar, fig=fig, data=toPlot_sorted), 
                                              repeat=True, frames=cycleLength-1, interval=interval )
    else:
        ani = animation.FuncAnimation( fig, partial(animateData, pathCollObj=scatPCA, data=toPlot_sorted), 
                                              repeat=True, frames=cycleLength-1, interval=interval )
        
    
    fig.subplots_adjust( right=0.9 )                                        ## create space on the right hand side

    # plt.show()

    


    ''' Save '''
    #-------------------------------------
    if save: 
        
        writer = animation.PillowWriter( fps=15, bitrate=1800 )
        
        
        
        if useLongFilename: 
            
            aniFilename = getLongFilename( simOptions, '', fileType )
            
            aniFilename = filenameBase + aniFilename
            
            # # aniFilename = 'stimLifecyclePCA_'  +  alphaRefStr.replace( '_', '-' )
            #  # longFilename = 'maxIter' + str( simOptions['parameters']['maxIter'] ) 
            #  longFilename = 'maxIter' + str( simOptions['maxIter'] ) 
            #  longFilename = longFilename + '_' + alphaRefStr.replace( '_', '-' )
            #  longFilename = longFilename + '_nEn' + str( simOptions['nEncodSteps'] ) + 'nRe' + str( simOptions['nRetenSteps'] ) 
            #  longFilename = longFilename + '_WC' + simOptions['weightCombos'][0].replace( '_', '-' )
            #  longFilename = longFilename + '_nEpochs' + str( simOptions['parameters']['nEpochs'] ) 
 
            #  aniFilename = longFilename + fileType        

        else: 
        
            aniFilename = 'stimLifecyclePCA_'  +  alphaRefStr.replace( '_', '-' )
            # phaseStepStr = '_nEn' + str( simOptions['nEncodSteps'] ) + 'nRe' + str( simOptions['nRetenSteps'] )   
            phaseStepStr = '_nEn' + str( simOptions['nEncodSteps_testing'] ) + 'nRe' + str( simOptions['nRetenSteps'] )   
            aniFilename = aniFilename  +  phaseStepStr  +  fileType
            
            
            aniFilename = filenameBase + aniFilename
            


        
        if saveDir is None:
            # saveLoc = aniFilename 
            saveDir = [ aniFilename ]
        elif type(saveDir) is str:
            saveDir = [ saveDir ]

        


        for i in range( len(saveDir) ):
            
            saveDirName = saveDir[ i ]
            
            if not os.path.exists( saveDirName ):
                os.makedirs( saveDirName )
            
            
            if i == 0:
                saveLoc = os.path.join( saveDirName, aniFilename ) 
                ani.save( saveLoc, writer=writer )
                # ani.save( saveLoc, writer=writer, bbox_inches='tight' )
                print( 'Saved animation to: ' )
                print( '\t ', saveDirName )
                
            else: 
                newSaveLoc = os.path.join( saveDirName, aniFilename )
                
                shutil.copyfile( saveLoc, newSaveLoc )
                print( '\t ', saveDirName )
        

                

        

        return ani, aniFilename


    
    else: 
        
        return ani 







# def animateData( frameInd, pathCollObj, data ):
def animateDataWithTimeBar( frameInd, fig, data ):
    ''' 
        BethAnna Thompson, August 1, 2024
    
        Animate by adjusting the path collection to the next point in the dataset 
    
        Data must be of size ( nFrames, origSize ) where origSize is the origData size 
    and must be the canonical input to pathCollObj
    
    '''
    
    
    [ scatterAx, timeAx, cbarAx ] = fig.get_axes()
    
    timeAx.axvline( frameInd, c='gray' )
    # timeAx.scatter( frameInd, 0, s=10, c='red' )

    scatterPlot = scatterAx.collections[ 0 ]
    scatterPlot._offsets3d = (  data[ frameInd, :, 0 ],  data[ frameInd, :, 1 ],  data[ frameInd, :, 2 ]  )



    return fig







def animateData( frameInd, pathCollObj, data ):
    ''' 
        BethAnna Thompson, August 1, 2024
    
        Animate by adjusting the path collection to the next point in the dataset 
    
        Data must be of size ( nFrames, origSize ) where origSize is the origData size 
    and must be the canonical input to pathCollObj
    
    '''
    
    pathCollObj._offsets3d = (  data[ frameInd, :, 0 ],  data[ frameInd, :, 1 ],  data[ frameInd, :, 2 ]  )



    return pathCollObj






    

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#%% Extract Encoding Phase
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------



def extractEncodingPhase( simOptions, state, stimInds=None ):
    '''  '''
    
    #---------------------------------------------------
    ''' Timing ''' 
    #---------------------------------------------------
    [ N, nStates ] = state.shape 
    nTimes = nStates - 1                        ## Remove initial condiiton
    
    nEncodSteps = simOptions[ 'nEncodSteps' ]
    nRetenSteps = simOptions[ 'nRetenSteps' ]
    nTotalPhaseSteps = nEncodSteps + nRetenSteps
    
    stimTimeInds = list(  range( 1, nStates, nTotalEvolSteps )  )
    
    #-------------------------------------------
    
    if stimInds is None:
        stimTimes = stimTimeInds
    else: 
        stimTimes = stimTimeInds[ stimInds ]
    
    


    #---------------------------------------------------
    ''' Extract ''' 
    #---------------------------------------------------
    [ encodingTimeInds, retentionTimeInds ] = getDiffPeriodTimeInds( nEncodSteps, nRetenSteps, stimTimes )

    
    encodingPhases = state[ :, encodingTimeInds ]
    retentionPhases = state[ :, retentionTimeInds ]
    
    


    return encodingPhases, retentionPhases







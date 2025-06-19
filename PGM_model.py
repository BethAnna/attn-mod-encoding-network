#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:14:06 2024

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

from torch.utils.data import Dataset, DataLoader


import numpy as np 
import random


import math 






#----------------------------
''' Visualization '''
#-----------------------------
import matplotlib.pyplot as plt



#-----------------------------
''' Saving '''
#-----------------------------
import pickle 








#=========================================================================================
#=========================================================================================
#%% MODEL 
#=========================================================================================
#=========================================================================================




class memoryModel( nn.Module ):


    # def __init__( self, signalDim, networkDim, weightCombo, trainMatsDirectly=True, requireStable=True ):
    def __init__( self, signalDim, networkDim, weightCombo, simOptions ):
        '''  '''
        
        super().__init__() 
        
        #---------------------------------------------------------------------------------
        ''' Model contextual info (hyperparameters) '''
        #---------------------------------------------------------------------------------
        self.signalDim = signalDim
        self.networkDim = networkDim
        
        
        self.weightCombo = weightCombo    
        [ weights, weightDict ] = weightComboInfo( weightCombo )
        # [ errW, effW, sW, hW, rW, fW ] = weightComboInfo( weightCombo )
        self.errorWeight = weightDict[ 'err' ]
        self.efficiencyWeight = weightDict[ 'eff' ]
        self.sparsityWeight = weightDict[ 's' ]
        self.historyWeight = weightDict[ 'h' ]
        self.retentionWeight = weightDict[ 'r' ]
        
        self.frugalityWeight = weightDict[ 'f' ]
        
        
        
        #---------------------------------------------------------------------------------
        ''' Attentional parameter '''
        #---------------------------------------------------------------------------------
        # self.alpha = simOptions['parameters'][ 'alpha' ]
        # self.alphaType = simOptions[ 'alphaType' ]
        
        self.retentionAlpha = simOptions[ 'retentionAlpha' ]
        self.encodingAlpha = simOptions[ 'encodingAlpha' ]

        
        #---------------------------------------------------------------------------------
        ''' Initialize the parameters to learn '''
        #---------------------------------------------------------------------------------
        self.requireStable = simOptions['requireStable']            ## Require H to be Hurwtiz 
        self.trainMatsDirectly = simOptions['trainMatsDirectly']    ## Train D,H  (not W, Ms, Md )
        
        self.processingTermWithD = simOptions[ 'processingTermWithD' ]
        self.addStimSpaceProcessTerm = simOptions[ 'addStimSpaceProcessTerm' ]
        
        
        if self.trainMatsDirectly:
            self.parameterNames = [ 'D', 'H' ]
        else:
            self.parameterNames = [ 'W', 'Md', 'Ms' ]
            
        
        self = self.setInitConnections( useQ2Also=simOptions['useQ2Also'] )
        
        
        L = self.LipschitzConstant()
        self.stepSize = 1 / L
        
        # self.trainingVars = None            # for Kafashan learning 
        
        
    
    
    #=====================================================================================
    ''' Evolution Functions '''
    #=====================================================================================
    
    
        
    def forward( self, currStim, referenceState, currState, useQ2Also=False, refState2=None ):
        ''' One proximal step (one time step) '''
        
        #currStim = currStim.reshape( self.signalDim, 1 )#.double()
        #referenceState = referenceState.reshape( self.networkDim, 1 )#.double()
        #currState = currState.reshape( self.networkDim, 1 )#.double()
        
        
        #---------------------------------------------------------------------------------
        ''' Linear Gradient ''' 
        #---------------------------------------------------------------------------------  
        linearGrad = self.gradF( currStim, referenceState, currState, useQ2Also=useQ2Also, refState2=refState2 )
        
        # print('linearGrad', linearGrad)
        #---------------------------------------------------------------------------------
        
        
            
        #---------------------------------------------------------------------------------
        ''' Compute proximal of the gradient step ''' 
        #---------------------------------------------------------------------------------   
        proxInput = currState  -  self.stepSize * linearGrad             # ( N, 1 )
        
        
        
        #----------------------------------------------
        if torch.any(  torch.isnan( proxInput )  ):
            print( 'NaN found in proxInput! ' )
            print( )
            print( '\t epochNum =', self.epochNum )
            print( '\t currStim =', currStim.T )
            print( )

            # print( 'currState', currState )
            # print( 'referenceState', referenceState )
            # print( 'currStim', currStim )
            # print( 'linearGrad', linearGrad )
            # print( 'proxInput', proxInput )
            # print( )
        #---------------------------------------------- 
        
        
        
        sWeight = self.sparsityWeight     
        # newState = self.proximal_l1( proxInput, self.stepSize, sWeight ) 
        newState = proximal_l1( proxInput, self.stepSize, sWeight ) 
        
        
        
        #----------------------------------------------------------
        ''' If linear, returns linear gradient step (==proxInput) ''' 
        #----------------------------------------------------------   
        if sWeight == 0:
            # testClose = np.all( np.isclose( proxInput, newState ) )
            testClose = torch.all( torch.isclose( proxInput, newState ) )
            
            if not testClose:
                print( '[updateState_PGM] Proximal for sWeight=0 is incorrect ' )
        

        #---------------------------------------------------------------------------------



        return newState
        # return newState, proxInput
        
    
    
    
    
    
    def computeAlpha( self, currStim=None, currState=None, refState=None ):
    
        
        ## THIS NEEDS TO CHANGE IF THE STIMS ARE NOISY!!!
        
        #---------------------------------------------------------------------------------
        ''' If alpha is a fixed value: '''
        #---------------------------------------------------------------------------------
        if torch.norm( currStim, p=2 ) == 0:
            alpha = self.retentionAlpha
            
        else:
            alpha = self.encodingAlpha
        
        
            
        #---------------------------------------------------------------------------------
        ''' If alpha is a type instead of fixed value: '''
        #---------------------------------------------------------------------------------
        if type( alpha ) is str: 
            
            if alpha == 'context':
                # decoded = self.D @ currState
                decoded = self.D @ refState
                toCorr = torch.tensor([decoded, currStim])
                # corr = torch.corrcoef( decoded, currStim )[ 0, 1 ]  
                corr = torch.corrcoef( toCorr )[ 0, 1 ]  
                alpha = corr 
                
            else: 
                raise Exception( '[computeAlpha] Model attentional parameter not understood'  )
                
        
                
        return alpha
            
            
        
    
    
        # # alphaType = self.alphaType
    
    
        # #--------------------------------------------
        # ''' Simple "on/off" '''
        # #--------------------------------------------
        # if alphaType == 'simple':
            
        #     # alpha = 0.9
        #     # # alpha = 0.7

            
        #     if torch.norm( currStim, p=2 ) == 0:
        #         alpha = 0.1
        #         # alpha = 0.05
        #         # alpha = 0
        #         # alpha = 0.01
        #         # alpha = 0
        #         # alpha = 0.2
        #         # alpha = self.retentionWeight
        #     else:
        #         alpha = 0.8
        #         # alpha = 0.9
        #         # alpha = 1
                
                
                
        # elif alphaType == 'context':
        #     # decoded = self.D @ currState
        #     decoded = self.D @ refState
        #     toCorr = torch.tensor([decoded, currStim])
        #     # corr = torch.corrcoef( decoded, currStim )[ 0, 1 ]  
        #     corr = torch.corrcoef( toCorr )[ 0, 1 ]  
        #     alpha = corr 
            
                
                
        # else: 
        #     alpha = 0.9
    
    
    
        # return alpha
    
    
    
    
    
    
    
    def gradF( self, currStim, referenceState, currState, 
                                              historyReconstr=False, 
                                              useQ2Also=False, refState2=None ):
        ''' 
        Gradient of 
                f(x) =  (errWeight/2) * f_err(x)   +   (effWeight/2) * f_eff  
                            +   (hWeight/2) * f_h(x)   +   (rWeight/2) * f_r(x)   
                            +   (fWeight/2) * f_f(x),
        where 
                f_err(x)    =   ||  currStim - C*currState  ||_2^2
                f_eff(x)    =   ||  currState  ||_2^2
                f_h(x)      =   ||  referenceState - H*currState  ||_2^2 
                f_r(x)      =   ||  currState - referenceState  ||_2^2 
                
                f_f(x)      =   ||  currState - A*referenceState  ||_2^2. 
        '''
        
        
        if useQ2Also:
            if refState2 is None:
                raise Exception( '[gradF] Cannot use q=2 cost term since referenceState2 was not given. ' )
        
        
        
        
        #-------------------------------------------------------------------------------------
        ''' The gradient '''
        #-------------------------------------------------------------------------------------
        if hasattr(self,'D')  and  (self.D in self.parameters()):
            [ D, H ] = [ self.D, self.H ]
            
            
            if useQ2Also:
                [ W, Md, Md2, Ms, Ms2 ] = self.computeConnectionMats( D, H, useQ2Also=useQ2Also )
                # Md2 = Md2.detach() 
                # Ms2 = Ms2.detach() 
            else:
                [ W, Md, Ms ] = self.computeConnectionMats( D, H )
                
                
            # [ W, Md, Ms ] = self.computeConnectionMats( D, H )

                
        # else: 
        #     [ D, H ] = self.computeDandH( )
        #     [ W, Md, Ms ] = [ self.W, self.Md, self.Ms ]
            
            
            
        # D = D.detach()
        # H = H.detach()
        
        # W = W.detach()
        # Md = Md.detach()
        # Ms = Ms.detach()
        
        
    
    
    
        
        
        #-------------------------------------------
        ''' FEEDFORWARD Connections '''
        #-------------------------------------------
        grad1 = (-1)  *  (W @ currStim)                             
        # grad1 = W @ currStim                 
        
        
        #-------------------------------------------
        ''' RECURRENT Connections  '''
        #-------------------------------------------
        grad2 = Md @ currState           
              
        if useQ2Also:
            grad2b = Md2 @ currState 
            grad2 = grad2 + grad2b
        
        
        #-------------------------------------------
        ''' DELAYED RECURRENT Connections  '''
        #-------------------------------------------
        grad3 = (-1)  *  (Ms @ referenceState)
        
        if useQ2Also:
            grad3b = (-1) *  (Ms2 @ refState2) 
            grad3 = grad3 + grad3b
        
        
        
        #-------------------------------------------
        ''' Efficiency constraint '''
        #-------------------------------------------
        grad4 = self.efficiencyWeight * currState                  
                
        
        
        #---------------------------------------------------------------------------------
        ''' Put it all together '''
        #---------------------------------------------------------------------------------
        grad = grad1 + grad2 + grad3 + grad4
        grad = grad.detach()


        return grad





    # def proximal_l1( self, proxInput, stepSize, sWeight ):
    # # def proximal_muG( proxInput, stepSize, sWeight ):
    #     ''' 
    #         Computes the proximal for a given function and over input proxInput x:
    #             prox_g(x) = arg min, z  {   (1/2)*|| x - z ||_2^2   +   g(x)   }
    #     where we assume 
    #             g(x) = sWeight * ||  x  ||_1, 
    #     resulting in a soft-thresholding or "deadzone nonlinearity."
    #     '''
            
        
        
    #     if sWeight == 0:
    #         proxOutput = proxInput


    #     else:     
    #         threshold = stepSize * sWeight
    #         proxOutput = self.softThreshold( proxInput, threshold )
            
            
            
    #         ''' Compare to the method Kafashan used ''' 
    #         #-----------------------------------------------------------------------------
    #         N = proxInput.shape[0]
    #         zeros = torch.zeros( N,1 )
    #         proxOutput2 = torch.maximum(zeros, proxInput - threshold)  -  torch.maximum(zeros, -proxInput - threshold)
            
    #         if torch.all( proxOutput != proxOutput2 ):
    #             print( )
    #             print( 'Different proxOutput from Kafashan' )
            
        
        
    #     return proxOutput
        
        
        
    
    # def softThreshold( self, inputVector, threshold ):
    
    #     #----------------------------------------------
    #     d1 = inputVector.shape[0]
    #     stOutput = torch.zeros( [ d1, 1 ] )        # assumes inputVector is a column vector 
    #     #----------------------------------------------
    
    
    #     ''' Threshold each element of the input '''
    #     #----------------------------------------------
    #     stPattern = [ ]
        
        
    #     # print( 'inputVector', inputVector )
    #     # print( 'inputVector.shape', inputVector.shape )
        
        
    #     for i in range( d1 ):
            
    #         ri = inputVector[ i ]
            
    #         # print( 'ri', ri )
    #         # print( 'threshold', threshold )
            
    #         if ri > threshold: 
    #             stOutput[ i ] = ri - threshold
    #             stPattern.append( '-' )
                
    #         elif abs(ri) <= threshold:
    #             stOutput[ i ] = 0
    #             stPattern.append( '0' )
                
    #         elif ri < (-1)*threshold:
    #             stOutput[ i ] = ri + threshold
    #             stPattern.append( '+' )
                
    #         else: 
    #             print( 'ri', np.round(ri,2) )
    #             print( 'threshold', threshold )
    #             print( 'stOutput[i]', np.round(stOutput[i],2) ) 
                
    #             raise Exception( '[softThreshold] Error in thresholding operation' )
    #     #----------------------------------------------
        
    #     # print( 'Soft-thresholding pattern: ', stPattern )
        
    
    
    #     return stOutput

        






    
    #=====================================================================================
    ''' Initialization Functions '''
    #=====================================================================================

    
    def setInitConnections( self, W=None, Md=None, Ms=None,  D=None, H=None, useQ2Also=False ):
        
        
        if 'D' in self.parameterNames:

            
            #-----------------------------------------------------------------------------
            ''' Decoder and history mats '''
            #-----------------------------------------------------------------------------
            if (D is None) or (H is None):
                
                
                ''' If only one was given, ask for user input on how to proceed '''
                #-------------------------------------------------------------------------
                if (D is not None) or (H is not None): 
                    connInput = input( '\n[memoryModel.setInitConnections] Redefining both D and H even though one was given. \n\t Continue? (Y/N) ' )
                    if (connInput == 'n') or (connInput == 'N'):  
                        raise Exception( '[memoryModel.setInitConnections] Stopping in response to user input. ' )
            
            
                D, H = self.initDandH( )

            
            #------------------------------------------------------
            self.D = nn.Parameter( D, requires_grad=True )
            self.H = nn.Parameter( H, requires_grad=True ) 
            #------------------------------------------------------
            
            #------------------------------------------------------
            if useQ2Also:
                [ self.W, self.Md, self.Md2, self.Ms, self.Md2 ]  = self.computeConnectionMats( D, H, useQ2Also=useQ2Also )
            else:
                self.W, self.Md, self.Ms = self.computeConnectionMats( D, H )
            #------------------------------------------------------
        
        
        
        else: 
        
            #-----------------------------------------------------------------------------
            ''' Connection matrices '''
            #-----------------------------------------------------------------------------
            if (W is None) or (Md is None) or (Ms is None):
                
                if (W is not None) or (Md is not None) or (Ms is not None): 
                    connInput = input( '\n[memoryModel.setInitConnections] Redefining all W, Md, Ms even though some were given. \n\t Continue? (Y/N) ' )
                    if (connInput == 'n') or (connInput == 'N'):  
                        raise Exception( '[memoryModel.setInitConnections] Stopping in response to user input. ' )
                
                self.W = nn.Parameter( torch.empty((self.networkDim, self.signalDim)), requires_grad=True )
                self.Ms = nn.Parameter( torch.empty( (self.networkDim, self.networkDim) ), requires_grad=True )
                self.Md = nn.Parameter( torch.empty( (self.networkDim, self.networkDim) ), requires_grad=True )
    
    
                W, Md, Ms = self.computeConnectionMats( D, H )
    
            
            ''' Assign '''
            #------------------------------------------------------
            self.W = nn.Parameter( W, requires_grad=True )
            self.Md = nn.Parameter( Md, requires_grad=True )
            self.Ms = nn.Parameter( Ms, requires_grad=True )
            #------------------------------------------------------
            
            # D, H = self.computeDandH( )
            
        
    
        return self
    
    
    
    
    
    # def initDandH( self, requireStable=True, initDNorm=4, initHNorm=8 ):
        
    # def initDandH( self, initDNorm=0.7, initHNorm=0.4 ):
    # def initDandH( self, initDNorm=1.5, initHNorm=0.1 ):
    # def initDandH( self, initDNorm=1, initHNorm=1 ):
    # def initDandH( self, initDNorm=3, initHNorm=0.2 ):
    # def initDandH( self, initDNorm=3, initHNorm=1.5, complexEigvals=True ):
        
        
    # def initDandH( self, initDNorm=3, initHNorm=1.5, complexEigvals=True ):
    # def initDandH( self, initDNorm=1, initHNorm=0.75, complexEigvals=True ):
        
    def initDandH( self, initDNorm=1, initHNorm=1, complexEigvals=True ):
    # def initDandH( self, initDNorm=1, initHNorm=1, complexEigvals=True ):
    # def initDandH( self, initDNorm=0.5, initHNorm=0.5, complexEigvals=True ):
    # def initDandH( self, initDNorm=1.25, initHNorm=1.25, complexEigvals=True ):

        
        
    # def initDandH( self, initDNorm=3, initHNorm=5 ):
    # def initDandH( self, initDNorm=5, initHNorm=5, complexEigvals=True ):
        
    # def initDandH( self, initDNorm=1, initHNorm=0.5 ):
    # def initDandH( self, initDNorm=1.25, initHNorm=0.5 ):
    # def initDandH( self, initDNorm=1.25, initHNorm=1 ):
    # def initDandH( self, initDNorm=1.5, initHNorm=0.5 ):


        
    # def initDandH( self, initDNorm=0.7, initHNorm=1 ):
    # def initDandH( self, initDNorm=1.2, initHNorm=4 ):
        
        
    # def initDandH( self, initDNorm=2, initHNorm=4 ):
    # def initDandH( self, initDNorm=3.5, initHNorm=2.5 ):
    # def initDandH( self, initDNorm=0.8, initHNorm=3 ):
        
    # def initDandH( self, requireStable=True, initDNorm=1, initHNorm=0.55 ):
        ''' 
            Create decoder D and history reconstruction matrices H from scratch (when W, Ms, 
        and Md are not defined, e.g.).   
        '''
                
        
        
        #---------------------------------------------------------------------------------
        ''' Set the seed for consistent randomness for sanity-checking '''
        #---------------------------------------------------------------------------------
        seedNum = 0
        seedNum = 10
        
        torch.manual_seed( seedNum )
        random.seed( seedNum )
        np.random.seed( seedNum )
        
        
        
        
        
        #---------------------------------------------------------------------------------
        ''' Make the decoder D '''
        #---------------------------------------------------------------------------------
        
        # D = fullRankMatrix( self.signalDim, self.networkDim, requireStable=requireStable )
        D = torch.rand( self.signalDim, self.networkDim )
        
        
        #----------------------------------------------
        ''' Force it to have given norm value '''
        #----------------------------------------------
        Dnorm = torch.norm( D, p='fro' )
        D = (D / Dnorm) * initDNorm 
        
        
        
        
        
        
        #---------------------------------------------------------------------------------
        ''' Make the reconstructor H '''
        #---------------------------------------------------------------------------------
        
        
        if self.requireStable:
            # desiredEigvals = np.ones( [self.networkDim, 1] )
            # desiredEigvals = np.random.uniform( 0, 1, self.networkDim )
            desiredEigvals = np.random.uniform( -1, 0, self.networkDim )
            
            if complexEigvals:
                desiredEigvals_imag = np.random.uniform( -1, 0, self.networkDim )
                desiredEigvals = np.array( [   complex( desiredEigvals[i], desiredEigvals_imag[i] )   for i in range( len(desiredEigvals) )  ] )
            
            H = matrixFromEigvals( desiredEigvals )[0]
            H = torch.as_tensor( H, dtype=torch.float32 )
            

            
            
        else:
            # H = fullRankMatrix( self.networkDim, self.networkDim, requireStable=requireStable )
            H = torch.rand( self.networkDim, self.networkDim )
            
            
            #----------------------------------------------
            ''' Force it to have given norm value '''
            #----------------------------------------------  
            Hnorm = torch.norm( H, p='fro' )
            H = (H / Hnorm) * initHNorm

        

        # #----------------------------------------------
        # ''' Force it to have given norm value '''
        # #----------------------------------------------     
        # Hnorm = torch.norm( H, p='fro' )
        # H = (H / Hnorm) * initHNorm           




        #---------------------------------------------------------------------------------
        ''' Finalize '''
        #---------------------------------------------------------------------------------
        D = torch.as_tensor( D, dtype=torch.float32 ) 
        H = torch.as_tensor( H, dtype=torch.float32 )
        
        
        return D, H
    
    
    
    
    
    
    # def computeNullSoln( self, refState, prevState ):  
    
    #     D = self.D.detach().numpy()
    #     H = self.H.detach().numpy()
        
    #     N = H.shape[0]
        
    
    #     nullspace = computeNullspace( D )           ## ( N, nBasisVectors )
        
        
    #     DtD = D.T @ D
    #     HtDtDH = H.T @ DtD @ H
    #     HtDtDH_inv = np.linalg.pinv( HtDtDH )
        
    #     optimalState = HtDtDH_inv @ HtDtDH @ refState.detach().numpy()
    #     optimalNullVect = optimalState - prevState.detach().numpy()
    #     # scalars = np.linalg.solve( nullspace, optimalNullVect )          ##  solve Ax = b for x 
        
    #     isNullBool =  np.isclose( D @ optimalNullVect, 0 )
    #     if np.all( isNullBool ):
    #         optimalNullVect = optimalNullVect.reshape( N, 1 ).detach()
    #         return torch.tensor( optimalNullVect )
    #     else:
    #         soln = np.sum( nullspace, axis=1 )          ## ( N, 1 )
    #         soln = soln.reshape( N, 1 )
    #         return torch.tensor( soln )
    
    
    
    
    # def initConnectionMats( self, D=None, H=None  ):
    #     '''  '''
        
    #     # if not trainMatsDirectly:
        
    #     ''' Initialize them as parameters '''
    #     #---------------------------------------------------------------------------------
    #     self.W = nn.Parameter( torch.empty((self.networkDim, self.signalDim)), requires_grad=True )
    #     self.Ms = nn.Parameter( torch.empty( (self.networkDim, self.networkDim) ), requires_grad=True )
    #     self.Md = nn.Parameter( torch.empty( (self.networkDim, self.networkDim) ), requires_grad=True )


    #     '''  '''
    #     #---------------------------------------------------------------------------------
    #     W, Md, Ms = self.computeConnectionMats( D, H )

    
    
    #     ''' Assign '''
    #     #---------------------------------------------------------------------------------
    #     self.W = nn.Parameter( W, requires_grad=True )
    #     self.Md = nn.Parameter( Md, requires_grad=True )
    #     self.Ms = nn.Parameter( Ms, requires_grad=True )
        
        
    #     return W, Md, Ms
    
    
    


    
    #=====================================================================================
    ''' Computing Functions '''
    #=====================================================================================
    
    
    def computeConnectionMats( self, D=None, H=None, useQ2Also=False ):
        '''  '''
        
        
            
        
        #-------------------------------------------------------------------------------------
        ''' Get the decoder and history matrices '''
        #-------------------------------------------------------------------------------------
        if (D is None) or (H is None):
            if hasattr( self, 'D' )  and  (self.D in self.parameters()):
                D = self.D
                H = self.H 
            else:
                D, H = self.initDandH( )
        
        
        
        #---------------------------------------------------------------------------------
        ''' Compute the connection matrices & set as model parameters '''
        #---------------------------------------------------------------------------------
        
        
        #----------------------------
        ''' Weight values '''
        #----------------------------
        errW = self.errorWeight 
        hW = self.historyWeight 
        
        
        
        # if hasattr( self, 'retentionWeight' ):
        #     rW = self.retentionWeight

        # else: 
        #     rW = 0 

       


        #--------------------------------------------------
        DtD = D.T  @  D                         
        Ht_DtD_H = H.T  @  DtD  @  H
        
        HtH = H.T @ H
        
        
        if useQ2Also:
            HtHt = H.T  @  H.T
            HH = H  @  H 
            
            HtHt_DtD_HH =  HtHt @  DtD  @  HH
            HtHt_HH = HtHt @ HH
        #--------------------------------------------------

        
        
        
        
        
        
        #--------------------------------------------------
        ''' FEEDFORWARD Connections (stimulus) '''
        #--------------------------------------------------
        W = errW * D.T    
                     
        
        
        #--------------------------------------------------
        ''' RECURRENT Connections (state) '''
        #--------------------------------------------------
        Md = (errW * DtD)
        
        if self.processingTermWithD:
            Md = Md  +  (hW * Ht_DtD_H)
        else:
            Md = Md  +  (hW * HtH)
            
        # if self.addStimSpaceProcessTerm:
        #     Md = Md  +  (hW * Ht_DtD_H)
            
            

        
        
        if useQ2Also:
            if self.processingTermWithD:
                Md2 = hW * HtHt_DtD_HH
            else:
                Md2 = hW * HtHt_HH

        
        
        
        #--------------------------------------------------
        ''' DELAYED RECURRENT Connections (refState) '''
        #--------------------------------------------------
        if self.processingTermWithD:
            Ms = hW  *  (H.T @ DtD) 
        else:
            Ms = hW  *  H.T
            
        
        # if self.addStimSpaceProcessTerm:
        #     Md = Md  +  ( hW * (H.T @ DtD) )
            


        if useQ2Also:
            if self.processingTermWithD:
                Ms2 = hW  *  (HtHt  @  DtD)       
            else:
                Ms2 = hW  *  HtHt       
                
        
        
        
        
        W = torch.as_tensor( W, dtype=torch.float32 )   
        Md = torch.as_tensor( Md, dtype=torch.float32 ) 
        Ms = torch.as_tensor( Ms, dtype=torch.float32 )     
        
        
        
        if useQ2Also:
            return W, Md, Md2, Ms, Ms2
        else:
            return W, Md, Ms
    
    
    
    
    
    # def computeDandH( self, W=None, Ms=None ):
        
    #     #---------------------------------------------------------------------------------
    #     if W is None:
    #         W = self.W
    #     if Ms is None:
    #         Ms = self.Ms 
    #     #--------------------------------------------------------------------------------- 
        
        
    #     #---------------------------------------------------------------------------------
    #     errW = self.errorWeight 
    #     if errW != 0:
    #         D = (1/errW) * W.T                                          ##  W = errW * D.T 
    #     else:
    #         H = torch.zeros(  [ self.signalDim, self.networkDim ]  )    ## ( d, N )
    #     #---------------------------------------------------------------------------------
        
        
    #     #---------------------------------------------------------------------------------
    #     hW = self.historyWeight 
    #     if hW != 0:
    #         H = (1/hW) * Ms.T                                           ## Ms = hW * H.T
    #     else: 
    #         H = torch.zeros(  [ self.networkDim, self.networkDim ]  )   ## ( N, N )
    #     #---------------------------------------------------------------------------------
        
        
    #     return D, H 
    
    
    
    
    
    def LipschitzConstant( self ):
        '''  
        Computes the Lipschitz constant of the linear function 
            errWeight * J_e   +   effWeight * J_eff   +   hWeight * J_h +   fWeight * J_f 
        where 
            J_err = ||  x(t) - D*r(t)  ||_2^2
            J_eff = ||  r(t)  ||_2^2
            J_h = ||  r(t-1) - H*r(t)  ||_2^2
            J_f = ||  r(t) - A*r(t-1)  ||_2^2,
            
        which can be shown to be
            L =  || Md ||  +  || effW ||,
        where  
            Md = ( errWeight * D.T @ D )  +  ( hWeight * H.T @ H )  +  ( fWeight * np.eye(N) ). 
        '''
    
    
        if hasattr( self, 'Md2' ): 
            Md_tot = self.Md + self.Md2
        else:
            Md_tot = self.Md 
            
            
        L = torch.norm( Md_tot, p=2 )  +  abs( self.efficiencyWeight ) 
        # L = torch.norm( Md_tot, p=2 )  
    
    
    
        return L 

    





    # def updateDecoderAndHistoryMats( self ):
        
        
    #     errW = self.errorWeight 
    #     hW = self.historyWeight 
        
        
    #     self.D = (1/errW) * self.W.T                ##  W = errW * D.T   
    #     self.H = (1/hW) * self.Ms.T                 ##  Ms = hW * H.T              
        
        
    #     return self 


    
    #=====================================================================================
    ''' Performance Metrics '''
    #=====================================================================================
    
    # def lossFunc( self, x, r, r_tm1, D=None, H=None, r_tm2=None, converged=None ):
    def lossFunc( self, x, r, r_tm1, D=None, H=None, r_tm2=None ):
        '''  
            x       -   current stimulus                            ( d, 1 )
            r       -   current state                               ( N, 1 )
            r_tm1   -   current reference state (prev. stim)        ( N, 1 )
            D       -   decoder matrix                              ( d, N )
            H       -   reconstructor matrix                        ( N, N )
            r_tm2   -   reference state for stim before prev stim   ( N, 1 )
            convR   -   converged state (end of the encoding phase) ( N, 1 )
            
            
            
        Note that r_tm1 (reference state) is the *decayed* representation of the previous 
        stim, "snapshotted" at the end of it's retention period
        
        Also note the convR IS ASSUMED TO BE USED ONLY DURING TRAINING since backprop
        occurs for each stim only at the end of the delay period so r would be the decayed 
        state 
        
        
        '''
        
        
        refState2 = r_tm2
        
        
        ''' 1. Update D,H based on current W,Md,Ms ''' 
        #---------------------------------------------
        if (D is None) or (H is None):
            
            if hasattr( self, 'D' ):
                D, H = [ self.D, self.H ]
            # else:
            #     D, H = self.computeDandH( ) 

                
        
        ''' 2. Cost terms ''' 
        #---------------------------------------------
        # [ errLoss, hLoss, effLoss, sLoss ] = self.costTerms( x, r, r_tm1, D, H )
        [ errLoss, hLoss, effLoss, sLoss, eigLoss ] = self.costTerms( x, r, r_tm1, D, H, 
                                                                         # r_tm2=refState2, converged=converged )
                                                                         r_tm2=refState2  )
        
        
        
        ''' 3. Total cost ''' 
        #---------------------------------------------
        errW = self.errorWeight 
        effW = self.efficiencyWeight
        hW = self.historyWeight 
        sW = self.sparsityWeight
        # rW = self.retentionWeight

        lossVal = (errW * errLoss)  +  (hW * hLoss)  +  (effW * effLoss)  +  (sW * sLoss) 
    
        lossVal = lossVal + eigLoss
    
    
        return lossVal 

        
        
    
    
    # def currentCostTerms( self, currStim, currState, referenceState, D=None, H=None  ):
    # def costTerms( self, currStim, currState, referenceState, prevState, D=None, H=None  ):
    def costTerms( self, currStim, currState, referenceState, D=None, H=None, r_tm2=None ):
        ''' '''
        
        
        refState2 = r_tm2
        
        
        ''' 1. Update D,H based on current W,Md,Ms ''' 
        #=================================================================================
        if (D is None) or (H is None):
            
            if hasattr( self, 'D' ):
                D, H = [ self.D, self.H ]
            # else:
            #     D, H = self.computeDandH( ) 
            
        
        
        ''' 2. Compute the cost terms ''' 
        #=================================================================================
        
        
        #-------------------------------------------------
        ''' Error '''
        #-------------------------------------------------
        errLoss = torch.norm(  currStim - (D @ currState),  p=2  )**2          
        
        
        #-------------------------------------------------
        ''' History '''
        #-------------------------------------------------
        reconstructed = (H @ currState)
        
        
        reconErr = referenceState - reconstructed
        if self.processingTermWithD:
            reconErr = D @ reconErr        
        hLoss = torch.norm(  reconErr,  p=2  )**2  

        
        if refState2 is not None:
            reconstructed2 = (H@H @ currState)
            
            reconErr2 = refState2 - reconstructed2
            if self.processingTermWithD:
               reconErr2 = D @ reconErr2
            hLoss2 = torch.norm(  reconErr2,  p=2  )**2         ## History
            
            hLoss = hLoss + hLoss2
            
        
        #-------------------------------------------------
        ''' Efficiency (L2) '''
        #-------------------------------------------------
        effLoss = torch.norm( currState, p=2 )                                  
        
        
        #-------------------------------------------------
        ''' Sparsity (L1) '''
        #-------------------------------------------------
        sLoss = torch.norm( currState, p=1 )                                    
        

        #-------------------------------------------------
        ''' Stabiliity of H '''
        #-------------------------------------------------
        
        if self.requireStable:
            # eigVals = torch.linalg.eig( H )[0].detach().numpy()
            eigVals = torch.linalg.eig( H )[0]
            eigVals_real = torch.real( eigVals )
            eigVals_posReal = eigVals_real[  eigVals_real > 0  ]
            
            eigLoss = torch.sum(   eigVals_posReal   )
            
            # eigLoss = torch.sum(  torch.relu( eigVals_posReal )  )
            # 
            # eigLoss = 1000 * eigLoss 
            eigLoss = 1e5 * eigLoss 
        
        
        else:
            eigLoss = 0
        
        
        
        # trace = torch.trace( H )
        # # eigLoss = 100 * (-1) * trace
        # eigLoss = 100 * trace
        
        
        
        # rank = torch.linalg.matrix_rank( H )
        # eigLoss = H.shape[0] - rank
        
        # eigLoss = 100 * eigLoss
        
        
        #--------------------------------------------------------
        return errLoss, hLoss, effLoss, sLoss, eigLoss 
        #--------------------------------------------------------
     
    
    
    
        
    def KafashanLearning( self, currStim, currState, referenceState, connectionVars ):
        ''' 
            Implements a learning update to the connection matrices as described in 
        Algorithm 1 (Kafashan, et. al., 2017). Essentailly performs optimizer.step().
        
        
        *  M. Kafashan and S. Ching, ‘Recurrent networks with soft-thresholding nonlinearities
        for lightweight coding’, Neural Netw., vol. 94, pp. 212–219, Oct. 2017.
        '''
        
        #---------------------------------------------------
        prevA = connectionVars[ 'A' ]
        prevB = connectionVars[ 'B' ] 
        
        
        eps_a = connectionVars[ 'eps_a' ] 
        eps_b = connectionVars[ 'eps_b' ] 
        
        tau_w = connectionVars[ 'tau_w' ] 
        tau_ms = connectionVars[ 'tau_ms' ] 
        tau_md = connectionVars[ 'tau_md' ] 
        
        alpha = connectionVars[ 'alpha' ] 
        #---------------------------------------------------
        
        
        #---------------------------------------------------
        currStim = currStim.detach( )
        currState = currState.detach( )
        referenceState = referenceState.detach( )
        
        
        W = self.W.detach()
        Ms = self.Ms.detach()
        Md = self.Md.detach()
        #---------------------------------------------------
        
          
        ''' A and B '''
        #---------------------------------------------------  
        x_rT = currStim @ currState.T                        # ( d, N )
        A =  eps_a * prevA   +   (1-eps_a) * x_rT           # ( d, N )
        
        ref_rT = referenceState @ currState.T                # ( N, N )
        B =  eps_b * prevB   +   (1-eps_b) * ref_rT         # ( N, N )
        
        
        
        ''' Connection Matrices'''
        #---------------------------------------------------
        W = W  +  tau_w * (A.T - alpha * W)
        
        
        prevMs_ref_refT = Ms @ referenceState  @  referenceState.T
        # prevMs_ref = np.matmul( Ms, referenceState )
        # prevMs_ref_refT = np.matmul( prevMs_ref,  referenceState.T )
        Ms = Ms  +  tau_ms * (B.T - alpha * prevMs_ref_refT )
        
        
        prevW_A = W @ A 
        prevMs_B = Ms @ B 
        Md = Md  +  tau_md * (prevW_A  +  prevMs_B  -  alpha * Md)
        #---------------------------------------------------
        
        
        
        #---------------------------------------------------
        self.W = nn.Parameter( torch.tensor(W) )
        self.Ms = nn.Parameter( torch.tensor(Ms) )
        self.Md = nn.Parameter( torch.tensor(Md) )
        
        connectionVars[ 'A' ] = A
        connectionVars[ 'B' ] = B
        #---------------------------------------------------
        
        
        
        
        return self, connectionVars
        
        
      
        
      
        
      
        
#=========================================================================================
#=========================================================================================
#%%  proximal 
#=========================================================================================
#=========================================================================================


def proximal_l1( proxInput, stepSize, sWeight ):
# def proximal_muG( proxInput, stepSize, sWeight ):
    ''' 
        Computes the proximal for a given function and over input proxInput x:
            prox_g(x) = arg min, z  {   (1/2)*|| x - z ||_2^2   +   g(x)   }
    where we assume 
            g(x) = sWeight * ||  x  ||_1, 
    resulting in a soft-thresholding or "deadzone nonlinearity."
    '''
        
    
    
    if sWeight == 0:
        proxOutput = proxInput


    else:     
        threshold = stepSize * sWeight
        proxOutput = softThreshold( proxInput, threshold )
        
        
        
        ''' Compare to the method Kafashan used ''' 
        #-----------------------------------------------------------------------------
        N = proxInput.shape[0]
        zeros = torch.zeros( N,1 )
        proxOutput2 = torch.maximum(zeros, proxInput - threshold)  -  torch.maximum(zeros, -proxInput - threshold)
        
        if torch.all( proxOutput != proxOutput2 ):
            print( )
            print( 'Different proxOutput from Kafashan' )
        
    
    
    return proxOutput
    
    
    

def softThreshold( inputVector, threshold ):

    #----------------------------------------------
    d1 = inputVector.shape[0]
    stOutput = torch.zeros( [ d1, 1 ] )        # assumes inputVector is a column vector 
    #----------------------------------------------


    ''' Threshold each element of the input '''
    #----------------------------------------------
    stPattern = [ ]
    
    
    # print( 'inputVector', inputVector )
    # print( 'inputVector.shape', inputVector.shape )
    
    
    for i in range( d1 ):
        
        ri = inputVector[ i ]
        
        # print( 'ri', ri )
        # print( 'threshold', threshold )
        
        if ri > threshold: 
            stOutput[ i ] = ri - threshold
            stPattern.append( '-' )
            
        elif abs(ri) <= threshold:
            stOutput[ i ] = 0
            stPattern.append( '0' )
            
        elif ri < (-1)*threshold:
            stOutput[ i ] = ri + threshold
            stPattern.append( '+' )
            
        else: 
            # print( 'ri', torch.round(ri,decimals=2) )
            # print( 'threshold', threshold )
            # print( 'stOutput[i]', torch.round(stOutput[i],decimals=2) ) 
            
            raise Exception( '[softThreshold] Error in thresholding operation' )
    #----------------------------------------------
    
    # print( 'Soft-thresholding pattern: ', stPattern )
    


    return stOutput




    
def softThresholdScalar( inputValue, threshold ):

    
    if inputValue > threshold: 
        stOutput = inputValue - threshold
        
    elif abs(inputValue) <= threshold:
        stOutput = 0
        
    elif inputValue < (-1)*threshold:
        stOutput = inputValue + threshold
        
    else: 
        
        raise Exception( '[softThreshold] Error in thresholding operation' )
    #----------------------------------------------
    
    


    return stOutput

    





def plotSoftThresholding( xPts, threshold ):

    # xPts = [  np.array([[x]]) for x in xPts if type(x) is not np.array  ]
    
    yPts = [  softThresholdScalar(x,threshold) for x in xPts  ]
    
    
    fig = plt.figure( )
    plt.scatter( xPts, yPts )


    slopes = [  (yPts[i]-yPts[i+1]) / (xPts[i]-xPts[i+1])  for i in range(len(xPts)-1)   ]
    print(   np.round( slopes, 2 )  )

    return fig 




#=========================================================================================
#=========================================================================================
#%% Data (stimuli)
#=========================================================================================
#=========================================================================================









class inputStimuli( Dataset ):
    
    
    def __init__( self, signalDim, nBetas, simOptions, stimScale=0.5 ):
        
    
        self.signalDim = signalDim
        self.nBetas = nBetas
        
        
        self.circleStims = simOptions['circleStims']
        self.stimScale = stimScale
        
        self.endWithZeroStim = simOptions['endWithZeroStim']
        
        
        
        
        #------------------------------------------
        seedNum = 0
        seedNum = 10
        
        torch.manual_seed( seedNum )
        random.seed( seedNum )
        np.random.seed( seedNum )
        #------------------------------------------
        
        
        
        
        
        if (self.circleStims) and (self.signalDim != 2):
            print( '\n[inputStimuli] Given signalDim must be 2 for circle stims. Updating... d=2 ' )
            self.signalDim = 2
        
        # if self.circleStims:
        #     self.stimMat, self.thetas = self.generateStimMat( nBetas )
        # else: 
        self.stimMat = self.generateStimMat( nBetas )

        
        
        
        
        
        
        
        
        
    
    def generateStimMat( self, nStims ):
        '''  ''' 
        
        if self.circleStims:
            
            thetas = torch.rand( size=(1,nStims) )  *  360      
            # thetas = torch.tensor(   [  int(x) for x in thetas[0]  ]   )
            
            thetas = torch.deg2rad( thetas ) 
            # thetas = torch.tensor(   [  int(x) for x in thetas[0]  ]   )

            
            if self.endWithZeroStim:
                thetas[ -1 ] = 0
            
            stimMat = torch.cat(   [ torch.cos(thetas), torch.sin(thetas) ],  dim=0   )
            
            
            if hasattr( self, 'thetas' ):
                if self.endWithZeroStim:
                    self.thetas = torch.cat(   [ self.thetas[0:-1], thetas ],  dim=0   )
                else:
                    self.thetas = torch.cat(   [ self.thetas, thetas ],  dim=0   )
            else: 
                self.thetas = thetas
                
                
            
        else: 
            stimMat = torch.normal( 0, 1, size=(self.signalDim, nStims), dtype=torch.float32 )
            stimMat = stimMat * self.stimScale
            
            if self.endWithZeroStim:
                stimMat[ :, -1 ] = torch.zeros( stimMat[ :, -1 ].shape )
        
        
        return stimMat
        
    
    
    # def trueRadiansToXY( self, thetas ):
    #     '''
    #         Assumes given theta (angle) vector is in radians for degrees in [0,360), i.e., 
    #     radian values [0, 2*pi)  which is approximately [0, 6.283).                                                         
    #     '''
        
        
    #     radMax = torch.max( thetas ) 
    #     if radMax > (math.pi * 2):
    #         raise Exception( 'At least one theta value is greater than 360 degrees' )
        
    #     radMin = torch.min( thetas ) 
    #     if radMin < 0:
    #         raise Exception( 'At least one theta value is less than 0 degrees' )



    #     coors = torch.cat(   [ torch.cos(thetas), torch.sin(thetas) ],  dim=0   )

        


    
    
    def getStim( self, stimInd ):    
        return self.stimMat[ :, stimInd ]
    
    
    
    
    def addStims( self, nNewStims ):
        '''   '''
        
        if nNewStims == 0:
            return self 
        
        
        
        elif nNewStims > 0:    
            
            if self.endWithZeroStim:
                newStims = self.generateStimMat( nNewStims+1 )
                self.stimMat = torch.cat(  ( self.stimMat[ :, 0:-1 ], newStims ),  dim=1  ) 
                
            else: 
                newStims = self.generateStimMat( nNewStims )
                self.stimMat = torch.cat(  ( self.stimMat, newStims ),  dim=1  ) 
        
        
        
        elif nNewStims < 0:
            
            if abs(nNewStims) >= self.nBetas:
                raise Exception( '[inputStimuli.addStims] Number of stims to remove is >= nBetas! ' )
                
            else: 
                
                self.stimMat = self.stimMat[ :, 0:nNewStims ]
                
                if self.endWithZeroStim:
                    self.stimMat[ :, -1 ] = torch.zeros( self.stimMat[ :, -1 ].shape )
                
        
        self.nBetas = self.nBetas + nNewStims
        
        
        return self
    
        
    
    def plotStims( self ):
        
        if self.circleStims:
            
            fig, ax = plt.subplots( )
            
            t = np.linspace( 0, 359, 360 )          
            plt.plot(  np.cos(t), np.sin(t), linewidth=0.1, c='k'  ) 
        
            plt.scatter( self.stimMat[0,:], self.stimMat[1,:], c='r', linewidth=0.1 )
        
            ax.set_aspect( 1 )
            
        
        return 
     
        
     
    def getThetas( self, circlePts=None ):
         
        if circlePts is None:
            circlePts = self.stimMat
        elif circlePts.shape[0] != 2:
            raise Exception( 'Cannot compute thetas since given circlePts is not 2D.' )
             
            
        x = circlePts[ 0, : ]      ## cos( theta )  =  adj / hyp
        y = circlePts[ 1, : ]      ## sin( theta )  =  opp / hyp
        ## for unit circle (assumed), radius = hyp = 1
         
        thetas = x.acos()      
         
        return thetas
    
    
    
    
    def makeNoise( self, SNR=20 ):
        
        
        # ## Power of the input signal
        # squaredAvg = (1/self.nStims)   *   np.sum( self.stimMat**2, dim=0 )
        # rms =  math.sqrt( squaredAvg )                                   ## root mean square 
        
        
        stimNoise = np.random.normal(  0,  0.1,  [ self.signalDim, self.nStims ]  )
        self.stimNoise = stimNoise

        return stimNoise
    
    
    
    
    
# #-----------------------------------------------------------------------------------------
# ''' Dataloader '''
# #-----------------------------------------------------------------------------------------

# stimulusData = inputStimuli( d, nEpochs )






#=========================================================================================
#%% Connections from D,H
#=========================================================================================


        
def initModelParameters_( model, D=None, H=None ):
    '''  '''
    
    #-------------------------------------------------------------------------------------
    ''' Get the decoder and history matrices '''
    #-------------------------------------------------------------------------------------
    if D is None:
        D = fullRankMatrix( model.signalDim, model.networkDim )
        
    if H is None:
        H = fullRankMatrix( model.networkDim, model.networkDim )
    
    

    #-------------------------------------------------------------------------------------
    ''' Compute the connection matrices & set as model parameters '''
    #-------------------------------------------------------------------------------------
    errW = model.errorWeight 
    hW = model.historyWeight 
    
    
    W = errW * D.T                                  ## FEEDFORWARD Connections   
    W = torch.from_numpy( W )
    model.W = nn.Parameter( W , requires_grad=True)
                        
    Md = errW * (D.T @ D)   +   hW * (H.T @ H)      ## RECURRENT Connections 
    Md = torch.from_numpy( Md )
    model.Md = nn.Parameter( Md , requires_grad=True)
    
    Ms = hW * H.T                                   ## DELAYED RECURRENT Connections 
    Ms = torch.from_numpy( Ms )
    model.Ms = nn.Parameter( Ms , requires_grad=True)
     
    
    
    return model 
    





def fullRankMatrix( dim1, dim2, precision=100, minM=None, maxM=None, requireStable=False ):
    
    #----------------------------
    attemptCount = 0
    maxAttempts = 10
    
    if minM is None:
        minM = precision * -10          # minimum element in M
    if maxM is None:
        maxM = precision * 10           # max element in M
    #----------------------------
    
    
    
    while attemptCount < maxAttempts:
    
        M_list = [  random.randrange(minM, maxM)/precision for k in range(dim1*dim2)  ]
        M = np.array(M_list).reshape([dim1, dim2])
        
        attemptCount = 1
        
        #----------------------------

        if np.linalg.matrix_rank( M ) == min( dim1, dim2 ):
            return M
        
        elif attemptCount == maxAttempts:
            # print( '[fullRankMatrix] Unsuccessful after {} attempts', maxAttempts )
            raise Exception( '[fullRankMatrix] Unsuccessful after {} attempts', maxAttempts )
    
     






def matrixFromEigvals( desiredEigvals, nMats=1, array=True, JordanForm=False, diagonal=False ):
    '''  '''
    
    
    
    # #=====================================================================================
    # import scipy.linalg as la 
    
    # n = len( desiredEigvals )
    # s = np.diag( desiredEigvals )
    # q, _ = la.qr(  np.random.rand( n, n )  )
    # semidef = q.T @ s @ q
    # print( np.linalg.eigvalsh( semidef ) )
    # #=====================================================================================
    
    
    
    
    
    #=====================================================================================    
    eigvalShape = np.shape( desiredEigvals )
    
    if (len(eigvalShape) == 1) or (eigvalShape[1] == 1) :
        N = eigvalShape[0]
        desiredEigvals = desiredEigvals.reshape( [1,N] )
        # print( 'desiredEigvals',desiredEigvals )
    else: 
        N = eigvalShape[1]
    #=====================================================================================    
    
    
    
    M = np.round( np.random.rand(N,N), 2 )
    offDiag = np.triu( M, k=1 )                 # random upper triangular mat
    
    jordanDiag = np.diagflat(  np.ones([ N-1, 1 ]),  1  )
    
    
    
    
    ''' Generate the matrix/matrices '''
    #==========================================================================
    matrixArray = np.zeros( [nMats, N, N] )
    
    
    for matInd in range( nMats ):
        
        ''' 1. The eigvals on the diagonal '''
        #----------------------------------------------------------------------
        currDesiredEigvals = desiredEigvals[ matInd ]        
        # print('shape(currDesiredEigvals)',  shape(currDesiredEigvals) )
        sortedEigvals = True
        if sortedEigvals:
            # currDesiredEigvals = sorted(currDesiredEigvals)
            currDesiredEigvals = np.array(  sorted(currDesiredEigvals)  )
            desiredEigvals[ matInd ] = currDesiredEigvals
            
        D = np.diagflat( currDesiredEigvals )
        
        currEigvalNorm = np.linalg.norm( currDesiredEigvals )
        #----------------------------------------------------------------------
        
    
        
        
        #======================================================================
        #======================================================================
        # print( diagonal )
        if diagonal:
            
            A = D
            
            
            
            
        else: 
        
            ''' 2a. If singular, use Jordan Form'''
            ## Use Jordan form to prevent an identity matrix simply scaled by the eigval (normal matrix)
            #--------------------------------------------------------------
            
            currDesiredEigvals_normed = currDesiredEigvals / np.max(currDesiredEigvals)
            ones = np.ones( np.shape(currDesiredEigvals) )
            singular = np.all( currDesiredEigvals_normed == ones )
            
            
            if singular or JordanForm:
                # A = D + offDiag
                A = D + jordanDiag
            #--------------------------------------------------------------
            
            
            else: 
            
                ''' 2b. Randomly generate an invertible V (eigenvectors) '''
                #----------------------------------------------------------------------
                # else:
                detV = 0
                V_count = 0
                
                while detV == 0:
                    
                    
                    # V = np.random.normal( size=[N,N] )
                    V = np.random.rand( N, N )
                    # V = np.round(  np.random.rand(N,N) * 10,  2  )
                    # V = np.round( V, precision_round ) 
                                        
                    
                    V, _ = np.linalg.qr(  V  )
                    # V = np.round( V, 2 ) 
                    # V = np.round( np.random.normal(size=[N,N]), precision_round ) 
                    
                    detV = np.linalg.det(V)
                    # print( 'detV', detV )
                    V_count = V_count + 1
                    if V_count >= 10:                # prevent while loop from iterating forever 
                        print( '[eigvalsToMatrix] Couldn''t generate in invertible matrix V' )
                        break 
                #----------------------------------------------------------------------
                
                
                ''' 3b. The similarity transformation:  A  =  V * D * V_inv  '''
                #----------------------------------------------------------------------
                # V_inv = pinv(V)
                
            
                #************************
                V = V * currEigvalNorm
                #************************
                
                V_inv = np.linalg.inv(V)
                
                # [ checkVal, V_inv ] = invertibleCheck( V, checkType='id' )
                # if not checkVal:
                #     print( '[eigvalsToMatrix] Matrix V was not actually invertible' )
                
                
                # term1 = inv(V) @ D
                term1 = V_inv @ D
                A = term1 @ V
                
               
                A = np.round( A, 3 )
            #----------------------------------------------------------------------
            
            
        #======================================================================
        #======================================================================
        
        
        
        matrixArray[ matInd ] = A
        
        
        
        
        ''' 4. Check correct '''
        #----------------------------------------------------------------------
        actualEigvals = np.linalg.eig(A)[0]
        actualEigvals.sort()
        
        
        
        currDesiredEigvals.sort()
        eigvalBoolean = np.array([ np.isclose(actualEigvals[k],currDesiredEigvals[k],atol=1e-1) for k in range(N) ])
        
        
        if not np.all( eigvalBoolean ):
            # print( 'Generated eigvals:', actualEigvals )
            raise Exception('[eigvalsToMatrix]: The generated matrix did not have the desired eigenvalues')  
        
        
    #==========================================================================
    
    
    
    if not array:
        matrixArray = list( matrixArray )




    
    return matrixArray





def computeNullspace( A, atol=1e-13, rtol=0 ):
    ## https://stackoverflow.com/questions/49852455/how-to-find-the-null-space-of-a-matrix-in-python-using-numpy
    ## From SciPy Cookbook
    
    
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    
    return ns





#=========================================================================================
#%% weightCombos
#=========================================================================================



def makeWeightComboList( errWList=[0], effWList=[0], sWList=[0], hWList=[0], rWList=[0], fWList=[0], pWList=[0], roundingNum=2 ):
    ''' Iterate throught the lists to make all possible combos of given weights '''
    
    weightCombos = [ ]
    
    
    for errW in errWList:    
        for effW in effWList:
            for sW in sWList:
                for hW in hWList:
                    for rW in rWList:
                        for fW in fWList:
                            for pW in pWList:
                        
                                combo = makeWeightCombo( errW, effW, sW, hW, rW, fW, pW, roundingNum=roundingNum )
                                weightCombos.append( combo )
                
    return weightCombos





def shortenWeightCombo( weightCombo, roundingNum=2 ):
    '''  '''
    
    [ weights, weightDict ] = weightComboInfo( weightCombo )
    
    wc = ''
    
    #----------------------------------------------------------------------
    for key in list( weightDict.keys() ):
        
        weight = findNumInLabelledStr( weightCombo, key ) 
        if weight != 0:
            weightRound = np.round( weight, roundingNum )
            wc = wc + key + str(weightRound) + '_'
    #----------------------------------------------------------------------
    
    
    if wc[-1] == '_':
        wc = wc[0:-1]
        
    
    return wc





def makeWeightCombo( errWeight, effWeight, sWeight, hWeight, rWeight, fWeight, pWeight, roundingNum=2 ):
    ''' Make a string representation of the given weight combination (ignoring zeros) '''


    wc = ''

    #----------------------------------------------------------------------
    errRound = np.round( errWeight, roundingNum )       # Error        
    wc = wc + 'err' + str(errRound) + '_'
    
    effRound = np.round( effWeight, roundingNum )       # Efficiency        
    wc = wc + 'eff' + str(effRound) + '_'
    
    sRound = np.round( sWeight, roundingNum )           # Sparsity        
    wc = wc + 's' + str(sRound) + '_'
    
    hRound = np.round( hWeight, roundingNum )           # History        
    wc = wc + 'h' + str(hRound) + '_'
    
    rRound = np.round( rWeight, roundingNum )           # Retention        
    wc = wc + 'r' + str(rRound) + '_'
    
    fRound = np.round( fWeight, roundingNum )           # Processing        
    wc = wc + 'f' + str(fRound) + '_'
    
    pRound = np.round( pWeight, roundingNum )           # Processing        
    wc = wc + 'p' + str(pRound) + '_'
    #----------------------------------------------------------------------
        
        
    if wc[-1] == '_':
        wc = wc[0:-1]
        
        
    return wc

        



def weightComboInfo( weightCombo ):
    ''' 
    Grabs the cost weights from weightCombo of the form
        'err{}_eff{}_s{}_h{}_r{}_f{}_p{}'
    where the {} represent the corresponding error, efficiency, sparsity, history, 
    retention, frugality, and processing weights. 
    '''
    
    labelStrs = findLabelsInStr( weightCombo )
    # labelStrs = [ 'err', 'eff', 's', 'h', 'r', 'f', 'p' ]
    
    weightDict = { }
    weights = [ ]
    
    for label in labelStrs:
        # print(label)
        weight = findNumInLabelledStr( weightCombo, label ) 
        weightDict[ label ] = weight
        
        weights.append( weight )
    
    
    # errW = findNumInLabelledStr( weightCombo, labelStr='err' ) 
    # effW = findNumInLabelledStr( weightCombo, labelStr='eff' ) 
    # sW = findNumInLabelledStr( weightCombo, labelStr='s' ) 
    # hW = findNumInLabelledStr( weightCombo, labelStr='h' ) 
    # rW = findNumInLabelledStr( weightCombo, labelStr='r' ) 
    # fW = findNumInLabelledStr( weightCombo, labelStr='f' ) 
    # pW = findNumInLabelledStr( weightCombo, labelStr='p' ) 

    # return errW, effW, sW, hW, rW, fW, pW

    return weights, weightDict





def findLabelsInStr( origStr, seperator='_' ):
    ''' Assumes origStr takes the (repeated) format:
            {labelStr}{value}{seperator} 
    '''
    


    ''' Locate the seperators '''
    ##------------------------------------------------------------------------------------
    # strLen = len(origStr)
    # sepInds = [ i for i in range(strLen) if origStr.startswith(seperator, i) ]  
    subStrs = origStr.split( sep=seperator )
    
    
    ''' Locate the labels '''
    ##------------------------------------------------------------------------------------
    labels = [ ]
    
    for subStr in subStrs:
        label = ''
        for i in subStr:
            if i.isnumeric():
                break
            label = label + i
            # if not i.isnumeric():
            #     label = label + i
            
                
        labels.append( label )
    
    
    return labels





def findNumInLabelledStr( origStr, labelStr, seperator='_' ):
    ''' 
        Assumes the original string is of the (repeated) format:
            {labelStr}{value}{seperator} 
    '''
    
    
    ''' Locate the seperators '''
    ##------------------------------------------------------------------------------------
    strLen = len(origStr)
    sepInds = [ i for i in range(strLen) if origStr.startswith(seperator, i) ] 
    
    
    ''' Find the label '''
    ##------------------------------------------------------------------------------------    
    subStrs = origStr.split( sep=seperator )
    
    for subStr in subStrs:
        
        if labelStr in subStr:
            currLabel = ''
            for i in subStr:
                if i.isnumeric():
                    break
                currLabel = currLabel + i
            
            
            if currLabel == labelStr:
                for i in range(len(subStr)):
                    if subStr[i].isnumeric():
                        num = subStr[ i :: ]
                        
                        #--------------------
                        num = float(num)
                        
                        if num == int(num):
                            num = int(num)
                        #--------------------
                        
                        return num
                        # break
        
        
    return 0

                
    # labelInd = origStr.find( labelStr )                             # the label
    # if labelInd == -1:
    #     return 0
    
    
    # ''' Find the desired value '''
    # ##------------------------------------------------------------------------------------
    # # strSpilt = origStr.split( sep=seperator )

    
    # try:
    #     sepInd = [ ind for ind in sepInds if ind>labelInd ][0]      # the subsequent separator
    # except:
    #     sepInd = -1
    
    
    # if sepInd == -1:
    #     num = origStr[  (labelInd+len(labelStr))  :  :  ]          # the desired number
    # else: 
    #     num = origStr[  (labelInd+len(labelStr))  :  sepInd  ]     # the desired number
    
    
    # num = float( num )


    # return num 




def weightComboToReadable( weightCombo, shorten=True ):
    ''' '''
    
    if shorten:
        wc = shortenWeightCombo( weightCombo )
    else: 
        wc = weightCombo
        
        
    [ weights, weightDict ] = weightComboInfo( wc )
    
    
    labelStrs = list(  weightDict.keys()  )
    # labelStrs = findLabelsInStr( weightCombo )


    readableDict = {    'err' : r'$\lambda_{e}=$',
                        'eff' : r'$\lambda_{L_2}=$',
                        's' : r'$\lambda_{L_1}=$',
                        'h' : r'$\lambda_{h}=$',
                        'r' : r'$\lambda_{r}=$',
                        'p' : r'$\lambda_{p}=$',
                        'f' : r'$\lambda_{f}=$',
                       }
    
    
    ##--------------------------------------------------
    wcReadable = ''
    
    for label in labelStrs:
        numStr = str( weightDict[label] )
        fullStr = readableDict[label] + numStr

        wcReadable = wcReadable + fullStr
        
        if labelStrs[ -1 ] != label:
            wcReadable = wcReadable + ', '
    ##--------------------------------------------------

    
    return wcReadable





def weightCombo_ChangeSepVar( weightCombo, oldSeparator='_', newSeparator='-' ):
    
    wc = weightCombo.replace( oldSeparator, newSeparator )
    
    return wc 






# def makeWeightComboList( errWeightList, effWeightList, sWeightList, hWeightList, rWeightList, fWeightList=[None], roundingNum=2 ):
    
#     weightCombos = [ ]
    
    
#     for errW in errWeightList:    
#         for effW in effWeightList:
#             for sW in sWeightList:
#                 for hW in hWeightList:
#                     for rW in rWeightList:
#                         for fW in fWeightList:
                        
#                             combo = makeWeightCombo( errW, effW, sW, hW, rW, fW, roundingNum=roundingNum )
#                             weightCombos.append( combo )
                
#     return weightCombos





# def makeWeightCombo( errWeight, effWeight, sWeight, hWeight, rWeight, fWeight, roundingNum=2 ):


#     errRound = np.round( errWeight, roundingNum )       # Error
#     effRound = np.round( effWeight, roundingNum )       # Efficiency
#     sRound = np.round( sWeight, roundingNum )           # Sparsity
#     hRound = np.round( hWeight, roundingNum )           # History
#     rRound = np.round( rWeight, roundingNum )           # Retention
    
    
#     wc1 = 'err' + str(errRound) + '_eff' + str(effRound) + '_s' + str(sRound) + '_h' +  str(hRound) + '_r' +  str(rRound) 
    
    
#     # print( 'fWeight', fWeight )
    
    
    
#     if fWeight is None:
#         return wc1

#     else:
#         fRound = np.round( fWeight, roundingNum )           # Frugality
#         weightCombo = wc1 + '_f' + str(fRound)
#         return weightCombo

        



# def weightComboInfo( weightCombo ):
#     ''' 
#     Grabs the cost weights from weightCombo of the form
#         'err{}_eff{}_s{}_h{}_r{}_f{}'
#     where the {} represent the errWeight, effWeight, sWeight, hWeight, rWeight, and fWeights 
#     respectively
#     '''
    
#     #-----------------------------------
#     errInd = weightCombo.find( 'err' )
#     effInd = weightCombo.find( 'eff' )
#     sInd = weightCombo.find( '_s' )
#     hInd = weightCombo.find( '_h' )
#     rInd = weightCombo.find( '_r' )
    
#     fInd = weightCombo.find( '_f' )
#     #-----------------------------------
    
    
    
#     #-----------------------------------------------------
#     errWeight = weightCombo[  (errInd+3) : (effInd-1) ]
#     errWeight = float( errWeight )
    
#     effWeight = weightCombo[  (effInd+3) : sInd ]
#     effWeight = float( effWeight )
    
#     sWeight = weightCombo[  (sInd+2) : hInd ]
#     sWeight = float( sWeight )
    
#     hWeight = weightCombo[  (hInd+2) : rInd ]
#     hWeight = float( hWeight )
    
#     # rWeight = weightCombo[  (rInd+2) : hInd ]
#     # rWeight = float( sWeight )
#     #-----------------------------------------------------
    
    
    
#     #-----------------------------------------------------
#     if fInd == -1:
#         fWeight = None
#         rWeight = weightCombo[  (rInd+2) : : ]
        
        
#     else: 
#         rWeight = weightCombo[  (rInd+2) : fInd ]
        
#         fWeight = weightCombo[  (fInd+2) : : ]
#         fWeight = float( fWeight )
        
#     rWeight = float( rWeight )
#     #-----------------------------------------------------
    
    
    

#     return errWeight, effWeight, sWeight, hWeight, rWeight, fWeight




# def weightComboToReadable( weightCombo ):
    
#     [errWeight, effWeight, sWeight, hWeight, rWeight, fWeight] = weightComboInfo( weightCombo )
    
#     errW = r'$\lambda_{e}=$' + str(errWeight) 
#     effW = r'$\lambda_{L_2}=$' + str(effWeight) 
#     sW = r'$\lambda_{L_1}=$' + str(sWeight) 
#     hW = r'$\lambda_{h}=$' + str(hWeight) 
#     rW = r'$\lambda_{r}=$' + str(rWeight) 
    
    
#     wcReadable = errW + ', ' + effW + ', ' + sW + ', ' + hW+ ', ' + rW
    
#     if fWeight is not None:
#         fW = r'$\lambda_{f}=$' + str(fWeight) 
#         wcReadable = wcReadable + ', ' + fW
    
    
#     return wcReadable





# def weightCombo_ChangeSepVar( weightCombo, oldSeparator='_', newSeparator='-' ):
    
    
#     wc = weightCombo.replace( oldSeparator, newSeparator )
    
    
#     # # inds = [ ]
#     # wc = weightCombo
    
    
#     # while oldSeperator in wc:
#     #     ind = wc.index( oldSeparator )
#     #     wc[ ind ] = newSeparator
    
    
#     return wc 



#=========================================================================================
#%% stepSize
#=========================================================================================


# def LipschitzConstant( model ):
#     '''  
#     Computes the Lipschitz constant of the linear function 
#         errWeight * J_e   +   effWeight * J_eff   +   hWeight * J_h +   fWeight * J_f 
#     where 
#         J_err = ||  x(t) - D*r(t)  ||_2^2
#         J_eff = ||  r(t)  ||_2^2
#         J_h = ||  r(t-1) - H*r(t)  ||_2^2
#         J_f = ||  r(t) - A*r(t-1)  ||_2^2,
        
#     which can be shown to be
#         L =  || Md ||  +  || effW ||,
#     where  
#         Md = ( errWeight * D.T @ D )  +  ( hWeight * H.T @ H )  +  ( fWeight * np.eye(N) ). 
#     '''


#     # L = np.linalg.norm( model.Md, ord=2 )  +  abs( model.efficiencyWeight ) 
#     L = torch.norm( model.Md, p=2 )  +  abs( model.efficiencyWeight ) 


#     return L 





# def computeDandH( model ):
    
#     errW = model.errorWeight 
#     hW = model.historyWeight 
    
#     D = (1/errW) * model.W.T                ##  W = errW * D.T 
#     H = (1/hW) * model.Ms.T                 ## Ms = hW * H.T
    
    
#     return D, H 







#=========================================================================================
#%% Coor. <---> theta
#=========================================================================================



def coorsToTrueRad( x, y, radius=1 ):
    ''' 
        Due to the domain and range of arccos() and arcsin(), coordinates on a circle 
    can not be directly translated to their theta (angle) value (with 0 deg. being (1,0)). 
    That is, only for angles in the range [0,180] degrees or [0, pi] radians respectively, 
    does 
    
            theta = arccos(  cos(theta)  )          where x = cos(theta)
                  = arccos(  x  )
                  
            theta = arcsin(  sin(theta)  )          where y = sin(theta)
                  = arcsin(  y  )        
                  
    hold true. This function works with that knowledge to be able to transform any point 
    on an (assumed unit) circle to its corresponding radian theta value. To do the same for 
    degree value, see coorsToTrueDeg(). 
    '''
    
    
    if type( x ) is np.ndarray:
        x = torch.tensor( x )
    if type( y ) is np.ndarray:
        y = torch.tensor( y )
    
    nCoors = len(y)
    rads = torch.zeros( y.shape )  


    #---------------------------------------------------------------------------------
    ''' Adjust (x,y) coordinates if not the unit circle '''
    #---------------------------------------------------------------------------------
    if radius != 1:
        x = x / radius
        y = y / radius 



    # corrPairs = [   tuple( [x[i], y[i]] )   for i in range( nCoors )   ]
    
    
    #---------------------------------------------------------------------------------
    ''' Degrees in [0, 90) '''
    #---------------------------------------------------------------------------------
    # indsForLessThan180 = torch.where( y >= 0 )
    # rads[ indsForLessThan180 ] = np.arcsin(   y[indsForLessThan180]   )
    
    
    indsForDegBtwn0and90 = [  i for i in range(nCoors) if x[i] > 0 and y[i] >= 0  ]
    radsForDegBtwn0and90 = np.arcsin(   abs( y[indsForDegBtwn0and90] )   )
    
    currRads = radsForDegBtwn0and90
    rads[ indsForDegBtwn0and90 ] = currRads
    
    
    # print( )
    # print( rads )
    # print( np.round(  np.rad2deg(rads),  4  ) )
    
    
    
    #---------------------------------------------------------------------------------
    ''' Degrees in [90, 180] '''
    #---------------------------------------------------------------------------------
    indsForDegBtwn90and180 = [  i for i in range(nCoors) if x[i] <= 0 and y[i] >= 0  ]
    radsForDegBtwn90and180 = np.arcsin(   abs( x[indsForDegBtwn90and180] )   )
    
    currRads = radsForDegBtwn90and180  +  (math.pi/2)
    rads[ indsForDegBtwn90and180 ] = currRads
    
    # print( )
    # print( rads )
    # print( np.round(  np.rad2deg(rads),  4  ) )
    
    
    #---------------------------------------------------------------------------------
    ''' Degrees in (180, 270] '''
    #---------------------------------------------------------------------------------
    indsForDegBtwn180And270 = [  i for i in range(nCoors) if x[i] <= 0 and y[i] < 0  ]
    radsForDegBtwn180And270 = np.arcsin(   abs( y[indsForDegBtwn180And270] )   )
    
    # currRads = radsForDegBtwn180And270  +  (math.pi/2)
    currRads = radsForDegBtwn180And270  +  math.pi
    rads[ indsForDegBtwn180And270 ] = currRads
    
    # print( )
    # print(  np.round( x[indsForDegBtwn180And270], 4 )  )
    # print(  np.round( y[indsForDegBtwn180And270], 4 )  )
    # print(  '\t', np.round( radsForDegBtwn180And270, 4 )  )
    # print(  '\t rads:', np.round( currRads, 4 )  )
    # print(  '\t degs:', np.round( np.rad2deg(currRads), 4 )  )
    # print( rads )
    # print( np.round(  np.rad2deg(rads),  4  ) )
    
    
    
    #---------------------------------------------------------------------------------
    ''' Degrees in (270, 360) '''
    #---------------------------------------------------------------------------------
    indsForDegGreaterThan270 = [  i for i in range(nCoors) if x[i] > 0 and y[i] < 0  ]
    radsForDegGreaterThan270 = np.arcsin(   abs( x[indsForDegGreaterThan270] )   )
    
    currRads = radsForDegGreaterThan270  +  math.pi*(3/2)
    rads[ indsForDegGreaterThan270 ] = currRads

    # print( )
    # print(  np.round( x[indsForDegGreaterThan270], 4 )  )
    # print(  np.round( y[indsForDegGreaterThan270], 4 )  )
    # print(  '\t', np.round( radsForDegGreaterThan270, 4 )  )
    # print(  '\t rads:', np.round( currRads, 4 )  )
    # print(  '\t degs:', np.round( np.rad2deg(currRads), 4 )  )
    # print( np.round(rads,4) )
    # print( np.round(  np.rad2deg(rads),  4  ) )


    return rads







def coorsToTrueDeg( x, y ):
    ''' 
        Due to the domain and range of arccos() and arcsin(), coordinates on a circle 
    can not be directly translated to their theta (angle) value (with 0 being (1,0)). 
    That is, only for angles in the range [0,180] degrees or [0, pi] radians respectively, 
    
            theta = arccos( cos(theta) )        x-coor.
                  = arccos( x )
                  
            theta = arcsin( sin(theta) )        y-coor.
                  = arcsin( y )        y-coor.
                  
    hold true. This function works with that knowledge to be able to transform any point 
    a (assumed unit) circle to its corresponding degree theta value. To do the same for 
    theta value, see coorsToTrueRad(). 
    '''
    
    
    rads = coorsToTrueRad( x, y )
    degs = torch.rad2deg( rads )
    
    
    return degs 



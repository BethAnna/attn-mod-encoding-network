# attn-mod-encoding-network
Neural network trained for the attention-modulated encoding and retention of stimuli, whose functionally-optimal trajactory is determined by proximal gradient method (PGM). 

**Further documentation and re-upload of codebase in progress to be more open-source friendly and accessible.**

SCRIPT FILES: 
PGM_main.py - The main script to run, initiating the base simulation.
PGM_model.py - Contains the network class definition as well as auxiliary functions.
PGM_trainAndTest.py - Algorithm and auxiliary functions for training the network connections as well as time-step simulation via proximal gradient.
PGM_analysis.py - Functions to characterize resulting performance, dynamics, etc. 
retenAnalysis.py - Functions specifically characterizing the network's ability to retain input stimuli across multiple event horizons. 
retenAnalysisFuncs.py - Auxiliary functions for retenAnalysis.py
varyAlpha.py - Secondary "main" script to simulate and assess the impact of the attentional parameter specifically. 

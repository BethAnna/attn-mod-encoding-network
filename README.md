# attn-mod-encoding-network
Trains and simulates an attention-modulated and biologically inspired neural network that encodes and retains afferent stimuli within its dynamics. The network's dynamical trajectory is determined by proximal gradient method (PGM), resulting in real-time, functionally-optimal evolution. Here "functionally optimal" is quantified via an objective function containing encoding, decoding, and retention measures, embedding cognitive mnemonic principles. By default, the network connectivity matrices are trained via Adam optimization over the objective function and a Hurwitz constraint. For more information, the reader is referred to the author's paper [1]. \
\
**Further documentation and re-upload of codebase in progress to be more open-source friendly and accessible.** \


# SCRIPT FILES:
`PGM_main.py` - The main script to run, initiating the base simulation.\
`PGM_model.py` - Contains the network class definition as well as auxiliary functions.\
`PGM_trainAndTest.py` - Algorithm and auxiliary functions for training the network connections as well as time-step simulation via proximal gradient.\
`PGM_analysis.py` - Functions to characterize resulting performance, dynamics, etc.\
`retenAnalysis.py` - Functions specifically characterizing the network's ability to retain input stimuli across multiple event horizons.\
`retenAnalysisFuncs.py` - Auxiliary functions for `retenAnalysis.py`.\
`varyAlpha.py` - Secondary "main" script to simulate and assess the impact of the attentional parameter specifically.\
\
[1] B. Jones, L. Snyder, and S. Ching, ‘Heterogeneous forgetting rates and greedy allocation in slot-based memory networks promotes signal retention’, Neural Comput., vol. 36, no. 5, pp. 1022–1040, Apr. 2024. https://doi.org/10.1162/neco_a_01655

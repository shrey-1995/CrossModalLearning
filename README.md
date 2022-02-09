# CrossModalLearning

## Introduction
The recent work Text2Shape(#http://text2shape.stanford.edu/) and Y{\textsuperscript{2}}Seq2Seq bridges the gap between the Natural language description and 3D shapes to learn a cross modal representation. For the work of shape-to-text(S2T) retrieval and text-to-shape(T2S) retrieval, to further learn the correlation between the 3D shapes and text description we will use both cross modal reconstruction and metric learning approach, and produce a joint representation that captures the many-to-many relations between language and physical properties.

[Video](#https://www.youtube.com/watch?v=fxWC9Ubk4to&feature=youtu.be)

Important considerations:
* The whole code has been written to run on Google Colab to use GPUs and speed-up computations. You might need to do minor changes to adapt it for a different environment.
* I really recommend using GPUs. Otherwise, adversarial attacks generation and model predictions become really slow.
* Using these scripts, all results can be reproduced. Also data has been provided to avoid repeating the whole process and help future researchers.

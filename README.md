# EEG Tux Racer

Important things to note:
* ```eegtuxracer.py``` is the main file you should be running
* Line 71 of ```eegtuxracer.py``` asks you to put the file name for a tensorflow lite model. This model can be obtained by using the Edge Impulse
  platform to train a model using your EEG signals to classify left and right movement inputs
* When building your model/pipeline in Edge Impulse you'll want/need to add a processing block. The code in this repository expects that your data is
  processed using the spectral analysis processing module (which can be selected by choosing the 'Show all blocks anyway' option at the bottom). Ensure that
  the options you select in Edge Impulse and the relevant variables in the features function in ```eegtuxracer.py``` are the same.
* Be aware that the code is mostly untested as a whole as I was only really able to test a few things independently, so it's pretty likely things will break
  (although I will try as much as I can to resolve them)

P.S. The code used here was adapted and taken from https://github.com/YYK2007/VirtualScrollableKeyboard and https://github.com/edgeimpulse/processing-blocks/tree/master/spectral_analysis


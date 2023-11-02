# DPL

[Challenge](https://istethoscope.peterjbentley.com/heartchallenge/index.html)

Data from: [Dataset main](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds) + [Secondary Class organised](https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound)

<h4 align="left">💼 About the Data:</h4>   
 
Data has been gathered from two sources: **(A) from the general public via the iStethoscope Pro iPhone app, provided in Dataset A, and (B) from a clinic trial in hospitals using the digital stethoscope DigiScope, provided in Dataset B.**

**CHALLENGE 1 - Heart Sound Segmentation**

The first challenge is to produce a method that can locate S1(lub) and S2(dub) sounds within audio data, segmenting the Normal audio files in both datasets. To enable your machine learning method to learn we provide the exact location of S1 and S2 sounds for some of the audio files. You need to use them to identify and locate the S1 and S2 sounds of all the heartbeats in the unlabelled group. The locations of sounds are measured in audio samples for better precision. Your method must use the same unit.

**CHALLENGE 2 - Heart Sound Classification**

The task is to produce a method that can classify real heart audio (also known as “beat classification”) into one of four categories for Dataset A:

* Normal
* Murmur
* Extra Heart Sound
* Artifact
    
and three classes for Dataset B:

* Normal
* Murmur
* Extrasystole
  
You may tackle either or both of these challenges. If you can solve the first challenge, the second will be considerably easier! The winner of each challenge will be the method best able to segment and/or classify two sets of unlabelled data into the correct categories after training on both datasets provided below. The creator of the winning method will receive a WiFi 32Gb iPad as the prize, awarded at a workshop at AISTATS 2012.

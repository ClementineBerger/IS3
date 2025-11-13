% Example script that illustrates how to generate a multi-channel signal
% that can be played through a given 2D loudspeaker array/ring using the 
% Mixed-order Ambisonics (MOA) signals saved in the directory MOA31ch\. 

clear % clear all variables from workspace
clc % clear command window

%%% -----------------------------------------------------------------------
%%% Initialization
%%% -----------------------------------------------------------------------

M2 = 7;%2D HOA order - must be <= 7 (Note: requires horizontal ring of k >= 2*M2+1 loudspeakers)

[LsPos,Nlsp] = Lsp16ch;% Creates Nlsp x 2 matrix that defines the applied loudspeaker array layout. 
                       % Here, a 16 channel 2D array/ring is considered as an example.

[D,MOAch] = decMatrix(LsPos, 0, M2);% derive basic decoding matrix D (and utilized MOA channels)

Nbits = 32;% number of Bits per sample for writing multi-channel loudspeaker wav-file

%%% -----------------------------------------------------------------------
%%% Main processing
%%% -----------------------------------------------------------------------

sMOA = readMOAfileDialog(MOAch);% Read MOA sound file required for given decoding matrix D (opens Dialog)

sLsp = MOA2Lsp(sMOA,D);% Decode MOA signal into multi-channel loudspeaker signal

saveLspFileDialog(sLsp)% write decoded loudspeaker signal to file (opens Dialog)

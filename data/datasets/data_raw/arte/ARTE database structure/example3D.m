% Example script that illustrates how to generate a multi-channel signal
% that can be played through a given 3D loudspeaker array using the 
% Mixed-order Ambisonics (MOA) signals saved in the directory MOA31ch\. 
%
clear % clear all variables from workspace
clc % clear command window

%%% -----------------------------------------------------------------------
%%% Initialization
%%% -----------------------------------------------------------------------

M2 = 4;%2D HOA order - must be <= 7 (Note: requires horizontal ring of k >= 2*M2+1 loudspeakers)
M3 = 4;%3D HOA order - must be <= 4 (Note: requires regular array of k >= (M3+1)^2 loudspeakers)

[LsPos,Nlsp] = LspNAL41ch;% Creates Nlsp x 2 matrix that defines the applied loudspeaker array layout. 
                          % Here, a 41 channel 3D array is considered as an example.

[D,MOAch] = decMatrix(LsPos, M3, M2);% derive basic decoding matrix D (and utilized MOA channels)

Nbits = 32;% number of Bits per sample for writing multi-channel loudspeaker wav-file

%%% -----------------------------------------------------------------------
%%% Main processing
%%% -----------------------------------------------------------------------

sMOA = readMOAfileDialog(MOAch);% Read MOA sound file required for given decoding matrix D

sLsp = MOA2Lsp(sMOA,D);% Decode MOA signal into multi-channel loudspeaker signal

saveLspFileDialog(sLsp)% write decoded loudspeaker signal to file

% function [sMOA,fs] = readMOAfileDialog(MOAch,range)
%
% Reads a multi-channel MOA file.
%
% Inputs:
% MOAch   Indeces of the required MOA channels. This variable is given by
%         the function decMatrix.m. If not provided then all 31 channels
%         are used.
% range   Range of samples that should be read from MOA wav-file. If not
%         specified then the enire MOA signal is read.
%         If size 1x1: Reads the first "range" samples of the wav-file.
%         If size 1x2: Defines start and end sample.
%
% Outputs:
% sMOA    The multi-channel MOA sound file that is read from disk
% fs      sampling frequency of MOA sound file (Hz)

function [sMOA,fs] = readMOAfileDialog(MOAch,range)

defaultPath = [pwd '\MOA31ch\'];
[nameMOA,pathMOA] = uigetfile('*.wav','Select a MOA wav-file...',defaultPath);% open dialog for selecting MOA sound file 

if exist('range','var') % if range is defined as input variable
    if max(size(range)) == 1
        range = [1 range];
    end
    disp(['Reading ' num2str(range(2)-range(1)+1) ' samples of MOA sound file...'])
    [sMOA,fs] = audioread(fullfile(pathMOA,nameMOA),range);% read part of 31-channel MOA sound file
else
    disp('Reading entire MOA sound file...')
    [sMOA,fs] = audioread(fullfile(pathMOA,nameMOA));% read entire 31-channel MOA sound file
end

if exist('MOAch','var')
    sMOA = sMOA(:,MOAch);% only use required subset of MOA channels
end
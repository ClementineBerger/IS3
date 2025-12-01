% function [LsPos,N] = LspNAL41ch
% 
% Creates (Nx2) matrix of N loudspeaker directions [azimuth elevation] 
% in radians for a given loudspeaker array. Here, the 41 loudspeaker 3D 
% array available at the Australian Hearing Hub is used as an example.
%
% Note:
% Azimuth angle: 0 rad <= azimuth < 2*pi (anti-clockwise -> positive values to the left)
% Elevation angle: -pi/2 <= elevation <= pi/2
% 
% Outputs:
% LsPos   Directions [azimuth elevation] in radians for all loudspeakers in
%         the array (size Nx2)
% N       Number of loudspeakers in array (here 41)

function [LsPos,N] = LspNAL41ch

LsPos = [[[0 337.5:-22.5:22.5]' zeros(16,1)]; ... % horizontal ring of 16 loudspeakers
         [[0 315:-45:45]' 30*ones(8,1)] ; [[0 315:-45:45]' -30*ones(8,1)]; ... % 8 loudspeakers, each at +/- 30 deg elevation
         [[0 270:-90:90]' 60*ones(4,1)] ; [[0 270:-90:90]' -60*ones(4,1)]; ... % 4 loudspeakers, each at +/- 60 deg elevation
         [0 90]];% 1 loudspeaker in zenith (above the listener)

LsPos = LsPos/180*pi;% transform loudspeaker locations from degrees to radians

N = size(LsPos,1);% number of loudspeakers (here 41)

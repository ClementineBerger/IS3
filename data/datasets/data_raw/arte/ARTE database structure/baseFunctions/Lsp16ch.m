% function [LsPos,N] = Lsp16ch
% 
% Creates (Nx2) matrix of N loudspeaker directions [azimuth elevation] 
% in radians for a given 2D loudspeaker array. Here, a ring of 16 equi-distant 
% loudspeakers is used as an example. The first loudspeaker is in front and 
% then counted anti-clockwise. 
%
% Note:
% Azimuth angle: 0 rad <= azimuth < 2*pi (anti-clockwise -> positive values to the left)
% Elevation angle: -pi/2 <= elevation <= pi/2 (here elevation is always 0)
% 
% Outputs:
% LsPos   Directions [azimuth elevation] in radians for all loudspeakers in
%         the array (size Nx2)
% N       Number of loudspeakers in array (here 16)

function [LsPos,N] = Lsp16ch

LsPos = [(0:22.5:337.5)' zeros(16,1)]; % horizontal ring of 16 equidistant loudspeakers

LsPos = LsPos/180*pi;% transform loudspeaker locations from degrees to radians

N = size(LsPos,1);% number of loudspeakers (here 41)

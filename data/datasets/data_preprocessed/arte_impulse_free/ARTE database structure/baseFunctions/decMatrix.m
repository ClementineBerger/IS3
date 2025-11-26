% function [D,MOAch] = decMatrix(LsPos, M3, M2)
%
% Calculates the decoding matrix for basic MOA coding. 
%
% Input:
% LsPos   Loudspeaker locations [azimuth elevation] in radians. Size N x 2
% M3      HOA components for 3D reproduction
% M2      HOA components for 2D reproduction
% 
% Output:
% D       Basic decoding matrix
% MOAch   MOA channels required in decoding process.

function [D,MOAch] = decMatrix(LsPos, M3, M2)

if M3 > 4
    error('Only 3D order M3 <= 4 supported.')
end
L = size(LsPos, 1);

if exist('M2','var') % if M2 exists
    if M2 > 7
        error('Only 2D order M2 <= 7 supported.')
    end
    M = M2;
    SHFidx = sort([1:(M3+1)^2, (M3+1:M2).^2+1, (M3+1:M2).^2+2]); % keep MOA functions only
else
    M = M3;
    SHFidx = 1:(M3+1)^2;
end

Y_ls = YmnsM(LsPos(:,1), LsPos(:,2), M); % 3D spherical harmonics sampled at loudspeaker locations
D = pinv(Y_ls(:, SHFidx)).'; % Basic Decoding matrix

% Relate the SHF indeces to the physical channels used in the provided MOA recordings 
MOAchRef = [1:27 37 38 50 51];% The 31 SHFs used in the MOA recordings
MOAch = zeros(size(SHFidx));% memory allocation
for i = 1:length(SHFidx)
    MOAch(i) = find(MOAchRef == SHFidx(i));% MOA channels required in decoding process.
end

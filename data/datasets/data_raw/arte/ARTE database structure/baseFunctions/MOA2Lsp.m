% function [sLsp] = MOA2Lsp(sMOA,D)
%
% Decodes the provided MOA signal, sMOA, into a multi-channel loudspeaker 
% signal, sLsp, using the provided decoding Matrix D. 
%
% Input:
% sMOA   MOA signal of length N with Nf channels and size: N x Nf
% D      Decoding matrix derived for Nlsp loudspeakers of size: Nlsp x Nf
%
% Output:
% sLsp   Multi-channel loudspeaker signal of size: N x Nlsp

function [sLsp] = MOA2Lsp(sMOA,D)

disp('Decoding MOA sound file...')

N = size(sMOA,1);% duration of MOA signal (samples)
Nf = size(sMOA,2);% number of MOA channels
if Nf ~= size(D,2)
    error('Provided decoding matrix does not fit the provided MOA signal.' )
end
Nlsp = size(D,1);% number of loudspeaker channels

sLsp = zeros(N,Nlsp);% memory allocation for multi-channel loudspeaker signal
h = waitbar(0,'Decoding sound file...');
for i1 = 1:Nlsp % run through all loudspeaker channels
    for i2 = 1:Nf % run through all 31 MOA channels
        sLsp(:,i1) = D(i1,i2) * sMOA(:,i2) + sLsp(:,i1);
    end
    waitbar(i1/Nlsp,h)
end
close(h)

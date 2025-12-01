% function saveLspFileDialog(sLsp,fs,Nbits)
%
% Saves the derived multi-channel loudspeaker sound file to disk. 
% Input:
% sLsp   multichannel loudspeaker signal
% fs     sampling frequency (Hz) for saving sound file. Default: 44100 Hz
% Nbits  number of bits per sample used for wav-file. Default: 32 bit

function saveLspFileDialog(sLsp,fs,Nbits)

if ~exist('fs','var')
    fs = 44100;% default sampling frequency for saving in wav-file (Hz)
end
if ~exist('Nbits','var')
    Nbits = 32;% default number of bits per sample for saving in wav-file
end

defaultName = ['decoded\Lsp_' num2str(size(sLsp,2)) 'ch'];% default name and path for storing the decoded loudspeaker signal

[fileLsp,pathLsp] = uiputfile('.wav','Select a filename for storing created loudspeaker file...',defaultName);% open dialog for saving sound file
 
disp('Saving decoded sound file to disk...')
audiowrite(fullfile(pathLsp,fileLsp),sLsp,fs,'BitsPerSample',Nbits)% write multi-channel wav-file to disk
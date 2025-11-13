% Example script that illustrates how to generate a multi-channel noise signal 
% to which reverberant speech is added at a desired signal-to-noise
% ratio. The speech signal could be the sentence material of a "standard" 
% speech test.
%   1. Read a provided MOA-coded Room Impulse Reponse (RIR) and decode it 
%      into a multi-channel loudspeaker signal.
%   2. Convolve the resulting multi-channel RIR with anechoic target
%      speech to generate reverberant target speech.
%   3. Read a provided MOA-coded noise signal, which was recorded in the 
%      same acoustic environment as the RIR, and decode it into a 
%      multi-channel loudspeaker signal.
%   4. Combine the reverberant speech signal with the noise at the desired
%      SNR.

%%% -----------------------------------------------------------------------
%%% Initialization
%%% -----------------------------------------------------------------------

clear % clear all variables from workspace
clc % clear command window

addpath('baseFunctions') %set path to main MOA functions

M2 = 7;%2D HOA order - must be <= 7 (Note: requires horizontal ring of k >= 2*M2+1 loudspeakers)
M3 = 0;% M3 = 0 because a horizontal 2D loudspeaker array is considered here.

[LsPos,Nlsp] = Lsp16ch;% Creates Nlsp x 2 matrix that defines the applied loudspeaker array layout. 
                       % Here, a regular horizontal 16 channel 2D array is considered as an example.

[D,MOAch] = decMatrix(LsPos, M3, M2);% derive basic decoding matrix D (and utilized MOA channels)

Nbits = 32;% number of Bits per sample for writing multi-channel loudspeaker wav-file
fs = 44100;% Sampling frequency (Hz) NOte that all provided sound files were sampled at 44100 Hz

SNR = 0;% Signal-to-noise energy ratio (dB) for the reverberant speech in noise

enhanceDS = 1;% flag for "enhancing" the direct sound component (0: no, 1: yes)

%%% -----------------------------------------------------------------------
%%% Generate reverberant target speech for multi-channel loudspeaker playback
%%% -----------------------------------------------------------------------

if enhanceDS == 0% Do not "enhance" direct sound component
    
    % Read example MOA-coded room impulse reponse for target speech
    nameRIR = 'MOA31ch\RIRs\RIR\04_Living_Room_0deg_31ch.wav';
    hMOA = readMOAfile(nameRIR,MOAch);

    % Decode MOA-coded RIR into multi-channel loudspeaker signal
    hLsp = MOA2Lsp(hMOA,D);
    
elseif enhanceDS == 1% "Enhance" direct sound component
    
    % Read example MOA-coded room impulse reponse for target speech
    nameDS = 'MOA31ch\RIRs\DS\04_Living_Room_0deg_31ch.wav';
    hMOA_DS = readMOAfile(nameDS,MOAch);% read only the direct sound component
    
    nameREV = 'MOA31ch\RIRs\REV\04_Living_Room_0deg_31ch.wav';
    hMOA_REV = readMOAfile(nameREV,MOAch);% read only the reverberation component
    
    % Decode MOA-coded RIR components into multi-channel loudspeaker signals
    hLsp_DS = MOA2Lsp(hMOA_DS,D);% direct sound component
    hLsp_REV = MOA2Lsp(hMOA_REV,D);% reverberation component
    
    % Combine direct sound and reverberant component
    hLsp = hLsp_REV;% initialize the final multi-channel RIR with the reverberant component
    hLsp_DS = sum(hLsp_DS,2);% "enhance" direct sound by summing all channels
    hLsp(1:size(hLsp_DS,1),1) =  hLsp(1:size(hLsp_DS,1),1) + hLsp_DS;% and adding it to channel 1 (loudspeaker at zero degree azimuth) of the (time-aligned) reverberant component
    
end

% Read anechoic target speech signal example (single channel file)
nameSpeech = 'baseFunctions\male_speech_example_mono.WAV';
sTanechoic = audioread(nameSpeech);

% Convolve anechoic target speech with RIR to generate reverberant target speech
sTreverbLsp = fftfilt(hLsp,[sTanechoic ; zeros(size(hLsp,1)-1,1)]);

%%% -----------------------------------------------------------------------
%%% Get background noise for the same acoustic environment as for the
%%% above RIR.
%%% -----------------------------------------------------------------------

N = size(sTreverbLsp,1);% length of reverberant speech (samples)

%%% Read MOA-coded noise file that correpsonds to the above RIR and
%%% has the same length as the reverberant speech.
nameNoise = 'MOA31ch\noiseFiles\04_Living_Room_MOA_31ch.wav';
sNoiseMOA = readMOAfile(nameNoise,MOAch,N);

% Decode MOA-coded noise signal into multi-channel loudspeaker signal
sNoiseLsp = MOA2Lsp(sNoiseMOA,D);

%%% -----------------------------------------------------------------------
%%% Combine the target speech with the background noise at the given SNR. 
%%% Note that this should actually be done using real sound pressure level 
%%% measurements using a claibrated omni-directional microphone placed
%%% inside the centre of the physical loudspeaker array.
%%% -----------------------------------------------------------------------

% Estimate of the noise level in the centre of the loudspeaker array. 
Lnoise = 20*log10( rms(sum(sNoiseLsp,2)) / 2e-5 );% noise level (dB)

% Estimate of the reverberant speech level in the centre of the loudspeaker array. 
Lspeech = 20*log10( rms(sum(sTreverbLsp,2)) / 2e-5 );% noise level (dB)

% Gain required to add the target speech to the background noise at the
% given SNR
gT = 10^( (SNR-Lspeech+Lnoise)/20 ) ;

% Adding reverberant target speech to the background noise at the given SNR
sCombinedLsp = gT*sTreverbLsp + sNoiseLsp;

%%% -----------------------------------------------------------------------
%%% Write the derived multi-channel reverberant speech-in-noise file to
%%% disk and create info file.
%%% -----------------------------------------------------------------------

% Saving wav-file to disk
nameSave = ['decoded\04_Living_Room_noise_with_Speech_SNR0dB_Lsp_16ch.wav'];
audiowrite(nameSave,sCombinedLsp,fs,'BitsPerSample',Nbits)

% Generate text file containing important information about the generated wav-file
fid = fopen([nameSave(1:end-4) '_info.txt'],'w');
fprintf(fid,'%s %s\r\n','File generated on: ',datestr(now));
fprintf(fid,'%s %d\r\n','MOA order M2 = ',M2);
fprintf(fid,'%s %d\r\n','MOA order M3 = ',M3);
fprintf(fid,'%s %d\r\n','SNR (dB) = ',SNR);
fprintf(fid,'%s %d\r\n','Sampling frequency (Hz) = ',fs);
fprintf(fid,'%s %s\r\n','Name of anechoic speech file: ',nameSpeech);
if enhanceDS == 0 % no direct sound "enhancement" is applied
    fprintf(fid,'%s\r\n','Enhancement of direct sound component: no');
    fprintf(fid,'%s %s\r\n','Name of RIR: ',nameRIR);
elseif enhanceDS == 1 % direct sound "enhancement" is applied
    fprintf(fid,'%s\r\n','Enhancement of direct sound component: yes');
    fprintf(fid,'%s %s\r\n','Name of RIR - direct sound: ',nameDS);
    fprintf(fid,'%s %s\r\n','Name of RIR - reverberation: ',nameREV);
end
fprintf(fid,'%s %s\r\n','Name of noise file: ',nameNoise);
fprintf(fid,'%s %s\r\n','Name of final speech-in-noise file: ',nameSave);
fprintf(fid,'%s %d\r\n','Number of loudspeakers = ',Nlsp);
fprintf(fid,'%s %s\r\n','Applied loudspeaker equalization: ','No');
fprintf(fid,'%s\r\n','Loudspeaker locations [azimuth elevation] in degrees:');
fprintf(fid,'%0.1f %0.1f\r\n',LsPos'/pi*180);
fclose(fid);


% function y = YmnsM(theta, delta, M)
%
% Evaluates the real-valued spherical harmonics functions at the given
% positions.
%
% input:
%   theta       column vector (length N): azimuth (in rad)
%   delta       column vector (length N): elevation (in rad)
%   M           Ambisonic order (M2D for mixed-order)
%
% output:
%   y           (K x Q):  each column of the matrix corresponds to one
%               ambisonic component (Q elements), each row to a position (K elements).
%               3D: K = (M+1)^2
%               2D: K = 2M+1
%               mixed order: K = (M3D+1)^2 + 2(M2D-M3D)
%
% References:
%       Favrot et. al. (AES, 2011) 131st convention, paper 8528
%       Daniel, J., Nicol, R. & Moreau, S. (2003), ‘Further investigations of high order
%       ambisonics and wavefield synthesis for holophonic sound imaging’, Presented
%       at the AES 114th convention p. preprint 5788.
%       Daniel, J., Rault, J. B. & Polack, J. D. (1998), ‘Ambisonics encoding of other
%       audio formats for multiple listening conditions’, Presented at the AES 105th
%       convention p. preprint 4795.
%
% Normalized 3D or 2D convention

function y = YmnsM(theta, delta, M)

y = ones(length(theta),(M+1)^2); % Initialize the output
k = 1;
% Calculate sph. harm. up to the truncation order M
for m = 0:M
    % Schmidt Seminormalized Associated Legendre Functions (the term (-1)^n is not present
    % contrary to the Matlab help)
    leg=legendre(m,sin(delta),'sch');
    for n = m:-1:0
        for s = 1:-2:-(n~=0)
            y(:,k) = sqrt(2*m+1)*leg(n+1,:)'.*...
                ((n==0)+(n~=0)*((s==1)*cos(n*theta)+(s==-1)*sin(n*theta)));
            k = k+1;
        end
    end
end

clear
clc
pic='giraffe.jpg';
img_gr=rgb2gray(imread(pic));
img  = im2double(img_gr);
%% Do FFT and get amplitude A, L(f)=log(A(f)) and phase P(f)
f = fft2(img);
L = log(abs(f));
P = angle(f);
%% calculate the statistical singularities R(f)
k=3;
H(1:k,1:k)=1/(k*k);
flimg= imfilter(L, H, 'replicate');
R = L - flimg;
S = abs(ifft2(exp(R + i*P))).^2;
%% And then do inverse Fourier transform, use a Gaussian fuzzy filter to get the significant area.
G=imfilter(S, fspecial('gaussian', [10, 10], 2.5));
S = mat2gray(G);
imshow(S);
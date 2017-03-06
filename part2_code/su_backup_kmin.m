%% super pixel product
 clear all
 format long
pic='part2\bird.jpg';
img  = im2double(imread(pic));
sizeofimg=size(img);
S_pixel=zeros(sizeofimg(1),sizeofimg(2),3);
Scale=1; 
while(Scale<=8) %可调参数，论文用了r,r/2,r/4 三个尺度的块大小
%% product super pixel
    numberofsp=800/Scale; %1600
% img_gr=rgb2gray(imread(pic));
sp  = mexGenerateSuperPixel(img, numberofsp);
m=max(max(sp));
Scale_no=floor(log2(Scale))+1;
% subplot(1,3,Scale_no);
% imshow(sp/m);%将图像矩阵转化到0-1之间

%% color transform\position distance\color distance init
numofpatch=0;
position=zeros(m,2);
color=zeros(m,3);
cform = makecform('srgb2lab'); 
lab = applycform(imread(pic), cform);
L=lab(:,:,1); A=lab(:,:,2); B=lab(:,:,3);
%% 计算各个块的颜色均值和中心位置
for i=0:m
   [indr,indl]= find(sp==i);
       pxinpac=length(indr);
   if pxinpac~=0
       numofpatch=numofpatch+1;
       %positon center of each patch
       position(numofpatch,1)=mean(indr);
       position(numofpatch,2)=mean(indl);
       
       sumL=0; sumA=0; sumB=0;
       for c =1:pxinpac
           sumL=sumL+uint64(L(indr(c),indl(c)));
           sumA=sumA+uint64(A(indr(c),indl(c)));
           sumB=sumL+uint64(B(indr(c),indl(c)));
       end
       color(numofpatch,1)=sumL/pxinpac;
       color(numofpatch,2)=sumA/pxinpac;
       color(numofpatch,3)=sumB/pxinpac;
       clear sumL sumA sumB
   end
   clear indr indl
end
%颜色均值：color(numofpatch,3) 中心位置：position(numofpatch,2) 块数：numofpatch

%% 计算各个块的颜色欧式距离和位置距离,最后计算相似程度
Dc = pdist(color,'euclidean');
% dcolor=squareform(Dc);
Dp = pdist(position,'euclidean');
% dposition=squareform(Dp);
clear c
PARM_C=2; %可调参数c,论文为3 2
D=Dc./(1+PARM_C*Dp);
d_similar=squareform(D);

%度量相似程度的距离：d_similar(numofpatch,numofpatch)

%% 找出每个块最相似的k个块
PARM_K=floor(120/Scale);  %可调参数k，论文为64
S_patch=zeros(numofpatch,1);
for i=1:numofpatch
    sortD=sort(d_similar(i,:));
    kmin=sortD(PARM_K);
    S_patch(i)=1-exp(-sum(sortD(1:PARM_K))/PARM_K);
end
for i=1:sizeofimg(1)
   for j=1:sizeofimg(2)
       S_pixel(i,j,Scale_no)=S_patch(sp(i,j)+1);
   end
end

%% for next iteration
Scale=Scale*2;

% s=strcat('ss',num2str(Scale_no),'.png');
% imwrite(S_pixel(:,:,Scale_no),s,'png');
end

%% mean S of each pixel\ find the foci pixels
mean_S_pixel=mean(S_pixel,3);
% imshow(mean_S_pixel);
% imshow(S_pixel(:,:,3));%块数多的时候铁丝网的显著度低
[focix,fociy]=find(mean_S_pixel>0.08);%1:0.15 2:0.05 3:0.08 bird:0.14+0.02*(5-Scale_no)
foci=[focix,fociy];
foci_img=zeros(sizeofimg(1),sizeofimg(2));
if isempty(focix)
    disp('ERROR: Please set a suitable threshold for mean_S_pixel!!!');
end
for i=1:size(focix)
    foci_img(foci(i,1),foci(i,2))=0.5;
end
% imshow(foci_img);

%% d_foci(p) for every pixel p
dfoci=zeros(sizeofimg(1),sizeofimg(2));
% weighted_S_pixel=zeros(sizeofimg(1),sizeofimg(2));
for i=1:sizeofimg(1)
    for j=1:sizeofimg(2)
        index=[i, j];
        dall=pdist2(index,foci);
        dfoci(i,j)=min(dall);
        clear dall
    end
end
dfoci=dfoci./max(max(dfoci));
% weighted_S_pixeln=mean_S_pixel.*(1-dfoci);
faceDetector = vision.CascadeObjectDetector;
bboxes = step(faceDetector,imread(pic)); % Detect faces
numofface=size(bboxes);
if numofface(1)~=0 
    for i=1:numofface(1)
        x=bboxes(i,1);
        y=bboxes(i,2);
        w=bboxes(i,3);
        h=bboxes(i,4);
        dfoci(y:y+h,x:x+w)=-0;
    end
end
weighted_S_pixel=mean_S_pixel.*(1-dfoci);


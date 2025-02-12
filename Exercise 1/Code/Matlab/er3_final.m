% Paths for my images and built in matlab functions
photoDir = 'D:\ceid\computer vision\askhsh 1\photos\';
functionDir = 'D:\ceid\computer vision\askhsh 1\';
addpath(functionDir);

% Load all images
P200 = im2double(imread([photoDir, 'P200.jpg']));
dog1 = im2double(imread([photoDir, 'dog1.jpg']));
dog2 = im2double(imread([photoDir, 'dog2.jpg']));
cat = im2double(imread([photoDir, 'cat.jpg']));
bench = im2double(imread([photoDir, 'bench.jpg']));
myImg = im2double(imread([photoDir, 'My_image.jpg']));

% Convert all images to grayscale 
if size(P200, 3) == 3, P200 = rgb2gray(P200); end
if size(dog1, 3) == 3, dog1 = rgb2gray(dog1); end
if size(dog2, 3) == 3, dog2 = rgb2gray(dog2); end
if size(cat, 3) == 3, cat = rgb2gray(cat); end
if size(bench, 3) == 3, bench = rgb2gray(bench); end
if size(myImg, 3) == 3, myImg = rgb2gray(myImg); end

% Define base size for the composition
baseSize = size(P200);

% Resize all images to match base size
dog1 = imresize(dog1, baseSize);
dog2 = imresize(dog2, baseSize);
cat = imresize(cat, baseSize);
bench = imresize(bench, baseSize);
myImg = imresize(myImg, baseSize);

% Create masks that can overlap but will be normalized to sum to 1
m1 = zeros(baseSize);
m2 = zeros(baseSize);
m3 = zeros(baseSize);
m4 = zeros(baseSize);
m5 = zeros(baseSize);
m6 = zeros(baseSize);

% Define potentially overlapping regions
rows = baseSize(1);
cols = baseSize(2);

% Create initial masks 
m1(1:rows, 1:floor(2*cols/3)) = 1;  
m2(1:floor(2*rows/3), floor(cols/3):floor(5*cols/6)) = 1;  
m3(floor(rows/3):end, floor(cols/4):floor(3*cols/4)) = 1;  
m4(1:floor(3*rows/4), floor(cols/2):end) = 1;  
m5(floor(rows/4):end, floor(2*cols/3):end) = 1;  
m6(floor(rows/4):floor(3*rows/4), floor(cols/3):floor(2*cols/3)) = 1;  

% Normalize masks 
maskSum = m1 + m2 + m3 + m4 + m5 + m6;
m1 = m1 ./ maskSum;
m2 = m2 ./ maskSum;
m3 = m3 ./ maskSum;
m4 = m4 ./ maskSum;
m5 = m5 ./ maskSum;
m6 = m6 ./ maskSum;

% Number of pyramid levels
numLevels = 5;

% Generate pyramids 

Gm1 = genPyr(m1, 'gauss', numLevels);
Gm2 = genPyr(m2, 'gauss', numLevels);
Gm3 = genPyr(m3, 'gauss', numLevels);
Gm4 = genPyr(m4, 'gauss', numLevels);
Gm5 = genPyr(m5, 'gauss', numLevels);
Gm6 = genPyr(m6, 'gauss', numLevels);

% Laplacian pyramids for images
LI1 = genPyr(P200, 'lap', numLevels);
LI2 = genPyr(dog1, 'lap', numLevels);
LI3 = genPyr(dog2, 'lap', numLevels);
LI4 = genPyr(cat, 'lap', numLevels);
LI5 = genPyr(bench, 'lap', numLevels);
LI6 = genPyr(myImg, 'lap', numLevels);

% Blend pyramids 
B = cell(1, numLevels);
for j = 1:numLevels
    [levelRows, levelCols] = size(LI1{j});
    
    % Resize masks to match current level
    g1 = imresize(Gm1{j}, [levelRows, levelCols]);
    g2 = imresize(Gm2{j}, [levelRows, levelCols]);
    g3 = imresize(Gm3{j}, [levelRows, levelCols]);
    g4 = imresize(Gm4{j}, [levelRows, levelCols]);
    g5 = imresize(Gm5{j}, [levelRows, levelCols]);
    g6 = imresize(Gm6{j}, [levelRows, levelCols]);
    
    % Blend 
    B{j} = g1 .* LI1{j} + g2 .* LI2{j} + g3 .* LI3{j} + g4 .* LI4{j} + g5 .* LI5{j} + g6 .* LI6{j};
end

% Reconstruct 
result = pyrReconstruct(B);


figure;
imshow(result, []);
title('Final Composition');
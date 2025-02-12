% Paths for my images and built in matlab function
photoDir = 'D:\ceid\computer vision\askhsh 1\photos\';
functionDir = 'D:\ceid\computer vision\askhsh 1\';
addpath(functionDir);

% Load images
woman = im2double(imread([photoDir, 'woman.png']));
hand = im2double(imread([photoDir, 'hand.png']));

% Ensure that it is in grayscale
if size(woman, 3) == 3
    woman = rgb2gray(woman);
end
if size(hand, 3) == 3
    hand = rgb2gray(hand);
end

% Resize images to match
[rows, cols] = size(hand);
woman = imresize(woman, [rows, cols]);

% Create mask 
m1 = zeros(rows, cols);
eyeRegion = round([rows*0.3 rows*0.7 cols*0.3 cols*0.7]);
m1(eyeRegion(1):eyeRegion(2), eyeRegion(3):eyeRegion(4)) = 1;

% Number of levels
numLevels = 5;

% Generate pyramids 
Gm1 = genPyr(m1, 'gauss', numLevels);     
LI1 = genPyr(woman, 'lap', numLevels);     
LI2 = genPyr(hand, 'lap', numLevels);      

% Create blended pyramid 
B = cell(1, numLevels);
for j = 1:numLevels
   
    [levelRows, levelCols] = size(LI1{j});
 
    gj = imresize(Gm1{j}, [levelRows, levelCols]);
  
    B{j} = gj .* LI1{j} + (1 - gj) .* LI2{j};
end

% Reconstruct 
result = pyrReconstruct(B);


figure;
imshow(result, []);
title('Blended Result');
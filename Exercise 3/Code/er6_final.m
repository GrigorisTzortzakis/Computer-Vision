

clc; clear; close all;

% Read and Rotate Images
imageDir = 'D:\ceid\computer vision\askhsh 3';  % Adjust if needed
img1Name = 'balcony1.jpg';
img2Name = 'balcony2.jpg';

I1 = imread(fullfile(imageDir, img1Name));
I2 = imread(fullfile(imageDir, img2Name));

% Rotate each image 180 degrees 
I1 = imrotate(I1, 180);
I2 = imrotate(I2, 180);


if size(I1,3) == 3
    I1gray = rgb2gray(I1);
else
    I1gray = I1;
end

if size(I2,3) == 3
    I2gray = rgb2gray(I2);
else
    I2gray = I2;
end

% SIFT Keypoints
siftPoints1 = detectSIFTFeatures(I1gray, ...
    'ContrastThreshold',   0.04, ...
    'EdgeThreshold',       5, ...
    'NumLayersInOctave',   5, ...
    'Sigma',               sqrt(2));

siftPoints2 = detectSIFTFeatures(I2gray, ...
    'ContrastThreshold',   0.04, ...
    'EdgeThreshold',       5, ...
    'NumLayersInOctave',   5, ...
    'Sigma',               sqrt(2));

[features1, validPts1] = extractFeatures(I1gray, siftPoints1, 'Method','SIFT');
[features2, validPts2] = extractFeatures(I2gray, siftPoints2, 'Method','SIFT');

% Match Features
indexPairs = matchFeatures(features1, features2, ...
    'MaxRatio', 0.9, ...
    'Unique',   true);

matchedPts1 = validPts1(indexPairs(:,1));
matchedPts2 = validPts2(indexPairs(:,2));

% Visualize matches before RANSAC
figure('Name','Raw Matches (No RANSAC)','Color','white');
showMatchedFeatures(I1gray, I2gray, matchedPts1, matchedPts2, ...
    'montage','PlotOptions',{'ro','go','y--'});
title('Putative Matches Before RANSAC (Upright)');

%  Estimate Transform via RANSAC
[tform, inlierIdx] = estimateGeometricTransform2D( ...
    matchedPts1.Location, matchedPts2.Location, ...
    'similarity', ...
    'MaxDistance', 5, ...
    'Confidence',  99);

inlierPts1 = matchedPts1(inlierIdx);
inlierPts2 = matchedPts2(inlierIdx);

% Visualize the inlier matches after RANSAC
figure('Name','Matches After RANSAC (Upright)','Color','white');
showMatchedFeatures(I1gray, I2gray, inlierPts1, inlierPts2, ...
    'montage','PlotOptions',{'ro','go','y--'});
title('Inlier Matches After RANSAC (Upright)');


fprintf('\nTotal matches before RANSAC: %d\n', matchedPts1.Count);
fprintf('Inlier matches after RANSAC: %d\n', inlierPts1.Count);
fprintf('\nTransformation class: %s\n', class(tform));
disp('Transformation matrix (tform.T): ');
disp(tform.T);

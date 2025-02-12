% Paths for my images
photoDir = 'D:\ceid\computer vision\askhsh 1\photos\';
functionDir = 'D:\ceid\computer vision\askhsh 1\';

% Add the function directory to the MATLAB path
addpath(functionDir);

% Load the images
apple = im2double(imread([photoDir, 'apple.jpg']));
orange = im2double(imread([photoDir, 'orange.jpg']));

% Resize images to be of the same dimensions
[rows, cols, ~] = size(orange);
apple = imresize(apple, [rows, cols]);

% Define the number of pyramid levels
numLevels = 5;

%% Generate Gaussian pyramids for both images
gaussApple = genPyr(apple, 'gauss', numLevels);
gaussOrange = genPyr(orange, 'gauss', numLevels);

%% Generate Laplacian pyramids for both images
lapApple = genPyr(apple, 'lap', numLevels);
lapOrange = genPyr(orange, 'lap', numLevels);

%% Visualization: Gaussian and Laplacian Pyramids
% Gaussian pyramid visualization for apple
figure;
for i = 1:numLevels
    subplot(2, numLevels, i);
    imshow(gaussApple{i}, []);
    title(['Gaussian (Level ', num2str(i), ')']);
    
    subplot(2, numLevels, i + numLevels);
    imshow(lapApple{i} + 0.5, []); % Offset added for visualization
    title(['Laplacian (Level ', num2str(i), ')']);
end
sgtitle('Apple: Gaussian and Laplacian Pyramids');

% Gaussian pyramid visualization for orange
figure;
for i = 1:numLevels
    subplot(2, numLevels, i);
    imshow(gaussOrange{i}, []);
    title(['Gaussian - Orange (Level ', num2str(i), ')']);
    
    subplot(2, numLevels, i + numLevels);
    imshow(lapOrange{i} + 0.5, []); 
    title(['Laplacian - Orange (Level ', num2str(i), ')']);
end
sgtitle('Orange: Gaussian and Laplacian Pyramids');

%% Create Gaussian mask
mask = zeros(rows, cols, 3);
mask(:, 1:floor(cols/2), :) = 1; % Left half for apple, right half for orange
gaussMask = genPyr(mask, 'gauss', numLevels);


% Blend using the Laplacian pyramids and Gaussian mask
blendPyr = cell(1, numLevels);
for i = 1:numLevels
    % Resize the Gaussian mask to match the Laplacian pyramid level
    [levelRows, levelCols, ~] = size(lapApple{i});
    resizedMask = imresize(gaussMask{i}, [levelRows, levelCols]);
    
    % Perform blending for the current level
    blendPyr{i} = resizedMask .* lapApple{i} + (1 - resizedMask) .* lapOrange{i};
end

% Reconstruct the blended image
blendedImage = pyrReconstruct(blendPyr);

% Feathered blend: Blend directly without pyramids 
featheredBlend = mask .* apple + (1 - mask) .* orange;


figure;
subplot(1, 2, 1);
imshow(blendedImage);
title('Blend with gauss and laplace');

subplot(1, 2, 2);
imshow(featheredBlend);
title('Blend with just cropping');

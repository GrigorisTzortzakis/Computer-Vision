% Set up the directory
imageDir = 'D:\ceid\computer vision\askhsh 2';
imageFile = fullfile(imageDir, 'beach.jpg'); 
outputFile = fullfile(imageDir, 'scaled_composition.jpg');

% Read the image
originalImage = imread(imageFile);
[rows, cols, ~] = size(originalImage);

% Create canvas same size as original
canvas = uint8(zeros(rows, cols, 3));

% Define just 5 scaling factors and overlapping positions
scalingFactors = [1.0, 0.8, 0.6, 0.4, 0.2];  % Largest to smallest
positions = [
    1 1;                % Large image at top-left
    cols/3 1;          % Overlap right side
    1 rows/3;          % Overlap bottom
    cols/2 rows/2;     % Center
    cols/4 rows/4      % Fill any remaining gaps
];

% Place images from largest to smallest to ensure complete coverage
for i = 1:length(scalingFactors)
    scale = scalingFactors(i);
    tform = affine2d([scale 0 0; 0 scale 0; 0 0 1]);
    scaledImage = imwarp(originalImage, tform);
    
    x = round(positions(i, 1));
    y = round(positions(i, 2));
    
    [sRows, sCols, ~] = size(scaledImage);
    rowEnd = min(y + sRows - 1, rows);
    colEnd = min(x + sCols - 1, cols);
    
    % Place the scaled image
    canvas(y:rowEnd, x:colEnd, :) = scaledImage(1:rowEnd-y+1, 1:colEnd-x+1, :);
end

% Display result
imshow(canvas);
title('Scaled Beach Composition');
imwrite(canvas, outputFile);
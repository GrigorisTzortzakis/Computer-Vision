% Read the input image with alpha channel
[img, map, alpha] = imread('pudding.png');


if isempty(alpha) && size(img,3) == 4
    alpha = img(:,:,4);
    img = img(:,:,1:3);
end

% Create video writer object
v = VideoWriter('pudding_erotima3.avi');
v.FrameRate = 30;
open(v);

% Get dimensions
[height, width, ~] = size(img);

% Create canvas size
canvasWidth = width * 3;
canvasHeight = height;

% Parameters
maxShear = 0.2;
numFrames = 120;

% Create the vertical line boundary
boundary = bwperim(alpha > 0);
% Create vertical line pattern
[y, x] = find(boundary);
verticalLines = false(size(boundary));
for i = 1:length(x)
    if mod(x(i), 2) == 0
        verticalLines(max(1,y(i)-2):min(height,y(i)+2), x(i)) = true;
    end
end

for i = 1:numFrames
    % Create pure white background
    frame = uint8(255 * ones(canvasHeight, canvasWidth, 3));
    
    % Calculate shear
    shearFactor = maxShear * sin(2*pi*i/numFrames);
    
    % Create transformation matrix
    tform = affine2d([1 0 0; shearFactor 1 0; width*1.5 0 1]);
    
    % Apply transformations
    outputView = imref2d([canvasHeight canvasWidth]);
    shearedImg = imwarp(img, tform, 'OutputView', outputView, 'FillValues', 0);
    shearedAlpha = imwarp(alpha, tform, 'OutputView', outputView, 'FillValues', 0);
    shearedBoundary = imwarp(verticalLines, tform, 'OutputView', outputView, 'FillValues', 0);
    
    % Convert alpha to 3D
    alphaMask = repmat(double(shearedAlpha) / 255, [1 1 3]);
    
    % Composite main image
    frame = uint8(double(shearedImg) .* alphaMask + double(frame) .* (1 - alphaMask));
    
    % Add vertical line boundary 
    boundaryMask = repmat(shearedBoundary, [1 1 3]);
    frame(boundaryMask) = 0;
    
    % Write frame
    writeVideo(v, frame);
end

close(v);
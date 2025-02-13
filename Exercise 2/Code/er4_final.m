% Read the input image with alpha channel
[img, map, alpha] = imread('pudding.png');


if isempty(alpha) && size(img,3) == 4
    alpha = img(:,:,4);
    img = img(:,:,1:3);
end

% Create video writer object
v = VideoWriter('sheared_pudding.avi');
v.FrameRate = 60;  
open(v);

% Get dimensions
[height, width, ~] = size(img);

% Create canvas size 
canvasWidth = width * 3;
canvasHeight = height;

% Parameters
maxShear = 0.3;  % Maximum shear factor
numFrames = 180;  % Keep 3 seconds duration
totalCycles = 3;  % Exactly 3 complete cycles

% Create the vertical line boundary
boundary = bwperim(alpha > 0);
[y, x] = find(boundary);
verticalLines = false(size(boundary));
for i = 1:length(x)
    if mod(x(i), 2) == 0
        verticalLines(max(1,y(i)-2):min(height,y(i)+2), x(i)) = true;
    end
end

% Create a height map for progressive shearing
[Y, ~] = meshgrid(1:height, 1:width);
Y = Y'; % Transpose to match image dimensions
heightFactor = (height - Y) / height; % 0 at bottom, 1 at top

for i = 1:numFrames
    % Create white background
    frame = uint8(255 * ones(canvasHeight, canvasWidth, 3));
    
    % Calculate shear factor 
    currentShear = maxShear * sin(2*pi*totalCycles*i/numFrames);
    
    % Apply progressive shearing based on height
    shearedImg = uint8(zeros(size(frame)));
    shearedAlpha = zeros(size(frame, 1), size(frame, 2));
    shearedBoundary = false(size(frame, 1), size(frame, 2));
    
    % Apply shear to each row based on its height
    for y = 1:height
        shiftAmount = round(currentShear * (height - y));
        xOffset = width * 1.5; % Center position
        
        % Shift this row by the calculated amount
        if y <= height
            rowRange = 1:width;
            newX = rowRange + xOffset + shiftAmount;
            validX = newX > 0 & newX <= canvasWidth;
            
            % Copy image data for this row
            for c = 1:3
                shearedImg(y, newX(validX), c) = img(y, rowRange(validX), c);
            end
            shearedAlpha(y, newX(validX)) = alpha(y, rowRange(validX));
            shearedBoundary(y, newX(validX)) = verticalLines(y, rowRange(validX));
        end
    end
    
    % Convert alpha to 3D mask
    alphaMask = repmat(double(shearedAlpha) / 255, [1 1 3]);
    
    % Composite the image
    frame = uint8(double(shearedImg) .* alphaMask + double(frame) .* (1 - alphaMask));
    
    % Add the boundary lines
    boundaryMask = repmat(shearedBoundary, [1 1 3]);
    frame(boundaryMask) = 0;
    
    % Write the frame
    writeVideo(v, frame);
end

close(v);
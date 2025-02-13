%% Read the images
windmill = imread('windmill.png');
mask = imread('windmill_mask.png');
background = imread('windmill_back.jpeg');


windmill = im2double(windmill);
mask = im2double(rgb2gray(mask));
background = im2double(background);

%% Create a binary mask

mask = (mask > 0.5);  
mask = ~mask;         

%% Find the bounding box of the windmill 
[row, col] = find(mask);
crop_top    = min(row);
crop_bottom = max(row);
crop_left   = min(col);
crop_right  = max(col);

%% Crop the region of interest from the mask and the windmill
mask_cropped = mask(crop_top:crop_bottom, crop_left:crop_right);
windmill_cropped = windmill(crop_top:crop_bottom, crop_left:crop_right, :);

%%  scale factor 
scale_factor = 0.5;


resized_mask = imresize(mask_cropped, scale_factor, 'nearest');   % Mask -> always use 'nearest'
resized_windmill = imresize(windmill_cropped, scale_factor, 'bicubic');


resized_background = imresize(background, scale_factor, 'bicubic');


v = VideoWriter('transf_windmill.avi');
v.FrameRate = 30;  
open(v);

% Get dimensions
[bg_h, bg_w, ~] = size(resized_background);
[obj_h, obj_w, ~] = size(resized_windmill);

% Calculate offsets to roughly center the windmill on the background
y_offset = round((bg_h - obj_h) / 2);
x_offset = round((bg_w - obj_w) / 2);


num_frames = 200;
angle_step = -360 / num_frames;  %

%%  build the video frames
for frame_idx = 1:num_frames
    % Current angle
    angle = (frame_idx - 1) * angle_step;
    
    % Rotate the windmill and mask 
    rotated_windmill = imrotate(resized_windmill, angle, 'bicubic', 'crop');
    rotated_mask = imrotate(resized_mask, angle, 'nearest', 'crop');
    rotated_mask = rotated_mask > 0.5;  % Ensure it is still a binary mask
    
 
    current_frame = resized_background;
    
    % Define the region of interest 
    roi = current_frame(y_offset+1 : y_offset+obj_h, x_offset+1 : x_offset+obj_w, :);
    
    % Blend using the mask
    for c = 1:3
        roi(:,:,c) = rotated_windmill(:,:,c) .* rotated_mask + ...
                     roi(:,:,c)              .* (1 - rotated_mask);
    end
    
    
    current_frame(y_offset+1 : y_offset+obj_h, x_offset+1 : x_offset+obj_w, :) = roi;
    
    
    writeVideo(v, im2uint8(current_frame));
end

%% Close the video file
close(v);
disp('Video "transf_windmill1.avi" saved successfully!');

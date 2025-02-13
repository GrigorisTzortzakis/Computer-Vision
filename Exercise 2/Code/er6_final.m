% Read images
windmill = imread('windmill.png');
mask = imread('windmill_mask.png');
background = imread('windmill_back.jpeg');

% Convert images to double for processing
windmill = im2double(windmill);
mask = im2double(rgb2gray(mask));
background = im2double(background);

mask = mask > 0.5; 
mask = ~mask;      

% Find bounding box for cropping
[row, col] = find(mask);
crop_top = min(row);
crop_bottom = max(row);
crop_left = min(col);
crop_right = max(col);

% Crop images to bounding box
mask = mask(crop_top:crop_bottom, crop_left:crop_right);
windmill = windmill(crop_top:crop_bottom, crop_left:crop_right, :);

% Resize images
new_width = round(size(mask, 2) / 2);
new_height = round(size(mask, 1) / 2);
background = imresize(background, 0.5);

% interpolation methods
interpolation_methods = {'nearest', 'bilinear', 'bicubic'};
video_names = {'transf_windmill_nearest.avi', 'transf_windmill_linear.avi', 'transf_windmill_cubic.avi'};

% Process for each interpolation method
for method_idx = 1:length(interpolation_methods)
    method = interpolation_methods{method_idx};
    video_name = video_names{method_idx};
    
    % Resize mask and windmill
    resized_mask = imresize(mask, [new_height, new_width], method);
    resized_windmill = imresize(windmill, [new_height, new_width], method);

    % Get dimensions for positioning
    [bg_height, bg_width, ~] = size(background);
    y_offset = round((bg_height - new_height) / 2);
    x_offset = round((bg_width - new_width) / 2);

    % Create video writer
    v = VideoWriter(video_name);
    v.FrameRate = 30;
    open(v);

    
    num_frames = 200;
    angle_step = -360 / num_frames;

    for frame = 1:num_frames
        % Calculate rotation angle
        angle = (frame - 1) * angle_step;

        % Rotate windmill and mask with current method
        rotated_windmill = imrotate(resized_windmill, angle, method, 'crop');
        rotated_mask = imrotate(resized_mask, angle, 'nearest', 'crop'); 
        rotated_mask = rotated_mask > 0.5; 

        % Create current frame with background
        current_frame = background;

        % Define ROI 
        roi = current_frame(y_offset+1:y_offset+new_height, x_offset+1:x_offset+new_width, :);

        % Apply the mask and blend the images
        for c = 1:3
            roi(:,:,c) = rotated_windmill(:,:,c) .* rotated_mask + roi(:,:,c) .* (1 - rotated_mask);
        end

        % Insert blended ROI back into the frame
        current_frame(y_offset+1:y_offset+new_height, x_offset+1:x_offset+new_width, :) = roi;

     
        current_frame = im2uint8(current_frame);
        writeVideo(v, current_frame);
    end

    % Finalize and close video writer
    close(v);
    fprintf('Saved video with %s interpolation: %s\n', method, video_name);
end

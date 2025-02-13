clc; clear; close all;


ball = im2double(imread('ball.jpg'));            
ball_mask = im2double(rgb2gray(imread('ball_mask.jpg'))); 
beach = im2double(imread('beach.jpg'));          


ball_mask = ~ball_mask;

% === Parameters ===
fps = 30;                      
duration = 10;                 
total_frames = fps * duration; 
initial_ball_size = 150;       
min_ball_size = 50;            
pad = 50;                      
start_x = 100;                
end_x = 600;                   
start_y = 400;                 
end_y = 200;                   
rotation_speed = 90 / fps;     


beach = imresize(beach, [600, 1080]);


outputVideo = VideoWriter('ball_into_distance.avi');
outputVideo.FrameRate = fps;
open(outputVideo);


for frame = 1:total_frames
    % Copy the background frame
    current_frame = beach;
    
    % Calculate ball size 
    ball_size = initial_ball_size - ...
                (initial_ball_size - min_ball_size) * (frame / total_frames);
    
    % Resize ball and mask for current frame
    resized_ball = imresize(ball, [ball_size ball_size]);
    resized_mask = imresize(ball_mask, [ball_size ball_size]);

    % Add padding 
    padded_ball = padarray(resized_ball, [pad pad 0], 0, 'both');
    padded_mask = padarray(resized_mask, [pad pad], 0, 'both');

    % Calculate position 
    x = start_x + (end_x - start_x) * (frame / total_frames);
    y = start_y + (end_y - start_y) * (frame / total_frames);

    % Ball rotation
    angle = -(frame * rotation_speed);
    rotated_ball = imrotate(padded_ball, angle, 'bilinear', 'crop');
    rotated_mask = imrotate(padded_mask, angle, 'bilinear', 'crop');
    
    % Ball position
    x_pos = round(x - size(rotated_ball, 2) / 2);
    y_pos = round(y - size(rotated_ball, 1) / 2);

    % Ensure ball stays within scene bounds
    x_pos = max(1, min(size(beach, 2) - size(rotated_ball, 2), x_pos));
    y_pos = max(1, min(size(beach, 1) - size(rotated_ball, 1), y_pos));

    % Region of Interest 
    roi = current_frame(y_pos:y_pos+size(rotated_ball,1)-1, ...
                        x_pos:x_pos+size(rotated_ball,2)-1, :);

    % Blend ball with the scene using mask
    for c = 1:3
        roi(:,:,c) = rotated_ball(:,:,c) .* rotated_mask + ...
                     roi(:,:,c) .* (1 - rotated_mask);
    end
    
    % Place the ROI back into the frame
    current_frame(y_pos:y_pos+size(rotated_ball,1)-1, ...
                  x_pos:x_pos+size(rotated_ball,2)-1, :) = roi;

    
    writeVideo(outputVideo, im2uint8(current_frame));
end


close(outputVideo);
disp('Ball moving into the distance video created successfully!');

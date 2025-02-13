clc; clear; close all;

% === Load Images ===
ball = im2double(imread('ball.jpg'));            % Ball image
ball_mask = im2double(rgb2gray(imread('ball_mask.jpg'))); % Mask (grayscale)
beach = im2double(imread('beach.jpg'));          % Background

% === Invert the mask for direct blending ===
ball_mask = ~ball_mask;

% === Parameters ===
fps = 30;                      % Frames per second
duration = 10;                 % Animation duration (seconds)
total_frames = fps * duration; % Total number of frames
ball_size = 150;               % Ball size
pad = 50;                      % Padding for rotation edge issues
initial_height = 300;          % Initial bounce height
num_bounces = 8;               % Number of bounces
decay_factor = 0.65;           % Height decay per bounce
rotation_speed = 90 / fps;    % Rotation increment per frame (degrees)

% === Resize Images ===
ball = imresize(ball, [ball_size ball_size]);
ball_mask = imresize(ball_mask, [ball_size ball_size]);

% Add padding to ball and mask
padded_ball = padarray(ball, [pad pad 0], 0, 'both');
padded_mask = padarray(ball_mask, [pad pad], 0, 'both');

% Get scene size
[sceneH, sceneW, ~] = size(beach);
start_x = 100;
end_x = sceneW - 200;
ground_y = sceneH - 150;

% === Video Writer Setup ===
outputVideo = VideoWriter('ball_bounce.avi');
outputVideo.FrameRate = fps;
open(outputVideo);

% === Animation Loop ===
for frame = 1:total_frames
    % Copy the background frame
    current_frame = beach;

    % Horizontal movement
    x = start_x + (end_x - start_x) * (frame / total_frames);
    
    % Bounce height calculation (sinusoidal with decay)
    time_per_bounce = duration / num_bounces;
    local_time = mod((frame / fps), time_per_bounce);
    current_bounce = floor((frame / fps) / time_per_bounce);
    current_max_height = initial_height * (decay_factor ^ current_bounce);
    bounce_progress = local_time / time_per_bounce;
    y = ground_y - current_max_height * sin(bounce_progress * pi);

    % Ball rotation
    angle = -(frame * rotation_speed);
    rotated_ball = imrotate(padded_ball, angle, 'bilinear', 'crop');
    rotated_mask = imrotate(padded_mask, angle, 'bilinear', 'crop');
    
    % Ball position
    x_pos = round(x - size(rotated_ball, 2) / 2);
    y_pos = round(y - size(rotated_ball, 1) / 2);

    % Ensure ball stays within scene bounds
    x_pos = max(1, min(sceneW - size(rotated_ball, 2), x_pos));
    y_pos = max(1, min(sceneH - size(rotated_ball, 1), y_pos));

    % Region of Interest (ROI)
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

    % Write the frame
    writeVideo(outputVideo, im2uint8(current_frame));
end

% === Cleanup ===
close(outputVideo);
disp('Ball bounce animation created successfully!');

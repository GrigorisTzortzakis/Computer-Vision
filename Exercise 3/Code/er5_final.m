

clc; clear;

% Find the pics
imageDir   = 'D:\ceid\computer vision\askhsh 3';
imageFiles = {'plant1.jpg','plant2.jpg','plant3.jpg','study.jpg','entrance.jpg'};
nImages    = numel(imageFiles);

% Loop 
for idx = 1:nImages
    
    currentImagePath = fullfile(imageDir, imageFiles{idx});
    fprintf('Processing image: %s\n', currentImagePath);
    
    % Read the image.
    img = imread(currentImagePath);
    % Convert to grayscale 
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Overwrite Sift variable
    imwrite(img, fullfile(imageDir, 'cameraman.tif'));
    
    % Save our variables
    save('wrapperVars.mat', 'imageDir', 'imageFiles', 'nImages', 'idx');
    
    % Run SIFT
    run('SIFT_feature.m');
    
    
    load('wrapperVars.mat', 'imageDir', 'imageFiles', 'nImages', 'idx');
    
   
    drawnow;
    
  
    
    
    hOrig1 = findobj('Type','Figure','Number',1);
    if ~isempty(hOrig1)
        newFig1 = figure('Name', sprintf('SIFT Figure 1 - %s', imageFiles{idx}));
        
        
        origAxes = findobj(hOrig1, 'Type', 'axes');
        for j = 1:length(origAxes)
            newAx = copyobj(origAxes(j), newFig1);
            set(newAx, 'Position', get(origAxes(j), 'Position'));
           
            set(newAx, 'YDir', 'normal'); 
            axis(newAx, 'image', 'xy');  
        end
    end
    
    
    hOrig2 = findobj('Type','Figure','Number',2);
    if ~isempty(hOrig2)
        newFig2 = figure('Name', sprintf('SIFT Figure 2 - %s', imageFiles{idx}));
        origAxes = findobj(hOrig2, 'Type', 'axes');
        for j = 1:length(origAxes)
            newAx = copyobj(origAxes(j), newFig2);
            set(newAx, 'Position', get(origAxes(j), 'Position'));
           
            set(newAx, 'YDir', 'normal');
            axis(newAx, 'image', 'xy');
        end
    end
    
   
    if ~isempty(hOrig1)
        close(hOrig1);
    end
    if ~isempty(hOrig2)
        close(hOrig2);
    end
    
   
end


if exist('wrapperVars.mat','file')
    delete('wrapperVars.mat');
end

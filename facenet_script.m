function  output_label = facenet_script(train_path, test_path)
%%   Using a pretrained Facenet model based on the concept of siamese networks.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

% I do not claim the pretrained model as my own work, and was retrieved from:
% Source: https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn
% Github: https://github.com/nyoki-mtl/keras-facenet
% Using the facenet implementation in Keras2, by Hiroki Taniai

%% Load data

% Create face detector - cascadeObjectDetector uses Viola-Jones algorithm.
detector = vision.CascadeObjectDetector('ClassificationModel', 'FrontalFaceCART');

% This is used to read files from the datastore. The anonymous function
% allows us to pass in the face detector to be reused, rather than
% initializing it many times and slowing down the process.
read_f = @(filename) read_and_resize(filename, detector);

% Create the datastores
training_images = imageDatastore(train_path, 'IncludeSubfolders', true, ...
    'LabelSource','foldernames', 'ReadFcn', read_f);

test_images = imageDatastore(test_path, 'ReadFcn', read_f);

% Get labels
folder_names = ls(train_path);

labels = folder_names(3:end,:); % the folder names are the labels


%% Load the network
% Source: https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn
% Github: https://github.com/nyoki-mtl/keras-facenet
% Using the facenet implementation in Keras2, by Hiroki Taniai

% The regression layer is not needed, but a requirement by matlab. Instead
% we take activations from the layer before it to get the feature vector.
keras_layers = importKerasLayers("facenet_keras.h5", ...
    'ImportWeights', true,'OutputLayer', 'regression');% 'WeightFile', "facenet_keras_weights.h5", 'OutputLayer', 'regression');

% Need to replace lambda layers that aren't supported by MATLAB
% Their purpose is to add a layer that computes a custom function
% In facenet and inception_resnet_v2 that its based upon, lambda computes
% a scalesum. Sums the input and multiplies by a set scale (determined when
% creating the layer).

% Our 'lambda' layer is defined in scalesum_lambda_replacement.
% Now to replace the placeholder layers with it.


%% Replace each placeholder layer with the appropriate type layer
%  Scales From: https://github.com/nyoki-mtl/keras-facenet/blob/master/code/inception_resnet_v1.py
% use replaceLayer(net_layers, 'layer_name', new_layer) to replace them
% returns entire graph, with updated layer replaced.
placeholders = findPlaceholderLayers(keras_layers);
% Block 35 x5, scale = 0.17
for i = 1:5
    name = strcat("Block35_", num2str(i), "_ScaleSum");
    layer = scalesum_lambda_replacement(0.17, 2, name);
    keras_layers = replaceLayer(keras_layers, name, layer);
end

% Block 17 x10, scale = 0.1
for i = 1:10
    name = strcat("Block17_", num2str(i), "_ScaleSum");
    layer = scalesum_lambda_replacement(0.1, 2, name);
    keras_layers = replaceLayer(keras_layers, name, layer);

end

% Block 8 x5, scale = 0.2
for i = 1:5
    name = strcat("Block8_", num2str(i), "_ScaleSum");
    layer = scalesum_lambda_replacement(0.2, 2, name);
    keras_layers = replaceLayer(keras_layers, name, layer);

end

% Block 8, branch 6, scale = 1
name = "Block8_6_ScaleSum";
layer = scalesum_lambda_replacement(1., 2, name);
keras_layers = replaceLayer(keras_layers, name, layer);

%% Convert to DAGNetwork - assemble
% It is currently a layergraph object, assemble it for use.
net = assembleNetwork(keras_layers);
% Could save new net here.

%% Feed training images into the network
% Get the feature vector of each training image and store them,
% so they only need to be computed once.
% They need to be of the type dlarray to be compatible with the network.

% The last layer before the regression layer MATLAB required.
l_name = "Bottleneck_BatchNorm";
f_x1 = activations(net, training_images ,l_name, 'OutputAs','columns');

f_x2 = activations(net, test_images, l_name, 'OutputAs', 'columns');
%face net output needs to be normalized.


%% Compute labels
% Iterate through matrices of feature vectors to find the euclidean
% distance between them. The training vector that is closest to a test
% vector is the label.
output_label = [];
for i = 1:1344
    d = [];
    for j = 1:100
        % Feature vectors must be L2 normalized, so that they are unit
        % vector with magnitude of 1.
        d_j = (f_x2(:,i)/norm(f_x2(:,i)) - f_x1(:,j)/norm(f_x1(:,j)));
        d = [d sqrt(d_j' * d_j)];
    end
    [~, index] = min(d);
    output_label = [output_label; labels(index, :)];
end

%% Function that reads images in the datastore and performs pre-processing.
function processed_image = read_and_resize(filename, detector)
    % For facenet, a face detector is used, and a tight bounding box
    % surrounding the face is found. This face is cropped from the image,
    % resized to the network input dimensions and fed into the network.
    I = imread(filename);

    % Find faces in the image.
    faces = detector(I);
    if(size(faces, 1) > 1)
        % This goes through each row and ensures the 'face' found is the
        % largest one. This is to ensure a false positive isn't used in the
        % event multiple faces are found, as the actual face will be the
        % largest one in the image.

        % Initialize.
        best_face = [0 0 0 0];
        for f = 1:size(faces,1)
            if faces(f,3)*faces(f,4) > best_face(3)*best_face(4)
                best_face = faces(f,:);
            end
        end
        % Save best face to be cropped,
        faces = best_face;
    end

    % Now crop the image if a face has been found
    if ~isempty(faces)
        I = imcrop(I, faces);
    else
        % If a face isn't found, will have to do our best - crop the image
        % borders, assuming the face is roughly centered.
        I = imcrop(I, [45 45 (600-45) (600-45)]);
    end

    % Resize the image to the facenet input size.
    I = imresize(I, [160 160]); 

   % Standardize the image
   processed_image =(double(I) - mean(I, [1 2]))/std2(I);
   
end


end
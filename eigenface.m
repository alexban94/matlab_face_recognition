function  output_label = eigenface(trainPath, testPath)
%% A face recognition method using the Eigenface approach.




%% Retrieve training images and labels
folder_names = ls(trainPath);
train_image_set = zeros(600,600,3,length(folder_names) - 2); % all images are 3 channels with size of 600x600
label_image_set = folder_names(3:end,:); % the folder names are the labels
for i = 3:length(folder_names)
    image_name = ls([trainPath, folder_names(i,:),'\*.jpg']);
    train_image_set(:,:,:,i-2)= imread([trainPath, folder_names(i,:), '\', image_name]);
end


%% Pre-process each image and store in vector form.
I = [];
for i = 1:size(train_image_set, 4)
    image_vector = pre_process_image(train_image_set(:,:,:,i));
    
    I = [I image_vector]; 
end

%% Step 1: Computation of the eigenfaces.

% % Represent every image I_i as a vector, in matrix I.
[h, w] = size(I);
% Now every image is contained in I, each as a (h*w) by 1 vector.
% Compute mean face vector psi, by averaging by row.
psi = mean(I, 2);

% Subtract mean face from each image in I.
A = bsxfun(@minus, I, psi); % Expands psi to matrix I size, so can subtract from all columns at once.

% Computing covariance matrix is not possible, as A*A' is too large (N^2 *
% N^2)
% Compute the eigenvectors u_i of A*A'.

% Consider instead the matrix A'*A
% Compute eigenvectors v_i of A'*A
[V, mu] = eig(A'*A);

% V and mu are in reverse order - most significant towards the right.
V = fliplr(V); % Flip so most significant eigenvectors are left -> right.
mu = fliplr(mu);

% The first 3 eigenvectors contain the most illumination information,
% however dropping them now causes the accuracy to decrease, so too much
% information is lost by discarding them.
% V = V(:,4:end);
% AA' and A'A have the same eigenvalues mu, and their eigenvectors are
% related as u_i = A*v_i

% The M eigenvalues retrieved from A'A correspond to the M largest
% eigenvalues of AA' (along with eigenvectors).

% So compute the M best eigenvectors of AA', by u_i = A*v_i
% Can do it altogether as U = A*V

% Make sure to normalize such that ||u_i|| = 1, each column a unit vector.
U = normc(A * V);


% So U corresponds to the M best eigenvectors of AA'.
% Here we can do dimensionality reduction, keeping only the K eigenvectors,
% that correspond to the K largest eigenvalues.

% Now we need to project faces onto this basis.
% Each face, minus the mean (each column in A), can be represented as a
% linear combination of the K best eigenvectors.

% Each normalized training face (column in A) is represented in this basis
% by a vector of weights. So for M faces, there is a K x 1 vector
% (depending on how many eigenvectors are kept, if all are kept then K = M.

% Each u_j of U is an 'eigenface'.
W = [];
for(i = 1:w)
    w_i = U' * A(:,i);
    W = [W w_i];
end
% Again, make sure it is normalized as a unit vector.
W = normc(W);
% Now each w_j of W represents a set of weights representing a training
% face in this new basis, where W is K x M matrix. K being the number of
% eigenvectors kept, so one weight per eigenvector, and M being the number
% of faces.


%% Step 2: Face Recognition using Eigenfaces.
test_image_names = ls([testPath,'*.jpg']);


output_label = [];
for i=1:size(test_image_names,1)
    test_image = imread([testPath, test_image_names(i,:)]);
   
    % Perform the same pre-process as the train images
    test_vector = pre_process_image(test_image);
    
    % Normalize it by subtracting the mean image.
    phi = test_vector - psi;
    
    % Project it into the eigenspace, U' * phi, which gets the set of
    % weights (the K x 1 weight vector).
    test_W = U' * phi;
    test_W = normc(test_W); % Make sure it is normalized as with the others
    
    % Calculate distance between test weights, and each set of training
    % weights. Find the minimum.
    d = [];
    for j = 1:size(W,2)
        % Euclidean distance
        d_j = test_W - W(:,j);
        d = [d sqrt(d_j' * d_j)];
    end
   

    % Each distance in d, is the distance within the face space. The
    % minimum in d, is most likely to be the face from the training set. If
    % it is less than a given threshold T_r, then it is recognized as that
    % face. Need to define this threshold, but for now just take the min.
    [~, index] = min(d);
    output_label = [output_label; label_image_set(index, :)];
 
        
    
end

    % Function to preprocess an image for use with the eigenface method.
    % Returns image as a vector.
    function proc_image = pre_process_image(image)
       
        % Define a margin to crop the image by - 45 found to be good
        % performance after testing with a range of value.
        % Cropping is done to focus more on the face, and remove parts of
        % background and occlusions that appear on some test images.
        margin = 45;
        
        % Convert to grayscale
        image = rgb2gray(uint8(image));
       
        % Adjust contrast to ensure there's a wider range of pixel values
        image = imadjust(image, [0.3, 0.7]);
        
        % Crop image borders
        image = imcrop(image, [margin margin (600-margin) (600-margin)]);
        
        % Scale the image to increase computation speed.
        image = imresize(image, 0.25);
    
        % Normalize to the range 0 to 1 by dividing my 255.
        image = double(image)/255;
       
        % Standardize to have mean = 0 and s.d = 1
        image = (image - mean(image, [1 2]))/std2(image);
        
        % Conduct DoG (difference of gaussian) method on the image
        image = imgaussfilt(image,8) - imgaussfilt(image, 2);
        
        % Return in vector form as required by the eigenface method.
        proc_image = image(:);
    end


end





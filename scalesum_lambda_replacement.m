classdef scalesum_lambda_replacement < nnet.layer.Layer
    % Defined as input_0 + input_1 * scale,
    % Based on :
    % https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py
   
    % There are no learnable parameters, it simply performs this operation
    % to combine 2 inputs, and give 1 output.
    % MATLAB Deep Learning toolbox does not support lambda layers, so this
    % custom layer is defined to perform the same operation.
                    
       
    properties
        % Define scale property
        Scale;
    end

    methods
        function layer = scalesum_lambda_replacement(scale, numInputs,name) 
            % Constructor: define a 'scalesum' layer and specifies the
            % scale, the number of inputs and the layer name.
        
            % Set number of inputs.
            layer.NumInputs = numInputs;

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Scalesum with scale " + scale +  ... 
                " that was input";
            
            layer.Scale = scale;
        
           
        end
        
        % This method is used for prediction and a forward pass.
        function Z = predict(layer, varargin)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            X = varargin;
            
            % Initialize output
            X1 = X{1};
            sz = size(X1);
            Z = zeros(sz,'like',X1);
            
            % Apply the lambda operation.
            Z = X{1} + ( X{2} * layer.Scale);
        end
    end
end
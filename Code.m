%Element wise sigmoid function
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoidDir = @(x) sigmoid(x) .* (1 - sigmoid(x));

%Cost function using the squared error function
costDer = @(x,y) (x-y);


load("mnist.mat") % Loadint the MNIST dataset

%Input size is equal to pixel number
inputSize = training.width * training.height;

%Hidden layer perceptrons
hiddenSize = 15;

%Digits 0-9
outputSize = 10;

%Definig the weights for the neura network as noramly distributed weights
%and baises
WItoH = randn(hiddenSize,inputSize);
baisH = randn(hiddenSize, 1);
WHtoO = randn(outputSize,hiddenSize);
baisO = randn(outputSize, 1);

%Neural Netwoek parameters
numOfEpoch = 5;
learningRate = 3;
minibatchSize = 10;



%Training
for Epoch = 1:numOfEpoch
    
    %Randomize input data order
    miniBatch = [1:50000];
    miniBatch = miniBatch(randperm(length(miniBatch)));
    j = 0;

    %looping over the all minibatches
    while j < length(miniBatch)-100
        
        %Defining nabla for gradient descent
        nabla_bO = zeros(10,1);
        nabla_wO = zeros(10,15);
        nabla_bH = zeros(15,1);
        nabla_wH = zeros(15,784);
        
        %Training batch
        for i = miniBatch(j+1:j+minibatchSize)

            %Input image Vector
            inputimg = reshape(training.images(:,:,i),1,[])';

            %Target output activation
            targetO = zeros(1,10)';
            targetO(training.labels(i)+1) = 1;

            %Calculating the activations of the preceptors
            actiH = (WItoH * inputimg) + baisH;
            sigActiH = sigmoid(actiH);
            actiO = (WHtoO * sigActiH) + baisO;
            sigActiO = sigmoid(actiO);

            %Compute nabla for final layer
            deltaO = costDer(sigActiO,targetO) .* sigmoidDir(actiO);

            nabla_bO = nabla_bO + (deltaO);
            nabla_wO = nabla_wO + (deltaO * sigActiH');

            %Backpropogation
            deltaH = (WHtoO' * deltaO) .* sigmoidDir(actiH);

            nabla_bH = nabla_bH + (deltaH);
            nabla_wH = nabla_wH + (deltaH * inputimg');
        end

        j = j + minibatchSize;

        %learning step -> Updating the weights and biases
        WHtoO = WHtoO - ((learningRate/minibatchSize) .* nabla_wO);
        baisO = baisO - ((learningRate/minibatchSize) .* nabla_bO);

        WItoH = WItoH - ((learningRate/minibatchSize) .* nabla_wH);
        baisH = baisH - ((learningRate/minibatchSize) .* nabla_bH);

    end

    
    %Periodic test
    count = 0;
    for i = 50000:60000

        inputimg = reshape(training.images(:,:,i),1,[])';

        %Target output activation
        targetO = zeros(1,10)';
        targetO(training.labels(i)+1) = 1;

        %Calculating the activations of the preceptors
        actiH = WItoH * inputimg + baisH;
        sigActiH = sigmoid(actiH);
        actiO = WHtoO * sigActiH + baisO;
        sigActiO = sigmoid(actiO);
        
        %Testing the output of the neural network
        [~,indO] = max(sigActiO);
        [~,indT] = max(targetO);
        if(indO == indT)
            count = count + 1;
        end
    end


    fprintf("Epoch" + " " + Epoch + " / "  + numOfEpoch + "\n" + "correct = "+count + "/10000" + "\n")
end


testc = 0;

%Testing on a diffrent dataset

for i = 1:10000

    inputimg = reshape(test.images(:,:,i),1,[])';

    %Target output activation
    targetO = zeros(1,10)';
    targetO(test.labels(i)+1) = 1;

    %Calculating the activations of the preceptors
    actiH = WItoH * inputimg + baisH;
    sigActiH = sigmoid(actiH);
    actiO = WHtoO * sigActiH + baisO;
    sigActiO = sigmoid(actiO);
    %image(training.images(:,:,i) * 255)

    [~,indO] = max(sigActiO);
    [~,indT] = max(targetO);
    if(indO == indT)
        testc = testc + 1;
    end
end

fprintf("testing results = " + testc + "/10000\n");








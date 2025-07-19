% train_lstm_nasa_COMPLETE.m - COMPLETE VERSION FOR ALL BATTERY FILES
close all; clc;

fprintf('🚀 TRAINING LSTM WITH COMPLETE BATTERY DATASET...\n');

%% Step 1: Load Complete Dataset
try
    load('.\Outputs\nasa_training_data_COMPLETE.mat');
    fprintf('✅ Complete dataset loaded: %d sequences\n', length(XTrain));
catch
    error('❌ Run prepare_nasa_data_ALL.m first to create complete dataset!');
end

%% Step 2: Data Validation and Analysis
fprintf('📊 Analyzing complete dataset...\n');

% Analyze SOC distribution across all batteries
soc_values = zeros(length(YTrain), 1);
for i = 1:length(YTrain)
    soc_values(i) = double(YTrain{i});
end

fprintf('Complete Dataset Statistics:\n');
fprintf('  Total sequences: %d\n', length(XTrain));
fprintf('  SOC Min: %.3f\n', min(soc_values));
fprintf('  SOC Max: %.3f\n', max(soc_values));
fprintf('  SOC Mean: %.3f\n', mean(soc_values));
fprintf('  SOC Std: %.3f\n', std(soc_values));

% Verify data format
for i = 1:min(3, length(XTrain))
    [rows, cols] = size(XTrain{i});
    if rows ~= 3 || cols ~= 50
        error('❌ XTrain{%d} has wrong dimensions: %dx%d (expected 3x50)', i, rows, cols);
    end
end
fprintf('✅ Data format verification passed!\n');

%% Step 3: Convert Cell Arrays to Proper Format
fprintf('🔄 Converting data format for LSTM...\n');

XTrain_fixed = cell(size(XTrain));
YTrain_fixed = cell(size(YTrain));

for i = 1:length(XTrain)
    temp_matrix = double(XTrain{i});
    if size(temp_matrix, 1) == 3 && size(temp_matrix, 2) == 50
        XTrain_fixed{i} = temp_matrix;
    else
        error('❌ Sequence %d has wrong dimensions after conversion', i);
    end
    YTrain_fixed{i} = double(YTrain{i});
end
fprintf('✅ Data conversion completed\n');

%% Step 4: Split Data with Stratification
numSamples = length(XTrain_fixed);
numTrain = round(0.8 * numSamples);
numVal = numSamples - numTrain;

fprintf('🔄 Splitting complete dataset:\n');
fprintf('  Training samples: %d (%.1f%%)\n', numTrain, (numTrain/numSamples)*100);
fprintf('  Validation samples: %d (%.1f%%)\n', numVal, (numVal/numSamples)*100);

% Shuffle indices for better distribution
rng(42); % For reproducibility
shuffled_idx = randperm(numSamples);

train_idx = shuffled_idx(1:numTrain);
val_idx = shuffled_idx(numTrain+1:end);

% Split X data (keep as cell arrays for LSTM)
XTrainSet = XTrain_fixed(train_idx);
XValSet = XTrain_fixed(val_idx);

% Convert Y data to numeric arrays
YTrainSet = zeros(numTrain, 1);
YValSet = zeros(numVal, 1);

for i = 1:numTrain
    YTrainSet(i) = double(YTrain_fixed{train_idx(i)});
end

for i = 1:numVal
    YValSet(i) = double(YTrain_fixed{val_idx(i)});
end

fprintf('✅ Data splitting completed\n');
fprintf('  Training SOC range: %.3f to %.3f\n', min(YTrainSet), max(YTrainSet));
fprintf('  Validation SOC range: %.3f to %.3f\n', min(YValSet), max(YValSet));

%% Step 5: Enhanced Network Architecture for Large Dataset
fprintf('🏗️ Creating enhanced LSTM network...\n');

layers = [
    sequenceInputLayer(3, 'Name', 'input')
    lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm1')  % More neurons for complex data
    dropoutLayer(0.3, 'Name', 'dropout1')
    lstmLayer(64, 'OutputMode', 'last', 'Name', 'lstm2')   % Second LSTM layer
    dropoutLayer(0.2, 'Name', 'dropout2')
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'fc2')
    regressionLayer('Name', 'output')
];

% Training options optimized for large dataset
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 32, ...                 % Larger batches for big dataset
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'ValidationData', {XValSet, YValSet}, ...  % ← MISSING IN YOUR CODE
    'ValidationFrequency', 30, ...
    'ValidationPatience', 15, ...             % ← MISSING IN YOUR CODE
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...                      % ← MISSING IN YOUR CODE
    'ExecutionEnvironment', 'auto');

%% Step 6: Final Pre-Training Verification
fprintf('\n=== FINAL PRE-TRAINING CHECK ===\n');
fprintf('Network layers: %d\n', length(layers));
fprintf('Training samples: %d\n', length(XTrainSet));
fprintf('Validation samples: %d\n', length(XValSet));
fprintf('Sample input size: %d x %d\n', size(XTrainSet{1}));
fprintf('Sample output: %.4f\n', YTrainSet(1));

%% Step 7: Train the Network
fprintf('\n🚀 Starting LSTM training with complete battery dataset...\n');
fprintf('⏰ This may take 30-60 minutes with large dataset...\n');

try
    % ← THIS WAS COMPLETELY MISSING IN YOUR CODE
    tic; % Start timing
    recurrentNet = trainNetwork(XTrainSet, YTrainSet, layers, options);
    training_time = toc;
    
    fprintf('✅ Training completed successfully!\n');
    fprintf('⏰ Training time: %.1f minutes\n', training_time/60);
    
    % Save the trained model
    save('.\Outputs\nasa_lstmSOCNet_COMPLETE.mat', 'recurrentNet', 'training_time');
    fprintf('✅ Network saved to nasa_lstmSOCNet_COMPLETE.mat\n');
    
catch ME
    fprintf('❌ Training failed: %s\n', ME.message);
    fprintf('💡 Try reducing MiniBatchSize or MaxEpochs if memory issues\n');
    rethrow(ME);
end

%% Step 8: Performance Evaluation
fprintf('\n📊 Evaluating model performance...\n');

try
    % Predict on validation set
    YPred = predict(recurrentNet, XValSet);
    
    % Calculate performance metrics
    mse = mean((YValSet - YPred).^2);
    rmse = sqrt(mse);
    mae = mean(abs(YValSet - YPred));
    r2 = 1 - sum((YValSet - YPred).^2) / sum((YValSet - mean(YValSet)).^2);
    
    fprintf('🎯 FINAL PERFORMANCE METRICS:\n');
    fprintf('  RMSE: %.6f\n', rmse);
    fprintf('  MAE:  %.6f\n', mae);
    fprintf('  MSE:  %.6f\n', mse);
    fprintf('  R²:   %.4f\n', r2);
    
    % Create performance visualization
    figure('Name', 'Complete Dataset SOC Prediction Results', 'Position', [100 100 1000 400]);
    
    % Subplot 1: Prediction scatter plot
    subplot(1,2,1);
    scatter(YValSet, YPred, 'filled', 'Alpha', 0.6);
    hold on;
    plot([min(YValSet), max(YValSet)], [min(YValSet), max(YValSet)], 'r--', 'LineWidth', 2);
    xlabel('Actual SOC');
    ylabel('Predicted SOC');
    title(sprintf('SOC Prediction (R² = %.4f)', r2));
    grid on;
    legend('Predictions', 'Perfect Prediction', 'Location', 'best');
    axis equal;
    
    % Subplot 2: Error histogram
    subplot(1,2,2);
    errors = YValSet - YPred;
    histogram(errors, 50, 'FaceAlpha', 0.7);
    xlabel('Prediction Error');
    ylabel('Frequency');
    title(sprintf('Error Distribution (RMSE = %.6f)', rmse));
    grid on;
    
    % Save performance metrics
    performance = struct('RMSE', rmse, 'MAE', mae, 'MSE', mse, 'R2', r2, ...
                        'training_time', training_time);
    save('model_performance_COMPLETE.mat', 'performance', 'YValSet', 'YPred');
    
catch ME
    fprintf('⚠️ Performance evaluation failed: %s\n', ME.message);
end

fprintf('\n🎉 COMPLETE LSTM TRAINING FINISHED!\n');
fprintf('📁 Model saved as: nasa_lstmSOCNet_COMPLETE.mat\n');
fprintf('📊 Performance saved as: model_performance_COMPLETE.mat\n');
fprintf('🔧 Ready for Simulink integration!\n');

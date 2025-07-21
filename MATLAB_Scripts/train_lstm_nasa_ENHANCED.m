% train_lstm_nasa_ENHANCED.m - UPDATED FOR REAL SOC DATA
close all; clc;
fprintf('🚀 TRAINING LSTM WITH ENHANCED REAL SOC DATASET...\n');

%% Step 1: Load Enhanced Dataset
try
    load('..\Outputs\nasa_training_data_ENHANCED.mat');
    fprintf('✅ Enhanced dataset loaded: %d sequences\n', length(XTrain));
    
    % Display what SOC methods were used
    if exist('soc_methods_tried', 'var')
        fprintf('📊 SOC calculation methods used: %s\n', strjoin(soc_methods_tried, ', '));
    end
    if exist('prediction_horizon', 'var')
        fprintf('🎯 Prediction horizon: %d steps ahead\n', prediction_horizon);
    end
    
catch
    fprintf('⚠️ Enhanced dataset not found, trying original...\n');
    try
        load('..\Outputs\nasa_training_data_COMPLETE.mat');
        fprintf('⚠️ Using original dataset (synthetic SOC)\n');
        soc_methods_tried = {'Synthetic Linear SOC'};
        prediction_horizon = 0;
    catch
        error('❌ Run prepare_nasa_data_ENHANCED.m first to create enhanced dataset!');
    end
end

%% Step 2: Enhanced Data Validation and Analysis
fprintf('📊 Analyzing enhanced dataset...\n');

% Extract all SOC values for analysis
soc_values = zeros(length(YTrain), 1);
for i = 1:length(YTrain)
    soc_values(i) = double(YTrain{i});
end

% Enhanced dataset statistics
fprintf('Enhanced Dataset Statistics:\n');
fprintf('  Total sequences: %d\n', length(XTrain));
fprintf('  SOC Min: %.4f (%.1f%%)\n', min(soc_values), min(soc_values)*100);
fprintf('  SOC Max: %.4f (%.1f%%)\n', max(soc_values), max(soc_values)*100);
fprintf('  SOC Mean: %.4f (%.1f%%)\n', mean(soc_values), mean(soc_values)*100);
fprintf('  SOC Std: %.4f\n', std(soc_values));
fprintf('  SOC Range: %.4f (%.1f%% span)\n', max(soc_values)-min(soc_values), (max(soc_values)-min(soc_values))*100);

% Check for realistic SOC range
if min(soc_values) < 0 || max(soc_values) > 1
    warning('⚠️ SOC values outside 0-1 range detected! Check data preparation.');
end

% Analyze SOC distribution
fprintf('📈 SOC Distribution Analysis:\n');
soc_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
soc_counts = histcounts(soc_values, soc_bins);
bin_labels = {'0-20%', '20-40%', '40-60%', '60-80%', '80-100%'};
for i = 1:length(soc_counts)
    fprintf('  %s: %d sequences (%.1f%%)\n', bin_labels{i}, soc_counts(i), (soc_counts(i)/length(soc_values))*100);
end

% Verify data format
fprintf('🔍 Validating data format...\n');
for i = 1:min(3, length(XTrain))
    [rows, cols] = size(XTrain{i});
    if rows ~= 3 || cols ~= 50
        error('❌ XTrain{%d} has wrong dimensions: %dx%d (expected 3x50)', i, rows, cols);
    end
end
fprintf('✅ Data format verification passed!\n');

%% Step 3: Enhanced Data Preprocessing
fprintf('🔄 Enhanced data preprocessing...\n');

% Remove any invalid sequences (NaN or out-of-range SOC)
valid_indices = [];
for i = 1:length(YTrain)
    soc_val = double(YTrain{i});
    if ~isnan(soc_val) && soc_val >= 0 && soc_val <= 1
        valid_indices = [valid_indices, i];
    end
end

if length(valid_indices) < length(YTrain)
    fprintf('⚠️ Removed %d invalid sequences (%.1f%%)\n', ...
            length(YTrain) - length(valid_indices), ...
            ((length(YTrain) - length(valid_indices))/length(YTrain))*100);
    
    % Keep only valid sequences
    XTrain_clean = XTrain(valid_indices);
    YTrain_clean = YTrain(valid_indices);
else
    XTrain_clean = XTrain;
    YTrain_clean = YTrain;
end

% Convert to proper format
XTrain_fixed = cell(size(XTrain_clean));
YTrain_fixed = cell(size(YTrain_clean));

for i = 1:length(XTrain_clean)
    XTrain_fixed{i} = double(XTrain_clean{i});
    YTrain_fixed{i} = double(YTrain_clean{i});
end

fprintf('✅ Data preprocessing completed: %d valid sequences\n', length(XTrain_fixed));

%% Step 4: Enhanced Train/Validation Split
numSamples = length(XTrain_fixed);
numTrain = round(0.8 * numSamples);
numVal = numSamples - numTrain;

fprintf('🔄 Splitting enhanced dataset:\n');
fprintf('  Training samples: %d (%.1f%%)\n', numTrain, (numTrain/numSamples)*100);
fprintf('  Validation samples: %d (%.1f%%)\n', numVal, (numVal/numSamples)*100);

% Stratified splitting based on SOC ranges for better representation
rng(42); % For reproducibility

% Create SOC-based strata
all_soc = zeros(numSamples, 1);
for i = 1:numSamples
    all_soc(i) = YTrain_fixed{i};
end

% Divide into SOC bins for stratification
num_strata = 5;
soc_edges = linspace(min(all_soc), max(all_soc), num_strata + 1);
[~, soc_bins] = histc(all_soc, soc_edges);

% Stratified sampling
train_idx = [];
val_idx = [];

for stratum = 1:num_strata
    stratum_indices = find(soc_bins == stratum);
    n_stratum = length(stratum_indices);
    n_train_stratum = round(0.8 * n_stratum);
    
    if n_stratum > 0
        shuffled_stratum = stratum_indices(randperm(n_stratum));
        train_idx = [train_idx; shuffled_stratum(1:n_train_stratum)];
        val_idx = [val_idx; shuffled_stratum(n_train_stratum+1:end)];
    end
end

% Final data split
XTrainSet = XTrain_fixed(train_idx);
XValSet = XTrain_fixed(val_idx);

YTrainSet = zeros(length(train_idx), 1);
YValSet = zeros(length(val_idx), 1);

for i = 1:length(train_idx)
    YTrainSet(i) = YTrain_fixed{train_idx(i)};
end
for i = 1:length(val_idx)
    YValSet(i) = YTrain_fixed{val_idx(i)};
end

fprintf('✅ Stratified splitting completed\n');
fprintf('  Training SOC - Min: %.4f, Max: %.4f, Mean: %.4f\n', min(YTrainSet), max(YTrainSet), mean(YTrainSet));
fprintf('  Validation SOC - Min: %.4f, Max: %.4f, Mean: %.4f\n', min(YValSet), max(YValSet), mean(YValSet));

%% Step 5: Enhanced Network Architecture for Real SOC Data
fprintf('🏗️ Creating enhanced LSTM network for real SOC prediction...\n');

% Determine network complexity based on dataset size and SOC method
if numSamples > 50000
    lstm1_units = 256;
    lstm2_units = 128;
    fc_units = 64;
    fprintf('📊 Using large network architecture for %d samples\n', numSamples);
elseif numSamples > 20000
    lstm1_units = 128;
    lstm2_units = 64;
    fc_units = 32;
    fprintf('📊 Using medium network architecture for %d samples\n', numSamples);
else
    lstm1_units = 64;
    lstm2_units = 32;
    fc_units = 16;
    fprintf('📊 Using compact network architecture for %d samples\n', numSamples);
end

layers = [
    sequenceInputLayer(3, 'Name', 'input')
    lstmLayer(lstm1_units, 'OutputMode', 'last', 'Name', 'lstm1')
    dropoutLayer(0.3, 'Name', 'dropout1')
    lstmLayer(lstm2_units, 'OutputMode', 'last', 'Name', 'lstm2')
    dropoutLayer(0.2, 'Name', 'dropout2')
    fullyConnectedLayer(fc_units, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.1, 'Name', 'dropout3')  % Additional regularization
    fullyConnectedLayer(1, 'Name', 'fc2')
    regressionLayer('Name', 'output')
];

% Enhanced training options based on real SOC data characteristics
if contains(strjoin(soc_methods_tried), 'Direct SOC') || contains(strjoin(soc_methods_tried), 'capacity')
    % Real data is more accurate, can use higher learning rate
    initial_lr = 0.002;
    max_epochs = 100;
    fprintf('📈 Using optimized settings for high-quality SOC data\n');
else
    % Voltage-based or estimated SOC, more conservative training
    initial_lr = 0.001;
    max_epochs = 150;
    fprintf('📉 Using conservative settings for estimated SOC data\n');
end

% Adjust batch size based on dataset size
if numSamples > 30000
    batch_size = 64;
elseif numSamples > 10000
    batch_size = 32;
else
    batch_size = 16;
end

options = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', batch_size, ...
    'InitialLearnRate', initial_lr, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', round(max_epochs/3), ...
    'ValidationData', {XValSet, YValSet}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 20, ...  % Increased patience for real data
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto', ...
    'GradientThreshold', 1);  % Prevent gradient explosion with real data

%% Step 6: Enhanced Pre-Training Verification
fprintf('\n=== ENHANCED PRE-TRAINING CHECK ===\n');
fprintf('Network architecture: %d LSTM units → %d LSTM units → %d FC units\n', lstm1_units, lstm2_units, fc_units);
fprintf('Training samples: %d\n', length(XTrainSet));
fprintf('Validation samples: %d\n', length(XValSet));
fprintf('Batch size: %d\n', batch_size);
fprintf('Initial learning rate: %.4f\n', initial_lr);
fprintf('Max epochs: %d\n', max_epochs);
fprintf('Sample input size: %d x %d\n', size(XTrainSet{1}));
fprintf('Sample output range: %.4f to %.4f\n', min(YTrainSet), max(YTrainSet));
fprintf('Prediction type: %d step(s) ahead\n', prediction_horizon);

%% Step 7: Train the Enhanced Network
fprintf('\n🚀 Starting LSTM training with REAL SOC data...\n');
fprintf('⏰ Estimated training time: %.1f minutes\n', (numSamples * max_epochs) / (batch_size * 1000));

try
    tic; % Start timing
    recurrentNet = trainNetwork(XTrainSet, YTrainSet, layers, options);
    training_time = toc;
    
    fprintf('✅ Enhanced training completed successfully!\n');
    fprintf('⏰ Actual training time: %.1f minutes\n', training_time/60);
    
    % Save enhanced model with metadata
    model_metadata = struct(...
        'soc_methods_used', {soc_methods_tried}, ...
        'prediction_horizon', prediction_horizon, ...
        'training_time', training_time, ...
        'network_architecture', sprintf('%d-%d-%d', lstm1_units, lstm2_units, fc_units), ...
        'training_samples', numSamples, ...
        'data_version', 'ENHANCED_REAL_SOC');
    
    save('.\Outputs\nasa_lstmSOCNet_ENHANCED.mat', 'recurrentNet', 'model_metadata');
    fprintf('✅ Enhanced network saved to nasa_lstmSOCNet_ENHANCED.mat\n');
    
catch ME
    fprintf('❌ Training failed: %s\n', ME.message);
    fprintf('💡 Troubleshooting suggestions:\n');
    fprintf('   - Reduce batch size if memory issues\n');
    fprintf('   - Check for NaN values in SOC data\n');
    fprintf('   - Verify SOC range is 0-1\n');
    rethrow(ME);
end

%% Step 8: Enhanced Performance Evaluation
fprintf('\n📊 Evaluating enhanced model performance...\n');

try
    % Predict on validation set
    YPred = predict(recurrentNet, XValSet);
    
    % Enhanced performance metrics
    mse = mean((YValSet - YPred).^2);
    rmse = sqrt(mse);
    mae = mean(abs(YValSet - YPred));
    r2 = 1 - sum((YValSet - YPred).^2) / sum((YValSet - mean(YValSet)).^2);
    
    % Additional metrics for SOC prediction
    mape = mean(abs((YValSet - YPred) ./ (YValSet + eps))) * 100; % Mean Absolute Percentage Error
    max_error = max(abs(YValSet - YPred));
    
    fprintf('🎯 ENHANCED PERFORMANCE METRICS:\n');
    fprintf('  RMSE: %.6f (%.3f%% SOC)\n', rmse, rmse*100);
    fprintf('  MAE:  %.6f (%.3f%% SOC)\n', mae, mae*100);
    fprintf('  MAPE: %.2f%%\n', mape);
    fprintf('  Max Error: %.6f (%.3f%% SOC)\n', max_error, max_error*100);
    fprintf('  R²: %.4f\n', r2);
    
    % Performance quality assessment
    if r2 > 0.95
        fprintf('🏆 EXCELLENT: Model shows excellent predictive performance!\n');
    elseif r2 > 0.90
        fprintf('✅ GOOD: Model shows good predictive performance\n');
    elseif r2 > 0.80
        fprintf('⚠️ FAIR: Model performance is acceptable but could be improved\n');
    else
        fprintf('❌ POOR: Model performance needs improvement\n');
    end
    
    % Enhanced visualization
    figure('Name', 'Enhanced SOC Prediction Results', 'Position', [100 100 1200 800]);
    
    % Subplot 1: Prediction scatter plot
    subplot(2,2,1);
    scatter(YValSet*100, YPred*100, 'filled', 'Alpha', 0.6);
    hold on;
    plot([min(YValSet)*100, max(YValSet)*100], [min(YValSet)*100, max(YValSet)*100], 'r--', 'LineWidth', 2);
    xlabel('Actual SOC (%)');
    ylabel('Predicted SOC (%)');
    title(sprintf('Enhanced SOC Prediction (R² = %.4f)', r2));
    grid on;
    legend('Predictions', 'Perfect Prediction', 'Location', 'best');
    axis equal;
    
    % Subplot 2: Error histogram
    subplot(2,2,2);
    errors = (YValSet - YPred) * 100;
    histogram(errors, 50, 'FaceAlpha', 0.7);
    xlabel('Prediction Error (% SOC)');
    ylabel('Frequency');
    title(sprintf('Error Distribution (RMSE = %.3f%%)', rmse*100));
    grid on;
    
    % Subplot 3: Error vs Actual SOC
    subplot(2,2,3);
    scatter(YValSet*100, errors, 'filled', 'Alpha', 0.6);
    xlabel('Actual SOC (%)');
    ylabel('Prediction Error (% SOC)');
    title('Error vs Actual SOC');
    grid on;
    yline(0, 'r--', 'LineWidth', 2);
    
    % Subplot 4: Performance by SOC range
    subplot(2,2,4);
    soc_ranges = {'0-20%', '20-40%', '40-60%', '60-80%', '80-100%'};
    range_rmse = zeros(5, 1);
    range_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    
    for i = 1:5
        in_range = YValSet >= range_edges(i) & YValSet < range_edges(i+1);
        if sum(in_range) > 0
            range_rmse(i) = sqrt(mean((YValSet(in_range) - YPred(in_range)).^2)) * 100;
        end
    end
    
    bar(range_rmse);
    set(gca, 'XTickLabel', soc_ranges);
    ylabel('RMSE (% SOC)');
    title('Performance by SOC Range');
    grid on;
    
    % Save enhanced performance data
    performance = struct(...
        'RMSE', rmse, 'MAE', mae, 'MAPE', mape, 'R2', r2, ...
        'max_error', max_error, 'training_time', training_time, ...
        'soc_methods', {soc_methods_tried}, 'prediction_horizon', prediction_horizon, ...
        'network_architecture', sprintf('%d-%d-%d', lstm1_units, lstm2_units, fc_units));
    
    save('..\Outputs\model_performance_ENHANCED.mat', 'performance', 'YValSet', 'YPred', 'model_metadata');
    
catch ME
    fprintf('⚠️ Performance evaluation failed: %s\n', ME.message);
end

fprintf('\n🎉 ENHANCED LSTM TRAINING COMPLETED!\n');
fprintf('📁 Enhanced model: nasa_lstmSOCNet_ENHANCED.mat\n');
fprintf('📊 Performance data: model_performance_ENHANCED.mat\n');
fprintf('🔧 Ready for advanced Simulink integration with real SOC prediction!\n');

% Display final summary
fprintf('\n📋 TRAINING SUMMARY:\n');
fprintf('  Dataset: %d sequences with real SOC data\n', numSamples);
fprintf('  SOC calculation methods: %s\n', strjoin(soc_methods_tried, ', '));
fprintf('  Prediction horizon: %d step(s) ahead\n', prediction_horizon);
fprintf('  Final R²: %.4f\n', r2);
fprintf('  Training time: %.1f minutes\n', training_time/60);

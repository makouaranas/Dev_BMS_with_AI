% prepare_nasa_data_ALL.m - USE ALL BATTERY FILES
 clc;

fprintf('🚀 PREPARING NASA DATA FROM ALL BATTERY FILES...\n');

% Define your data directory
data_dir = '.\Data\';

% Find all .mat files in your dataset
mat_files = dir(fullfile(data_dir, 'B*.mat'));
fprintf('Found %d battery files to process\n', length(mat_files));

% Initialize global storage
XTrain = {};
YTrain = {};
global_sequence_count = 0;
sequenceLength = 50;

% Process each battery file
for file_idx = 1:length(mat_files)
    filename = mat_files(file_idx).name;
    filepath = fullfile(data_dir, filename);
    
    fprintf('\n📁 Processing file %d/%d: %s\n', file_idx, length(mat_files), filename);
    
    try
        % Load current battery file
        battery_data = load(filepath);
        
        % Get the variable name (should be like B0005, B0006, etc.)
        fields = fieldnames(battery_data);
        battery_var = battery_data.(fields{1});
        
        % Check if this battery has cycle data
        if ~isfield(battery_var, 'cycle')
            fprintf('⚠️ Skipping %s - no cycle data found\n', filename);
            continue;
        end
        
        % Find all discharge and charge cycles for this battery
        discharge_indices = [];
        charge_indices = [];
        
        for i = 1:length(battery_var.cycle)
            try
                if strcmp(battery_var.cycle(i).type, 'discharge')
                    discharge_indices = [discharge_indices, i];
                elseif strcmp(battery_var.cycle(i).type, 'charge')
                    charge_indices = [charge_indices, i];
                end
            catch
                % Skip cycles with missing type field
                continue;
            end
        end
        
        all_cycle_indices = [discharge_indices, charge_indices];
        cycle_types = [repmat({'discharge'}, 1, length(discharge_indices)), ...
                       repmat({'charge'}, 1, length(charge_indices))];
        
        fprintf('   Found %d total cycles (%d discharge, %d charge)\n', ...
                length(all_cycle_indices), length(discharge_indices), length(charge_indices));
        
        % Process all cycles in this battery
        battery_sequences = 0;
        
        for cycle_num = 1:length(all_cycle_indices)
            cycle_idx = all_cycle_indices(cycle_num);
            cycle_type = cycle_types{cycle_num};
            
            try
                data = battery_var.cycle(cycle_idx).data;
                voltage = double(data.Voltage_measured(:));
                current = double(data.Current_measured(:));
                temperature = double(data.Temperature_measured(:));
                capacity = double(data.Capacity);
                
                if length(voltage) < sequenceLength
                    continue;
                end
                
                % Normalize data
                voltage_norm = (voltage - min(voltage)) / (max(voltage) - min(voltage) + eps);
                current_norm = (current - min(current)) / (max(current) - min(current) + eps);
                temp_norm = (temperature - min(temperature)) / (max(temperature) - min(temperature) + eps);
                
                % SOC calculation
                if strcmp(cycle_type, 'discharge')
                    soc_values = linspace(1.0, capacity/2.0, length(voltage));
                else
                    soc_values = linspace(capacity/2.0, 1.0, length(voltage));
                end
                
                % Create sequences (use every 10th point for more data)
                for j = 1:10:length(voltage)-sequenceLength+1
                    global_sequence_count = global_sequence_count + 1;
                    battery_sequences = battery_sequences + 1;
                    
                    seq_start = j;
                    seq_end = j + sequenceLength - 1;
                    
                    % Build sequence matrix
                    sequence_matrix = zeros(3, sequenceLength);
                    sequence_matrix(1, :) = voltage_norm(seq_start:seq_end);
                    sequence_matrix(2, :) = current_norm(seq_start:seq_end);
                    sequence_matrix(3, :) = temp_norm(seq_start:seq_end);
                    %best approch
                    %target_soc_sequence = soc_values(seq_start:seq_end);  % 50 SOC values
                    %YTrain{global_sequence_count} = target_soc_sequence;
                    
                    target_soc = soc_values(seq_end);
                    
                    XTrain{global_sequence_count} = sequence_matrix;
                    YTrain{global_sequence_count} = target_soc;
                end
                
            catch ME
                fprintf('   ⚠️ Error processing cycle %d: %s\n', cycle_idx, ME.message);
                continue;
            end
        end
        
        fprintf('   ✅ Created %d sequences from %s\n', battery_sequences, filename);
        
    catch ME
        fprintf('   ❌ Error loading %s: %s\n', filename, ME.message);
        continue;
    end
end

% Final verification and save
fprintf('\n🎯 DATASET SUMMARY:\n');
fprintf('Total battery files processed: %d\n', length(mat_files));
fprintf('Total sequences created: %d\n', global_sequence_count);

if global_sequence_count > 0
    [test_rows, test_cols] = size(XTrain{1});
    fprintf('Sample dimensions: %d x %d\n', test_rows, test_cols);
    
    if test_rows == 3 && test_cols == 50
        fprintf('✅ SUCCESS! Complete dataset processed!\n');
        save('.\Outputs\nasa_training_data_COMPLETE.mat', 'XTrain', 'YTrain', '-v7.3');
        fprintf('✅ Saved to nasa_training_data_COMPLETE.mat\n');
        fprintf('📊 File size: Large dataset (using -v7.3 format)\n');
    else
        error('❌ Wrong dimensions: %dx%d', test_rows, test_cols);
    end
else
    error('❌ No sequences created from any files!');
end

fprintf('\n🚀 Ready for comprehensive LSTM training!\n');

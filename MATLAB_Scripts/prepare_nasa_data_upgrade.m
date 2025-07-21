% prepare_nasa_data_ENHANCED.m - IMPROVED VERSION WITH REAL SOC
clc;
fprintf('🚀 PREPARING NASA DATA WITH REAL SOC CALCULATIONS...\n');

% Define your data directory
data_dir = '..\Data\';
% Find all .mat files in your dataset
mat_files = dir(fullfile(data_dir, 'B*.mat'));
fprintf('Found %d battery files to process\n', length(mat_files));

% Initialize global storage
XTrain = {};
YTrain = {};
global_sequence_count = 0;
sequenceLength = 50;
prediction_horizon = 1;  % Predict 1 step ahead

% SOC calculation method preference (will be auto-detected)
soc_methods_tried = {};

fprintf('\n🔍 ANALYZING DATASET STRUCTURE...\n');

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
        
        % Analyze first cycle to understand data structure (only for first file)
        if file_idx == 1
            fprintf('🔬 ANALYZING DATA STRUCTURE FROM FIRST FILE...\n');
            sample_cycle_data = battery_var.cycle(1).data;
            available_fields = fieldnames(sample_cycle_data);
            
            fprintf('   Available fields in cycle data:\n');
            for i = 1:length(available_fields)
                fprintf('      - %s\n', available_fields{i});
            end
            
            % Check for SOC-related fields
            has_direct_soc = isfield(sample_cycle_data, 'SOC') || isfield(sample_cycle_data, 'soc');
            has_discharge_cap = isfield(sample_cycle_data, 'Discharge_capacity');
            has_charge_cap = isfield(sample_cycle_data, 'Charge_capacity');
            has_capacity = isfield(sample_cycle_data, 'Capacity');
            has_time = isfield(sample_cycle_data, 'Time');
            
            fprintf('\n   SOC Data Availability Analysis:\n');
            fprintf('      Direct SOC field: %s\n', yesno(has_direct_soc));
            fprintf('      Discharge_capacity: %s\n', yesno(has_discharge_cap));
            fprintf('      Charge_capacity: %s\n', yesno(has_charge_cap));
            fprintf('      Capacity: %s\n', yesno(has_capacity));
            fprintf('      Time: %s\n', yesno(has_time));
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
                
                if length(voltage) < sequenceLength + prediction_horizon
                    continue;  % Need extra points for prediction horizon
                end
                
                % ===============================================
                % IMPROVED SOC CALCULATION - USE REAL DATA
                % ===============================================
                
                [real_soc, soc_method_used] = calculate_real_soc(data, cycle_type, battery_var);
                
                if isempty(real_soc)
                    fprintf('   ⚠️ Could not calculate SOC for cycle %d, skipping\n', cycle_idx);
                    continue;
                end
                
                % Track which method was used (for reporting)
                if ~ismember(soc_method_used, soc_methods_tried)
                    soc_methods_tried{end+1} = soc_method_used;
                    fprintf('   📊 Using SOC method: %s\n', soc_method_used);
                end
                
                % Normalize data (same as before)
                voltage_norm = (voltage - min(voltage)) / (max(voltage) - min(voltage) + eps);
                current_norm = (current - min(current)) / (max(current) - min(current) + eps);
                temp_norm = (temperature - min(temperature)) / (max(temperature) - min(temperature) + eps);
                
                % ===============================================
                % IMPROVED SEQUENCE CREATION WITH REAL SOC
                % ===============================================
                
                % Create sequences with proper SOC prediction
                max_start_idx = length(voltage) - sequenceLength - prediction_horizon + 1;
                
                for j = 1:10:max_start_idx
                    global_sequence_count = global_sequence_count + 1;
                    battery_sequences = battery_sequences + 1;
                    
                    seq_start = j;
                    seq_end = j + sequenceLength - 1;
                    target_idx = seq_end + prediction_horizon;  % Predict future SOC
                    
                    % Build sequence matrix (input features)
                    sequence_matrix = zeros(3, sequenceLength);
                    sequence_matrix(1, :) = voltage_norm(seq_start:seq_end);
                    sequence_matrix(2, :) = current_norm(seq_start:seq_end);
                    sequence_matrix(3, :) = temp_norm(seq_start:seq_end);
                    
                    % Use REAL SOC as target (predict future value)
                    target_soc = real_soc(target_idx);
                    
                    % Validate SOC value
                    if isnan(target_soc) || target_soc < 0 || target_soc > 1
                        global_sequence_count = global_sequence_count - 1;
                        battery_sequences = battery_sequences - 1;
                        continue;
                    end
                    
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
fprintf('\n🎯 ENHANCED DATASET SUMMARY:\n');
fprintf('Total battery files processed: %d\n', length(mat_files));
fprintf('Total sequences created: %d\n', global_sequence_count);
fprintf('SOC calculation methods used: %s\n', strjoin(soc_methods_tried, ', '));
fprintf('Prediction horizon: %d steps ahead\n', prediction_horizon);

if global_sequence_count > 0
    [test_rows, test_cols] = size(XTrain{1});
    fprintf('Sample dimensions: %d x %d\n', test_rows, test_cols);
    
    if test_rows == 3 && test_cols == 50
        fprintf('✅ SUCCESS! Enhanced dataset processed with REAL SOC!\n');
        save('..\Outputs\nasa_training_data_ENHANCED.mat', 'XTrain', 'YTrain', ...
             'soc_methods_tried', 'prediction_horizon', '-v7.3');
        fprintf('✅ Saved to nasa_training_data_ENHANCED.mat\n');
        
        % Display SOC statistics
        all_soc_values = [YTrain{:}];
        fprintf('📊 SOC Statistics:\n');
        fprintf('   Min SOC: %.3f\n', min(all_soc_values));
        fprintf('   Max SOC: %.3f\n', max(all_soc_values));
        fprintf('   Mean SOC: %.3f\n', mean(all_soc_values));
        fprintf('   SOC Range: %.3f\n', max(all_soc_values) - min(all_soc_values));
        
    else
        error('❌ Wrong dimensions: %dx%d', test_rows, test_cols);
    end
else
    error('❌ No sequences created from any files!');
end

fprintf('\n🚀 Ready for LSTM training with REAL SOC data!\n');

% ===============================================
%  FUNCTIONS
% ===============================================

function [soc_values, method_used] = calculate_real_soc(data, cycle_type, battery_var)
    % Calculate real SOC using the best available method
    soc_values = [];
    method_used = '';
    
    % Method 1: Direct SOC field (best if available)
    if isfield(data, 'SOC')
        soc_values = double(data.SOC(:));
        method_used = 'Direct SOC field';
        return;
    elseif isfield(data, 'soc')
        soc_values = double(data.soc(:));
        method_used = 'Direct soc field';
        return;
    end
    
    % Method 2: Use Discharge_capacity or Charge_capacity
    if isfield(data, 'Discharge_capacity') && strcmp(cycle_type, 'discharge')
        discharge_cap = double(data.Discharge_capacity(:));
        % SOC = 1 - (discharged_capacity / total_capacity)
        total_capacity = max(discharge_cap);
        if total_capacity > 0
            soc_values = 1 - (discharge_cap / total_capacity);
            method_used = 'Discharge_capacity calculation';
            return;
        end
    elseif isfield(data, 'Charge_capacity') && strcmp(cycle_type, 'charge')
        charge_cap = double(data.Charge_capacity(:));
        % SOC increases during charging
        total_capacity = max(charge_cap);
        if total_capacity > 0
            soc_values = charge_cap / total_capacity;
            method_used = 'Charge_capacity calculation';
            return;
        end
    end
    
    % Method 3: Coulomb counting (most complex but most accurate)
    if isfield(data, 'Time') && isfield(data, 'Current_measured')
        try
            time_data = double(data.Time(:));
            current_data = double(data.Current_measured(:));
            
            % Calculate time differences
            time_diff = diff([time_data(1); time_data]);  % seconds
            
            % Estimate nominal capacity (from battery specs or data analysis)
            nominal_capacity_ah = estimate_nominal_capacity(battery_var);
            
            if nominal_capacity_ah > 0
                if strcmp(cycle_type, 'discharge')
                    % During discharge, current is typically negative
                    charge_consumed = cumsum(abs(current_data) .* time_diff / 3600);  % Ah
                    soc_values = max(0, 1 - (charge_consumed / nominal_capacity_ah));
                else
                    % During charge, current is typically positive
                    charge_added = cumsum(max(0, current_data) .* time_diff / 3600);  % Ah
                    initial_soc = 0.1;  % Assume charging starts at low SOC
                    soc_values = min(1, initial_soc + (charge_added / nominal_capacity_ah));
                end
                method_used = 'Coulomb counting';
                return;
            end
        catch
            % Coulomb counting failed, continue to next method
        end
    end
    
    % Method 4: Voltage-based estimation (least accurate, last resort)
    if isfield(data, 'Voltage_measured')
        voltage = double(data.Voltage_measured(:));
        if strcmp(cycle_type, 'discharge')
            % Simple voltage-based SOC (very rough approximation)
            v_max = max(voltage);
            v_min = min(voltage);
            soc_values = (voltage - v_min) / (v_max - v_min);
        else
            % For charge cycles
            v_max = max(voltage);
            v_min = min(voltage);
            soc_values = (voltage - v_min) / (v_max - v_min);
        end
        method_used = 'Voltage-based estimation (approximate)';
        return;
    end
    
    % If all methods fail
    method_used = 'Failed - no suitable data';
end

function nominal_capacity = estimate_nominal_capacity(battery_var)
    % Estimate nominal capacity from battery data
    nominal_capacity = 2.0;  % Default for NASA batteries (Ah)
    
    try
        % Try to get capacity from battery policy or impedance data
        if isfield(battery_var, 'policy') && isfield(battery_var.policy, 'capacity')
            nominal_capacity = battery_var.policy.capacity;
        elseif isfield(battery_var, 'impedance') && isfield(battery_var.impedance, 'capacity')
            nominal_capacity = battery_var.impedance.capacity;
        else
            % Use typical values for different battery types
            % NASA dataset typically uses Li-ion cells ~2Ah
            nominal_capacity = 2.0;
        end
    catch
        nominal_capacity = 2.0;  % Safe default
    end
end

function result = yesno(condition)
    if condition
        result = '✅ YES';
    else
        result = '❌ NO';
    end
end

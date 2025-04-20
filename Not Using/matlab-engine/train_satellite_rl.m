%% Satellite Simulation: GEO + LEO over Australia with RL Training
clear; clc; close all;
fprintf('=== Starting Satellite Simulation for RL Training ===\n');

% --- Python Environment Setup ---
try
    pe = pyenv; fprintf('MATLAB using Python: %s\n', pe.Executable);
    % Add the directory containing your Python script to Python's path
    % Add parentheses after path as suggested by the error
    py.sys.path().append(pwd); % <--- Corrected line (Line 10)
catch ME
    % Updated error message to reflect potential syntax issue
    error('Python environment not configured correctly (or syntax error calling py.sys.path): %s', ME.message);
end
% --- End Python Setup ---

% ... (Keep your existing code for loading maps, general params, freqs, power, antennas, constants) ...
maps = exist('maps.mat','file');
p836 = exist('p836.mat','file');
p837 = exist('p837.mat','file');
p840 = exist('p840.mat','file');
matFiles = [maps p836 p837 p840];
if ~all(matFiles)
    if ~exist('ITURDigitalMaps.tar.gz','file')
        url = 'https://www.mathworks.com/supportfiles/spc/P618/ITURDigitalMaps.tar.gz';
        websave('ITURDigitalMaps.tar.gz',url);
        untar('ITURDigitalMaps.tar.gz');
    else
        untar('ITURDigitalMaps.tar.gz');
    end
    addpath(cd);
end

%% General Simulation Parameters
fprintf('Initializing simulation parameters...\n');
startTime = datetime(2025,4,10,12,0,0);
% **Reduce duration significantly for online training test runs**
duration_sec = 60 * 10; % Example: 10 minutes (adjust as needed, full 30min will be LONG)
sampleTime = 1;
stopTime = startTime + seconds(duration_sec);

% Frequencies (Hz)
baseFreq = 1.5e9;
channelBW = 200e3;
channelFreqs = 1e9 * [1.498875, 1.500125, 1.500375, 1.500625, 1.500875, ...
                      1.501125, 1.501375, 1.501625, 1.501875, 1.502125]; % 10 channels

% Power (dBW)
geoPower = 10*log10(300);
leoPower = 10*log10(20);

% Antennas (m)
leoAntenna = 0.5; geoAntenna = 3; gsAntenna = 2.4;

% Constants
earthRadius = 6371; kb = physconst('Boltzmann'); tempK = 293;

%% Create Scenario
fprintf('Creating satellite scenario...\n');
sc = satelliteScenario(startTime, stopTime, sampleTime);

%% Ground Stations
fprintf('Setting up ground stations...\n');
cities = {'Melbourne', -37.8136, 144.9631; 'Sydney', -33.8688, 151.2093};
gsList = {};
for i = 1:size(cities,1)
    fprintf('  Creating ground station: %s\n', cities{i,1});
    gsList{end+1} = groundStation(sc, cities{i,2}, cities{i,3}, 'Name', cities{i,1});
end
numGS = numel(gsList);

%% Create GEO Satellites
fprintf('Creating GEO satellites...\n');
geoNum = 1; geoSats = {}; geoLongitudes = [160]; sma_geo = (35786 + earthRadius) * 1e3;
geoTx = cell(1, geoNum); geoTxGimbals = cell(1, geoNum);
for i = 1:geoNum
    geoSats{i} = satellite(sc, sma_geo, 0, 0, 0, 0, geoLongitudes(i), ...
        'Name', sprintf('GEO-%d', i), 'OrbitPropagator', 'two-body-keplerian');
    geoTxGimbals{i} = gimbal(geoSats{i});
    tx = transmitter(geoTxGimbals{i}, 'Frequency', baseFreq, 'Power', geoPower, 'SystemLoss', 1.0);
    gaussianAntenna(tx, 'DishDiameter', geoAntenna); geoTx{i} = tx;
    if ~isempty(gsList), pointAt(geoTxGimbals{i}, gsList{1}); end
end

%% Create LEO Satellites
fprintf('Creating LEO satellites...\n');
leoNum = 3; % MUST match Python NUM_LEOS
leoInclination = 90; leoRAANs = [150, 160, 170]; leoSats = {}; sma_leo = (650 + earthRadius) * 1e3;
leoTx = {};
for i = 1:leoNum
    leoSats{i} = satellite(sc, sma_leo, 0, leoInclination, leoRAANs(i), 0, -65, ...
        'Name', sprintf('LEO-%d', i), 'OrbitPropagator', 'two-body-keplerian');
    tx = transmitter(leoSats{i}, 'Frequency', channelFreqs(1), 'Power', leoPower, 'SystemLoss', 1.0);
    gaussianAntenna(tx, 'DishDiameter', leoAntenna); leoTx{i} = tx;
end

%% Receivers on Ground Stations
fprintf('Setting up ground station receivers...\n');
rxGimbals_GEO = containers.Map(); rxGimbals_LEO = containers.Map();
rxReceivers_GEO = containers.Map(); rxReceivers_LEO = containers.Map();
for i = 1:numGS
    gs = gsList{i}; gsName = gs.Name;
    % GEO Rx
    geoGimbal = gimbal(gs); geoRx = receiver(geoGimbal, 'GainToNoiseTemperatureRatio', 30, 'RequiredEbNo', 10, 'SystemLoss', 1.0);
    gaussianAntenna(geoRx, 'DishDiameter', gsAntenna);
    if ~isempty(geoSats), pointAt(geoGimbal, geoSats{1}); end
    rxGimbals_GEO(gsName) = geoGimbal; rxReceivers_GEO(gsName) = geoRx;
    % LEO Rx
    leoGimbal = gimbal(gs); leoRx = receiver(leoGimbal, 'GainToNoiseTemperatureRatio', 30, 'RequiredEbNo', 10, 'SystemLoss', 1.0);
    gaussianAntenna(leoRx, 'DishDiameter', gsAntenna);
    if ~isempty(leoSats), pointAt(leoGimbal, leoSats{1}); end % Initial point
    rxGimbals_LEO(gsName) = leoGimbal; rxReceivers_LEO(gsName) = leoRx;
end

% --- Initialize Python RL Trainer ---
fprintf('Initializing Python RL Trainer...\n');
try
    py.importlib.import_module('matlab_rl_trainer');
    % Initialize the trainer instance in Python (verbose=1)
    init_success = py.matlab_rl_trainer.initialize_trainer(int32(1));
    if ~init_success
        error('Failed to initialize Python trainer. Check Python script.');
    end
    fprintf('Python RL Trainer initialized successfully.\n');
catch ME
    error('Failed to import or run Python initialization: %s', ME.message);
end
% ---

%% Initialize data collection (Simplified for Training Focus)
ts = startTime:seconds(sampleTime):stopTime;
numTimeSteps = length(ts);
fprintf('Simulation duration: %.1f minutes (%d steps)\n', duration_sec/60, numTimeSteps);

% Store previous LEO frequencies for state calculation
previousLEOFreqs = zeros(1, leoNum);
for i = 1:leoNum, previousLEOFreqs(i) = leoTx{i}.Frequency; end

% Track rewards for basic analysis
rewardsLog = NaN(numTimeSteps, 1);
sinrLog = NaN(numTimeSteps, numGS); % Log average SINR per GS

%% Simulation Loop for Training
fprintf('Starting main simulation loop for training...\n');
totalReward = 0;

for tIdx = 1:numTimeSteps
    t = ts(tIdx);
    fprintf('\n--- Time Step %d/%d: %s ---\n', tIdx, numTimeSteps, datestr(t));

    % --- 1. Construct Current State Vector ---
    % **MUST MATCH Python's EXPECTED_OBSERVATION_SHAPE and order**
    stateVector = [];
    timeFeature = mod(seconds(t-startTime), 24*3600) / (24*3600);
    stateVector = [stateVector, timeFeature];
    stateVector = [stateVector, geoTx{1}.Frequency / 1e9]; % GEO Freq (GHz)
    for i = 1:leoNum
        [pos, ~] = states(leoSats{i}, t, 'CoordinateFrame', 'geographic');
        stateVector = [stateVector, pos(1)/90];  % Normalized Latitude
        stateVector = [stateVector, pos(2)/180]; % Normalized Longitude
        stateVector = [stateVector, previousLEOFreqs(i) / 1e9]; % Prev Freq (GHz)
    end
    % Ensure state vector has the correct shape (e.g., 11 elements)
     if numel(stateVector) ~= 11 % Hardcoded based on example OBS_SHAPE
          error('State vector size mismatch! Expected 11, Got %d', numel(stateVector));
     end

    % --- 2. Get Action from Python Trainer ---
    action_indices = []; % Initialize
    try
        py_state = py.list(stateVector);
        py_action_indices = py.matlab_rl_trainer.get_action_from_trainer(py_state);
        action_indices = double(py_action_indices); % 0-based indices

        if numel(action_indices) ~= leoNum
             warning('MATLAB:RLActionSize', 'Python returned %d actions, expected %d. Using default.', numel(action_indices), leoNum);
             action_indices = zeros(1, leoNum);
        end
        matlab_indices = action_indices + 1; % 1-based for MATLAB
        if any(matlab_indices < 1) || any(matlab_indices > numel(channelFreqs))
             warning('MATLAB:RLActionBounds', 'Python returned invalid index. Using default.');
             matlab_indices = ones(1, leoNum);
             action_indices = zeros(1, leoNum); % Log 0-based default
        end
        currentLEOFreqs = channelFreqs(matlab_indices);
        fprintf('  RL Action (Indices 0-based): %s -> Freqs (MHz): %s\n', ...
            mat2str(action_indices), mat2str(currentLEOFreqs/1e6));

    catch ME
        warning('MATLAB:PythonCallFail', 'Error calling Python get_action: %s. Using previous frequencies.', ME.message);
        currentLEOFreqs = previousLEOFreqs;
        % How to handle action for storage? Maybe store NaN or default?
         action_indices = NaN(1, leoNum); % Indicate failure
    end

    % --- 3. Apply Action: Update LEO Frequencies ---
    for i = 1:leoNum
        if ~isnan(currentLEOFreqs(i))
             leoTx{i}.Frequency = currentLEOFreqs(i);
        else
             % Handle case where frequency determination failed
             % Maybe keep previous: leoTx{i}.Frequency = previousLEOFreqs(i);
             % Or set to a default? For now, keep previous if NaN
             currentLEOFreqs(i) = previousLEOFreqs(i); % Ensure currentLEOFreqs is not NaN for state update
             leoTx{i}.Frequency = currentLEOFreqs(i);
        end
    end

    % --- 4. Simulate Step & Calculate Reward (SINR) ---
    %    (Includes LEO/GEO position updates implicitly via time `t`)
    %    Calculate SINR for GEO links, as LEO frequencies interfere with them.
    currentStepSINR_dB = []; % Store SINR for active links this step
    geoAccessThisStep = false;

    for i = 1:geoNum % Should only be 1 GEO here
        totalInterferenceW_perGS = zeros(1, numGS); % Track interference per GS
        signalPowerW_perGS = zeros(1, numGS);      % Track signal power per GS

        for gsIdx = 1:numGS
            gsName = gsList{gsIdx}.Name;
            rxGEO = rxReceivers_GEO(gsName);
            pointAt(rxGimbals_GEO(gsName), geoSats{i}); % Ensure pointing

            accCheck = access(geoSats{i}, gsList{gsIdx});
            if accessStatus(accCheck, t)
                geoAccessThisStep = true; % At least one GEO link is active
                try
                    % GEO Signal Power
                    linkGEO = link(geoTx{i}, rxGEO);
                    [~, Pwr_GEO_dBW] = sigstrength(linkGEO, t);
                    freqForAtmos = max(baseFreq, 4e9);
                    [~, elevAngle, ~] = aer(rxGEO, geoSats{i}, t); elevAngle = max(elevAngle, 0);
                    cfg = p618Config; cfg.Frequency = freqForAtmos; cfg.ElevationAngle = elevAngle;
                    cfg.Latitude = gsList{gsIdx}.Latitude; cfg.Longitude = gsList{gsIdx}.Longitude;
                    cfg.TotalAnnualExceedance = 0.01;
                    [pl, ~, ~] = p618PropagationLosses(cfg); atmosLoss_dB = pl.At;
                    if isnan(atmosLoss_dB) || isinf(atmosLoss_dB), atmosLoss_dB = 0; end
                    signalPwr_GEO_dBW = Pwr_GEO_dBW - atmosLoss_dB;
                    signalPowerW_perGS(gsIdx) = 10^(signalPwr_GEO_dBW / 10);

                    % Thermal Noise Power
                    noisePwr_W = kb * tempK * channelBW;

                    % Interference Power from LEOs
                    intfPowerSum_W = 0;
                    for j = 1:leoNum
                        txLEO_interf = leoTx{j}; intfFreq = txLEO_interf.Frequency;
                        overlapFactor = getOverlapFactor(baseFreq, channelBW, intfFreq, channelBW);
                        if overlapFactor > 0
                            interfAccCheck = access(leoSats{j}, gsList{gsIdx});
                            if accessStatus(interfAccCheck, t)
                                linkLEOtoGEOreceiver = link(txLEO_interf, rxGEO);
                                [~, intfPwr_dBW] = sigstrength(linkLEOtoGEOreceiver, t);
                                [~, elevLEO, ~] = aer(rxGEO, leoSats{j}, t); elevLEO = max(elevLEO, 0);
                                cfgLEO = cfg; cfgLEO.Frequency = max(intfFreq, 4e9); cfgLEO.ElevationAngle = elevLEO;
                                [plLEO, ~, ~] = p618PropagationLosses(cfgLEO); atmosLossLEO_dB = plLEO.At;
                                if isnan(atmosLossLEO_dB) || isinf(atmosLossLEO_dB), atmosLossLEO_dB = 0; end
                                intfPwr_dBW = intfPwr_dBW - atmosLossLEO_dB;
                                intfPwr_W = (10^(intfPwr_dBW / 10)) * overlapFactor;
                                intfPowerSum_W = intfPowerSum_W + intfPwr_W;
                            end
                        end
                    end
                    totalInterferenceW_perGS(gsIdx) = intfPowerSum_W;

                    % SINR Calculation for this link
                    totalIntfNoise_W = noisePwr_W + intfPowerSum_W;
                    link_SINR_dB = -Inf; % Default if signal is zero
                    if signalPowerW_perGS(gsIdx) > 0 && totalIntfNoise_W > 0
                         link_SINR_dB = 10 * log10(signalPowerW_perGS(gsIdx) / totalIntfNoise_W);
                    elseif signalPowerW_perGS(gsIdx) > 0
                         link_SINR_dB = Inf; % No noise/interference case
                    end
                    currentStepSINR_dB = [currentStepSINR_dB, link_SINR_dB]; % Add valid SINR
                    sinrLog(tIdx, gsIdx) = link_SINR_dB; % Log individual SINR

                    fprintf('    GEO-%d to %s: Access=YES, RSSI=%.2f dBm, SINR=%.2f dB (Intf=%.2f dBW)\n', ...
                           i, gsName, signalPwr_GEO_dBW + 30, link_SINR_dB, 10*log10(intfPowerSum_W));

                catch ME_link_geo
                     fprintf('    GEO-%d to %s: Error calculating link: %s\n', i, gsName, ME_link_geo.message);
                     sinrLog(tIdx, gsIdx) = NaN;
                end
            else % No GEO Access to this GS
                 sinrLog(tIdx, gsIdx) = NaN;
                 % fprintf('    GEO-%d to %s: Access=NO\n', i, gsName);
            end % End if access
        end % End loop GS
    end % End loop GEO

    % --- Define Reward ---
    % Example: Average SINR across all *active* GEO links this step.
    % Penalize heavily if SINR is very low? Or reward increase?
    % Avoid Inf values in mean calculation.
    validSINR = currentStepSINR_dB(~isinf(currentStepSINR_dB) & ~isnan(currentStepSINR_dB));
    if isempty(validSINR) || ~geoAccessThisStep
        % No active GEO links or all had issues / Inf SINR
        reward = -5; % Assign a negative reward if no useful link? Or zero? TUNABLE
        fprintf('  Reward Calc: No valid finite SINR values found. Reward = %.2f\n', reward);
    else
        % Example reward: average dB value. Could also use linear average.
        reward = mean(validSINR);
        % Alternative: Reward shaping, e.g., clip rewards, penalize < threshold
        % reward = max(-10, min(20, mean(validSINR))); % Clipped reward
        fprintf('  Reward Calc: Avg SINR = %.2f dB. Reward = %.2f\n', mean(validSINR), reward);
    end
    rewardsLog(tIdx) = reward;
    totalReward = totalReward + reward;


    % --- 5. Determine 'done' flag ---
    % Is this the end of the simulation (episode)?
    done = (tIdx == numTimeSteps);

    % --- 6. Construct Next State Vector ---
    % Required for the (s, a, r, s', done) tuple
    nextStateVector = [];
    if ~done
        t_next = ts(tIdx+1);
        nextTimeFeature = mod(seconds(t_next-startTime), 24*3600) / (24*3600);
        nextStateVector = [nextStateVector, nextTimeFeature];
        nextStateVector = [nextStateVector, geoTx{1}.Frequency / 1e9]; % GEO Freq
        for i = 1:leoNum
            [pos_next, ~] = states(leoSats{i}, t_next, 'CoordinateFrame', 'geographic');
            nextStateVector = [nextStateVector, pos_next(1)/90];  % Norm Lat
            nextStateVector = [nextStateVector, pos_next(2)/180]; % Norm Lon
            % Use the frequencies *chosen* in the current step as 'previous' for the next state
            nextStateVector = [nextStateVector, currentLEOFreqs(i) / 1e9];
        end
         % Ensure vector size consistency
         if numel(nextStateVector) ~= 11 % Hardcoded OBS_SHAPE size
             error('Next state vector size mismatch! Expected 11, Got %d', numel(nextStateVector));
         end
    else
        % If done, the next state doesn't matter much for A2C buffer, but needs correct shape
        nextStateVector = zeros(1, 11); % Use zeros or NaN? Python side expects shape.
    end

    % --- 7. Send Experience to Python Trainer ---
    % Pass action, reward, next_state, done
    if ~any(isnan(action_indices)) % Only store if action was valid
        try
            py_next_state = py.list(nextStateVector);
            py_reward = py.float(reward);
            py_done = py.bool(done);

            % Call Python function to store experience and potentially train
            % Action is implicitly known by Python from the last get_action call
             py.matlab_rl_trainer.store_and_train_step(py_reward, py_next_state, py_done);

        catch ME
            warning('MATLAB:PythonStoreFail', 'Error calling Python store_and_train_step: %s.', ME.message);
        end
    else
         fprintf('  Skipping storage for this step due to invalid action.\n');
    end


    % --- Update 'previous' frequencies for the *next* iteration's state ---
    previousLEOFreqs = currentLEOFreqs;


end % End loop Time

fprintf('\n--- Simulation Loop Finished ---\n');
fprintf('Total Reward Accumulated: %.2f\n', totalReward);

% --- 8. Save Trained Model ---
fprintf('Saving trained model from Python...\n');
try
    py.matlab_rl_trainer.save_trained_model();
    fprintf('Python save function called successfully.\n');
catch ME
    fprintf('Error calling Python save function: %s\n', ME.message);
end

%% Optional: Plot Reward Curve
figure;
plot(ts, rewardsLog);
xlabel('Time');
ylabel('Reward (Avg SINR dB)');
title('Reward per Time Step during Training');
grid on;

fprintf('=== Simulation and Training Complete ===\n');

% Helper function (no changes needed)
function overlapFactor = getOverlapFactor(txFreq, txBW, intfFreq, intfBW)
    txMin = txFreq - txBW/2; txMax = txFreq + txBW/2;
    intfMin = intfFreq - intfBW/2; intfMax = intfFreq + intfBW/2;
    overlapStart = max(txMin, intfMin); overlapEnd = min(txMax, intfMax);
    overlap = max(0, overlapEnd - overlapStart);
    if intfBW > 0, overlapFactor = overlap / intfBW; else, overlapFactor = 0; end
    overlapFactor = max(0, min(overlapFactor, 1));
end
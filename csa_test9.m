%% Satellite Simulation: GEO + LEO over Australia
clear; clc;
fprintf('=== Starting Satellite Simulation ===\n');

%% Earth-Space Propagation Losses from ITU documents
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
duration_sec = 60 * 3600;     % simulation duration in seconds
sampleTime = 60;            % second sampling time
stopTime = startTime + seconds(duration_sec);

% Frequencies (GHz)
baseFreq = 1.5e9;      % Base frequency
channelBW = 200e3;     % Channel bandwidth


channelFreqs = 1e9 * [1.498875, 1.500125, 1.500375, 1.500625, 1.500875, ...
                      1.501125, 1.501375, 1.501625, 1.501875, 1.502125]; % 10 channels

%{
channelFreqs = 1e9 * [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]; % channel for interference testing
%}

% Power (dBW)
geoPower = 10*log10(300); % 300 W in dBW
leoPower = 10*log10(20);  % 20 W in dBW

% Antennas (m)
leoAntenna = 0.5;      % LEO antenna diameter
geoAntenna = 3;        % GEO antenna diameter
gsAntenna = 2.4;       % Ground station antenna diameter

% Constants
earthRadius = 6371;     % km
kb = physconst('Boltzmann');
tempK = 293;           % Noise temperature

%% Create Scenario
fprintf('Creating satellite scenario...\n');
sc = satelliteScenario(startTime, stopTime, sampleTime);


%% Ground Stations in Australia
fprintf('Setting up ground stations in Australia...\n');

geoCities = {
    'Sydney',       -33.8688, 151.2093;
    'Melbourne',    -37.8136, 144.9631;
    'Brisbane',     -27.4698, 153.0251;
    'Perth',        -31.9505, 115.8605;
    'Adelaide',     -34.9285, 138.6007;
    'Hobart',       -42.8821, 147.3272;
    'Darwin',       -12.4634, 130.8456;
    'Canberra',     -35.2809, 149.1300;
    'Cairns',       -16.9203, 145.7710;
    'Gold_Coast',   -28.0167, 153.4000;
};

leoCities = {
    'Newcastle',    -32.9283, 151.7817;
    'Geelong',      -38.1499, 144.3617;
    'Sunshine_Coast', -26.6500, 153.0667;
    'Mandurah',     -32.5366, 115.7447;
    'Victor_Harbor', -35.5500, 138.6167;
    'Launceston',   -41.4333, 147.1667;
    'Katherine',    -14.4667, 132.2667;
    'Wollongong',   -34.4244, 150.8939;
    'Townsville',   -19.2500, 146.8167;
    'Toowoomba',    -27.5667, 151.9500;
};


geoGsList = cell(1, size(geoCities, 1));
leoGsList = cell(1, size(leoCities, 1));


%% Create GEO and LEO ground statsions
for i = 1:size(geoCities,1)
    geoGsList{i} = groundStation(sc, geoCities{i,2}, geoCities{i,3}, 'Name', geoCities{i,1});
    geoGsList{i}.MarkerColor = [1 0 0];  % Red
end
for i = 1:size(leoCities,1)
    leoGsList{i} = groundStation(sc, leoCities{i,2}, leoCities{i,3}, 'Name', leoCities{i,1});
    leoGsList{i}.MarkerColor = [0 0 1];  % Blue
end

%% Create GEO Satellites
fprintf('Creating GEO satellites...\n');
geoNum = 1; 
geoSats = [];
geoLongitudes = [150 160 170]; % Centered over Australia
sma_geo = (35786 + earthRadius) * 1e3; % GEO altitude


geoTx = cell(1, geoNum);
geoTxGimbals = cell(1, geoNum);

for i = 1:geoNum
    fprintf('  Creating GEO satellite %d at %d°E longitude\n', i, geoLongitudes(i));
    geoSats{i} = satellite(sc, sma_geo, 0, 0, 0, 0, geoLongitudes(i), ...
        'Name', sprintf('GEO-%d', i), 'OrbitPropagator', 'two-body-keplerian');
    geoSats{i}.MarkerColor = [0.9290 0.6940 0.1250];  % Orange
    
    % Add gimbal for pointing at ground stations
    geoTxGimbals{i} = gimbal(geoSats{i});
    
    % Create transmitter mounted on gimbal
    tx = transmitter(geoTxGimbals{i}, 'Frequency', baseFreq, 'Power', geoPower, 'SystemLoss', 1.0);
    gaussianAntenna(tx, 'DishDiameter', geoAntenna);
    geoTx{i} = tx;

    % Point gimbal at all GS (for now, point at 1st GS)
    % pointAt(geoTxGimbals{i}, geoGsList{i});
end


%% Create LEO Satellites
fprintf('Creating LEO satellites...\n');
leoNum = 3;
leoInclination = 90;  % Inclination for coverage
%leoRAANs = [170, 180, 190];
leotrueAnomaly = [-50,-60,-70];
leoRAANs = [155, 160, 165];
leoSats = [];
sma_leo = (650 + earthRadius) * 1e3; % Average LEO altitude
leoTx = [];

for i = 1:leoNum
    fprintf('  Creating LEO satellite %d with RAAN %d°\n', i, leoRAANs(i));
    leoSats{i} = satellite(sc, sma_leo, 0, leoInclination, leoRAANs(i), 0, leotrueAnomaly(i), ...
        'Name', sprintf('LEO-%d', i), 'OrbitPropagator', 'two-body-keplerian');
    leoSats{i}.MarkerColor = [0.3010 0.7450 0.9330];  % Light Blue

    tx = transmitter(leoSats{i}, 'Frequency', channelFreqs(1), 'Power', leoPower);
    gaussianAntenna(tx, 'DishDiameter', leoAntenna);   
    pattern(tx);
    leoTx{i} = tx;   
end

%% Receivers on Ground Stations
fprintf('Setting up ground station receivers...\n');
fprintf('Setting up dual receivers and gimbals for each ground station...\n');

rxGimbals_GEO = containers.Map();
rxGimbals_LEO = containers.Map();
rxReceivers_GEO = containers.Map();
rxReceivers_LEO = containers.Map();

% Create and store GEO receiver and gimbal for each GEO GS
for i = 1:numel(geoGsList)
    gs = geoGsList{i};
    gsName = gs.Name;

    % --- Gimbal and receiver for GEO GS
    geoGimbal = gimbal(gs);
    geoRx = receiver(geoGimbal, ...
        'GainToNoiseTemperatureRatio', 30, ...
        'RequiredEbNo', 10, ...
        'SystemLoss', 1.0);
    gaussianAntenna(geoRx, ...
        'DishDiameter', gsAntenna);

    % Point to first GEO for now
    pointAt(geoGimbal, geoSats{1});

    % Store
    rxGimbals_GEO(gsName) = geoGimbal;
    rxReceivers_GEO(gsName) = geoRx;
end

% Create and store LEO receiver and gimbal for each LEO GS
for i = 1:numel(leoGsList)
    gs = leoGsList{i};
    gsName = gs.Name;

    % --- Gimbal and receiver for LEO GS
    leoGimbal = gimbal(gs);
    leoRx = receiver(leoGimbal, ...
        'GainToNoiseTemperatureRatio', 30, ...
        'RequiredEbNo', 10, ...
        'SystemLoss', 1.0);
    gaussianAntenna(leoRx, ...
        'DishDiameter', gsAntenna);

    % Point to first LEO initially
    pointAt(leoGimbal, leoSats{1});

    % Store
    rxGimbals_LEO(gsName) = leoGimbal;
    rxReceivers_LEO(gsName) = leoRx;
end


%% Initialize data collection
fprintf('Initializing data collection...\n');
ts = startTime:seconds(sampleTime):stopTime;
validSamples = 0;

fprintf('Starting first pass to count valid samples...\n');
% First pass to count valid samples (where at least one LEO has access)
for tIdx = 1:length(ts)
    t = ts(tIdx);
    leoAccess = false;
    % access line is drawn based on visibility and elevation, not antenna pattern gain.
    
    % Check if any LEO has access to any ground station
    for i = 1:leoNum
        for gsIdx = 1:numel(leoGsList)
            lac = access(leoSats{i}, leoGsList{gsIdx});
            lac.LineColor = 'red';
            lac.LineWidth = 3;
            if accessStatus(lac, t)
                leoAccess = true;
                fprintf('  Found LEO access at %s (LEO-%d to %s)\n', datestr(t), i, leoGsList{gsIdx}.Name);
                break;
            end
        end
        if leoAccess, break; end
    end
    
    if leoAccess
        validSamples = validSamples + 1;
    end

    geoAccess = false;

    % Check if any GEO has access to any ground station
    for i = 1:geoNum
        for gsIdx = 1:numel(geoGsList)
            gac = access(geoSats{i}, geoGsList{gsIdx});
            gac.LineColor = 'blue';
            gac.LineWidth = 3;
            if accessStatus(gac, t)
                geoAccess = true;
                fprintf('  Found GEO access at %s (GEO-%d to %s)\n', datestr(t), i, geoGsList{gsIdx}.Name);
                break;
            end
        end
        if geoAccess, break; end
    end

    if geoAccess
        validSamples = validSamples + 1;
    end

end
fprintf('First pass complete. Found %d valid samples with LEO access.\n', validSamples);


% Pre-allocate based on valid samples
fprintf('Pre-allocating data structures...\n');
logData = struct();
logData.Time = NaT(validSamples, 1);
logData.GEO = struct();
logData.LEO = struct();

% Initialize GEO data
for i = 1:geoNum
    logData.GEO(i).Name = geoSats{i}.Name;
    logData.GEO(i).Latitude = zeros(validSamples, 1);
    logData.GEO(i).Longitude = zeros(validSamples, 1);
    logData.GEO(i).Frequency = baseFreq * ones(validSamples, 1);
    logData.GEO(i).Access = zeros(validSamples, numel(geoGsList));
    logData.GEO(i).SNR = NaN(validSamples, numel(geoGsList));
    logData.GEO(i).RSSI = NaN(validSamples, numel(geoGsList));
end

% Initialize LEO data
for i = 1:leoNum
    logData.LEO(i).Name = leoSats{i}.Name;
    logData.LEO(i).Latitude = zeros(validSamples, 1);
    logData.LEO(i).Longitude = zeros(validSamples, 1);
    logData.LEO(i).Frequency = zeros(validSamples, 1);
    logData.LEO(i).Access = zeros(validSamples, numel(leoGsList));
    logData.LEO(i).SNR = NaN(validSamples, numel(leoGsList));
    logData.LEO(i).RSSI = NaN(validSamples, numel(leoGsList));
end

%store time-series SNR for each satellite
snrTimeline = struct();
for i = 1:leoNum
    snrTimeline.(sprintf('LEO%d', i)) = [];
end
for i = 1:geoNum
    snrTimeline.(sprintf('GEO%d', i)) = [];
end
snrTimeline.Time = datetime.empty;



%% Simulation Loop with Selective Logging
fprintf('Starting main simulation loop...\n');
sampleCount = 0;

for tIdx = 1:length(ts)
    t = ts(tIdx);
    fprintf('\nProcessing time step %d/%d: %s\n', tIdx, length(ts), datestr(t));
    
    leoAccess = false;
    geoAccess = false;
    accessDetails = '';
    
    % First check if any LEO has access to any ground station
    for i = 1:leoNum
        for gsIdx = 1:numel(leoGsList)
            %lac = access(leoSats{i}, leoGsList{gsIdx});
            %lac.LineColor = 'red';
            %lac.LineWidth = 3;
            if accessStatus(lac, t)
                leoAccess = true;
                accessDetails = sprintf('LEO-%d to %s', i, leoGsList{gsIdx}.Name);
                fprintf('  Access detected: %s\n', accessDetails);
                break;
            end
        end
        if leoAccess, break; end
    end


    % Check if any GEO has access to any ground station
    for i = 1:geoNum
        for gsIdx = 1:numel(geoGsList)
            %gac = access(geoSats{i}, geoGsList{gsIdx});
            %gac.LineColor = 'blue';
            %gac.LineWidth = 3;
            if accessStatus(gac, t)
                geoAccess = true;
                fprintf('  Found GEO access at %s (GEO-%d to %s)\n', datestr(t), i, geoGsList{gsIdx}.Name);
                break;
            end
        end
        if geoAccess, break; end
    end
    
    % Only process if at least one LEO has access (GEOs can be added later)
    if leoAccess | geoAccess
        sampleCount = sampleCount + 1;
        logData.Time(sampleCount) = t;
        fprintf('  Processing sample %d (valid sample %d)\n', tIdx, sampleCount);
        
        %% Fequecy allocation logic (random channel selection for both GEO and LEO)
        % Update LEO frequencies
        currentLEOFreqs = channelFreqs(randi([1 10], 1, leoNum));
        fprintf('  Selected LEO frequencies: %s MHz\n', mat2str(currentLEOFreqs/1e6));

        % Update GEO frequencies
        currentGEOFreqs = channelFreqs(randi([1 10], 1, geoNum));
        fprintf('  Selected GEO frequencies: %s MHz\n', mat2str(currentGEOFreqs/1e6));

        % Update LEO satellite data
        for i = 1:leoNum
            [pos, ~] = states(leoSats{i}, t, 'CoordinateFrame', 'geographic');
            logData.LEO(i).Latitude(sampleCount) = pos(1);
            logData.LEO(i).Longitude(sampleCount) = pos(2);
            logData.LEO(i).Frequency(sampleCount) = currentLEOFreqs(i);
            
            % Update transmitter frequency for this LEO
            tx = leoTx{i};
            tx.Frequency = currentLEOFreqs(i);
            %pattern(tx);
            
            % Check access and calculate metrics for each ground station
            for gsIdx = 1:numel(leoGsList)
                %lac = access(leoSats{i}, leoGsList{gsIdx});
                %lac.LineColor = 'red';
                %lac.LineWidth = 3;
                acc = accessStatus(lac, t);
                logData.LEO(i).Access(sampleCount, gsIdx) = acc;
                
                %if acc
                if true

                    % Calculate link metrics
                    linkLEO = link(tx, rxReceivers_LEO(leoGsList{gsIdx}.Name));
                    [~, Pwr_dBW] = sigstrength(linkLEO, t);


                    % Calculate elevation angle
                    [~, elevationAngle, ~] = aer(rxReceivers_LEO(leoGsList{gsIdx}.Name), leoSats{i}, t);

                    % Use ITU-R P.618 atmospheric propagation loss model
                    cfg = p618Config;
                    cfg.Frequency = max(baseFreq, 4e9); %ITU P.618 model is not officially validated for frequencies below 4 GHz.
                    cfg.ElevationAngle = max(elevationAngle, 5);
                    cfg.Latitude = leoGsList{gsIdx}.Latitude;
                    cfg.Longitude = leoGsList{gsIdx}.Longitude;
                    cfg.TotalAnnualExceedance = 0.001; % Typical exceedance

                    [pl, ~, ~] = p618PropagationLosses(cfg);
                    atmosLoss = pl.At; % Atmospheric attenuation (dB)

                    rssi = Pwr_dBW - atmosLoss;
                    snr = rssi - 10*log10(kb*tempK*channelBW);
                    
                    logData.LEO(i).RSSI(sampleCount, gsIdx) = rssi;
                    logData.LEO(i).SNR(sampleCount, gsIdx) = snr;
                    
                    fprintf('    LEO-%d to %s (%.6f GHz): RSSI=%.2f dBm, SNR=%.2f dB\n', ...
                        i, leoGsList{gsIdx}.Name, currentLEOFreqs(i)/1e9, rssi, snr);
                end
            end
        end
        
        % Update GEO satellite data
        for i = 1:geoNum
            [pos, ~] = states(geoSats{i}, t, 'CoordinateFrame', 'geographic');
            logData.GEO(i).Latitude(sampleCount) = pos(1);
            logData.GEO(i).Longitude(sampleCount) = pos(2);
            logData.GEO(i).Frequency(sampleCount) = currentGEOFreqs(i);

            % Update transmitter frequency for this GEO
            tx = geoTx{i};
            tx.Frequency = currentGEOFreqs(i);
            pattern(tx);
            
            % Check access and calculate metrics for each ground station
            for gsIdx = 1:numel(geoGsList)
                %gac = access(geoSats{i}, geoGsList{gsIdx});
                %gac.LineColor = 'blue';
                %gac.LineWidth = 3;
                acc = accessStatus(gac, t);
                logData.GEO(i).Access(sampleCount, gsIdx) = acc;
                
                %if acc
                if true

                    % Calculate link metrics
                    linkGEO = link(geoTx{i}, rxReceivers_GEO(geoGsList{gsIdx}.Name));
                    [~, Pwr_dBW] = sigstrength(linkGEO, t);

                    %signalPwr_dBW = signalPwr_dBW - (rainLoss + cloudLoss);

                    % Calculate elevation angle
                    [~, elevationAngle, ~] = aer(rxReceivers_GEO(geoGsList{gsIdx}.Name), geoSats{i}, t);

                    % Use ITU-R P.618 atmospheric propagation loss model
                    cfg = p618Config;
                    cfg.Frequency = max(baseFreq, 4e9);
                    cfg.ElevationAngle = elevationAngle;
                    cfg.Latitude = geoGsList{gsIdx}.Latitude;
                    cfg.Longitude = geoGsList{gsIdx}.Longitude;
                    cfg.TotalAnnualExceedance = 0.001; % Typical exceedance

                    [pl, ~, ~] = p618PropagationLosses(cfg);
                    atmosLoss = pl.At; % Atmospheric attenuation (dB)

                    % Apply to signal power
                    signalPwr_dBW = Pwr_dBW - atmosLoss;

                    signalPwr_W = 10^(signalPwr_dBW / 10);
                    
                    % Thermal noise in W
                    noisePwr_W = kb * tempK * channelBW;
                    
                    % Interference from each LEO
                    intfPowerSum_W = 0;
                    for j = 1:leoNum
                        txLEO = leoTx{j};
                        intfFreq = txLEO.Frequency;
                        intfBW = channelBW; 
                    
                        %overlapFactor = getOverlapFactor(baseFreq, channelBW, intfFreq, intfBW);
                        overlapFactor = getOverlapFactor(tx.Frequency, channelBW, intfFreq, intfBW);
                        if overlapFactor > 0
                            linkLEO2GS = link(txLEO, rxReceivers_GEO(geoGsList{gsIdx}.Name));
                            [~, intfPwr_dBW] = sigstrength(linkLEO2GS, t);
                            intfPwr_dBW = intfPwr_dBW - atmosLoss;
                            intfPwr_W = 10^(intfPwr_dBW / 10) * overlapFactor;
                            intfPowerSum_W = intfPowerSum_W + intfPwr_W;
                        end
                    end
                    
                    % Total interference + noise
                    totalIntfNoise_W = noisePwr_W + intfPowerSum_W;
                    SINR_dB = 10 * log10(signalPwr_W / totalIntfNoise_W);
                    
                    % Store
                    logData.GEO(i).RSSI(sampleCount, gsIdx) = 10 * log10(signalPwr_W);
                    logData.GEO(i).SNR(sampleCount, gsIdx) = SINR_dB;
                    
                    fprintf('GEO-%d to %s | SINR: %.2f dB | Signal: %.2f dBm | Intf: %.2f dBW\n', ...
                        i, geoGsList{gsIdx}.Name, SINR_dB, 10*log10(signalPwr_W)+30, 10*log10(intfPowerSum_W));


                end
            end
        end
        % Log average SNRs only after SNRs have been computed
        snrTimeline.Time(end+1) = t;
        for i = 1:leoNum
            avgSNR = nanmean(logData.LEO(i).SNR(sampleCount, :));
            snrTimeline.(sprintf('LEO%d', i))(end+1) = avgSNR;
        end
        for i = 1:geoNum
            avgSNR = nanmean(logData.GEO(i).SNR(sampleCount, :));
            snrTimeline.(sprintf('GEO%d', i))(end+1) = avgSNR;
        end
        
    else
        fprintf('  No LEO access detected - skipping this time step\n');

    end
end
fprintf('\nMain simulation loop completed. Processed %d valid samples.\n', sampleCount);


%% Save Data to CSV (only valid samples)
fprintf('\nPreparing data for CSV export...\n');
% Prepare data for CSV export
csvData = table();
csvData.Time = logData.Time;

% Add GEO data
for i = 1:geoNum
    fprintf('  Adding GEO-%d data to CSV structure\n', i);
    csvData.(sprintf('GEO%d_Name', i)) = repmat(logData.GEO(i).Name, validSamples, 1);
    csvData.(sprintf('GEO%d_Lat', i)) = logData.GEO(i).Latitude;
    csvData.(sprintf('GEO%d_Lon', i)) = logData.GEO(i).Longitude;
    csvData.(sprintf('GEO%d_Freq_Hz', i)) = logData.GEO(i).Frequency;
    
    for gsIdx = 1:numel(geoGsList)
        gsName = strrep(geoGsList{gsIdx}.Name, ' ', '_');
        csvData.(sprintf('GEO%d_%s_Access', i, gsName)) = logData.GEO(i).Access(:, gsIdx);
        csvData.(sprintf('GEO%d_%s_SNR_dB', i, gsName)) = logData.GEO(i).SNR(:, gsIdx);
        csvData.(sprintf('GEO%d_%s_RSSI_dBm', i, gsName)) = logData.GEO(i).RSSI(:, gsIdx);
    end
end

% Add LEO data
for i = 1:leoNum
    fprintf('  Adding LEO-%d data to CSV structure\n', i);
    csvData.(sprintf('LEO%d_Name', i)) = repmat(logData.LEO(i).Name, validSamples, 1);
    csvData.(sprintf('LEO%d_Lat', i)) = logData.LEO(i).Latitude;
    csvData.(sprintf('LEO%d_Lon', i)) = logData.LEO(i).Longitude;
    csvData.(sprintf('LEO%d_Freq_Hz', i)) = logData.LEO(i).Frequency;
    
    for gsIdx = 1:numel(leoGsList)
        gsName = strrep(leoGsList{gsIdx}.Name, ' ', '_');
        csvData.(sprintf('LEO%d_%s_Access', i, gsName)) = logData.LEO(i).Access(:, gsIdx);
        csvData.(sprintf('LEO%d_%s_SNR_dB', i, gsName)) = logData.LEO(i).SNR(:, gsIdx);
        csvData.(sprintf('LEO%d_%s_RSSI_dBm', i, gsName)) = logData.LEO(i).RSSI(:, gsIdx);
    end
end

% Write to CSV
fprintf('Writing data to CSV file...\n');
writetable(csvData, 'Satellite_Australia_Simulation_Log.csv');
fprintf('CSV saved with %d valid samples: Satellite_Australia_Simulation_Log.csv\n', validSamples);

%% Play Simulation
%fprintf('\nStarting visualization...\n');
%v = satelliteScenarioViewer(sc);
%v.ShowDetails = true;
%play(sc, 'PlaybackSpeedMultiplier', 100);
fprintf('=== Simulation Complete ===\n');


function overlapFactor = getOverlapFactor(txFreq, txBW, intfFreq, intfBW)
    txRange = [txFreq - txBW/2, txFreq + txBW/2];
    intfRange = [intfFreq - intfBW/2, intfFreq + intfBW/2];
    overlap = max(0, min(txRange(2), intfRange(2)) - max(txRange(1), intfRange(1)));
    overlapFactor = overlap / intfBW;
end


%% Save Simulation State
fprintf('\nSaving simulation scenario and log data...\n');
save('SatelliteSimulationState.mat', 'sc', 'logData', 'geoSats', 'leoSats', 'geoGsList', 'leoGsList', 'geoTx', 'leoTx', 'snrTimeline');
fprintf('Simulation state saved to SatelliteSimulationState.mat\n');

%% Load Simulation State
fprintf('\nLoading simulation scenario data...\n');
load('SatelliteSimulationState.mat');
v = satelliteScenarioViewer(sc);
v.ShowDetails = true;
play(sc, 'PlaybackSpeedMultiplier', 100);
fprintf('=== Simulation Load Complete ===\n');



%% plot
figure;
hold on;

plotHandles = [];
legendEntries = {};

% Plot LEO SNR
for i = 1:leoNum
    h = plot(snrTimeline.Time, snrTimeline.(sprintf('LEO%d', i)), '-o');
    plotHandles(end+1) = h;
    legendEntries{end+1} = sprintf('LEO-%d', i);
end

% Plot GEO SNR
for i = 1:geoNum
    h = plot(snrTimeline.Time, snrTimeline.(sprintf('GEO%d', i)), '-x');
    plotHandles(end+1) = h;
    legendEntries{end+1} = sprintf('GEO-%d', i);
end

xlabel('Time'); ylabel('Avg SNR (dB)');
title('Replay: SNR Over Time');
legend(plotHandles, legendEntries, 'Location', 'northeastoutside');
grid on;
hold off;



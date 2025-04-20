leoAccess = false;
geoAccess = false;
accessDetails = '';

% First check if any LEO has access to any ground station
for i = 1:leoNum
    for gsIdx = 1:numel(gsList)
        if accessStatus(access(leoSats{i}, gsList{gsIdx}), t)
            leoAccess = true;
            accessDetails = sprintf('LEO-%d to %s', i, gsList{gsIdx}.Name);
            fprintf('  Access detected: %s\n', accessDetails);
            break;
        end
    end
    if leoAccess, break; end
end


% Check if any GEO has access to any ground station
for i = 1:geoNum
    for gsIdx = 1:numel(gsList)
        if accessStatus(access(geoSats{i}, gsList{gsIdx}), t)
            geoAccess = true;
            fprintf('  Found GEO access at %s (GEO-%d to %s)\n', datestr(t), i, gsList{gsIdx}.Name);
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
    
    % Update LEO frequencies (random channel selection)
    currentLEOFreqs = channelFreqs(randi([1 10], 1, leoNum));
    fprintf('  Selected LEO frequencies: %s MHz\n', mat2str(currentLEOFreqs/1e6));



    fprintf(' Sending inital state to py function ...\n');


    % Initialize a struct to store the LEO satellite data
    snd_state = struct;

    % Add base frequency or frequency of GEO to the state
    snd_state.GeobaseFreq = baseFreq;
    snd_state.time = datestr(t)

    for i = 1:leoNum
        % Get position in geographic coordinates (Latitude, Longitude)
        [pos, ~] = states(leoSats{i}, t, 'CoordinateFrame', 'geographic');
        
        % Initialize a struct to hold the satellite data for this LEO
        satellite_data = struct;
        satellite_data.LEO_Num = i;
        satellite_data.Latitude = pos(1);
        satellite_data.Longitude = pos(2);
        
        % Create a struct for access status
        accessStatusStruct = struct;
        
        % For each ground station, check if the satellite has access
        for gsIdx = 1:numel(gsList)
            gsName = strrep(gsList{gsIdx}.Name, ' ', '_');
            accObj = access(leoSats{i}, gsList{gsIdx});
            accessStatusStruct.(gsName) = accessStatus(accObj, t);
        end
        
        % Add AccessStatus to the satellite data
        satellite_data.AccessStatus = accessStatusStruct;
        
        % Store the satellite data
        fieldName = sprintf("LEO_%d", i);
        snd_state.(fieldName) = satellite_data;
    end

    % Convert MATLAB struct to Python dict
    py_state = py.dict(snd_state);

    display(py_state);
end
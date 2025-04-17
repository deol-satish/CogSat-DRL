function [sc, gsList, geoSats, geoTx, geoTxGimbals, leoSats, leoTx, params] = initializeScenario()
    % Load digital maps if needed
    maps = exist('maps.mat','file');
    p836 = exist('p836.mat','file');
    p837 = exist('p837.mat','file');
    p840 = exist('p840.mat','file');
    if ~all([maps p836 p837 p840])
        if ~exist('ITURDigitalMaps.tar.gz','file')
            url = 'https://www.mathworks.com/supportfiles/spc/P618/ITURDigitalMaps.tar.gz';
            websave('ITURDigitalMaps.tar.gz',url);
        end
        untar('ITURDigitalMaps.tar.gz');
        addpath(cd);
    end

    % Time & Frequencies
    params.startTime = datetime(2025,4,10,12,0,0);
    params.duration_sec = 60 * 3;
    params.sampleTime = 10;
    params.stopTime = params.startTime + seconds(params.duration_sec);
    params.ts = params.startTime:seconds(params.sampleTime):params.stopTime;
    
    % System Parameters
    params.baseFreq = 1.5e9;
    params.channelBW = 200e3;
    params.channelFreqs = 1e9 * [1.498875, 1.500125, 1.500375, 1.500625, 1.500875, ...
                                 1.501125, 1.501375, 1.501625, 1.501875, 1.502125];
    params.geoPower = 10*log10(300);
    params.leoPower = 10*log10(20);
    params.geoAntenna = 3;
    params.leoAntenna = 0.5;
    params.gsAntenna = 2.4;
    params.earthRadius = 6371;
    params.kb = physconst('Boltzmann');
    params.tempK = 293;

    % Create scenario
    sc = satelliteScenario(params.startTime, params.stopTime, params.sampleTime);

    % Ground Stations
    cityData = {'Melbourne', -37.8136, 144.9631; 'Sydney', -33.8688, 151.2093};
    gsList = cell(size(cityData, 1), 1);
    for i = 1:size(cityData, 1)
        gsList{i} = groundStation(sc, cityData{i,2}, cityData{i,3}, 'Name', cityData{i,1});
    end

    % GEO
    geoNum = 1;
    geoSats = cell(1, geoNum);
    geoTx = cell(1, geoNum);
    geoTxGimbals = cell(1, geoNum);
    geoLongitudes = [160];
    sma_geo = (35786 + params.earthRadius) * 1e3;
    for i = 1:geoNum
        geoSats{i} = satellite(sc, sma_geo, 0, 0, 0, 0, geoLongitudes(i), ...
            'Name', sprintf('GEO-%d', i), 'OrbitPropagator', 'two-body-keplerian');
        geoTxGimbals{i} = gimbal(geoSats{i});
        geoTx{i} = transmitter(geoTxGimbals{i}, 'Frequency', params.baseFreq, 'Power', params.geoPower, 'SystemLoss', 1.0);
        gaussianAntenna(geoTx{i}, 'DishDiameter', params.geoAntenna);
        pointAt(geoTxGimbals{i}, gsList{1});
    end

    % LEO
    leoNum = 3;
    leoRAANs = [150, 160, 170];
    sma_leo = (650 + params.earthRadius) * 1e3;
    leoSats = cell(1, leoNum);
    leoTx = cell(1, leoNum);
    for i = 1:leoNum
        leoSats{i} = satellite(sc, sma_leo, 0, 90, leoRAANs(i), 0, -65, ...
            'Name', sprintf('LEO-%d', i), 'OrbitPropagator', 'two-body-keplerian');
        leoTx{i} = transmitter(leoSats{i}, 'Frequency', params.channelFreqs(1), 'Power', params.leoPower);
        gaussianAntenna(leoTx{i}, 'DishDiameter', params.leoAntenna);
    end
end

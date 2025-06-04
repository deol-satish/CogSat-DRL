%% Main Script to define the Geometrical simulator, Receivers, and Interference
clear; clc;close all hidden
%% Define Parameters and Ground stations
P01_Parameters
P02_GStations
%% Create Scenario
fprintf('Creating satellite scenario...\n');
sc = satelliteScenario(startTime, stopTime, sampleTime);
%% Satellite and GS creation
P03_GeometricSimulatoion
% play(sc,PlaybackSpeedMultiplier=100);
% save('Geometric',"gsAntenna","fc","c","leoAntenna","ElLEO","RhoLEO","eff", ...
%     "geoAntenna","ElGEO", "RhoGEO", "NumLeoUser", "NumGeoUser", "ts", "GS", ...
%     "leoNum", "geoNum", "GSLEOFilter", "GSGEOFilter", "leoPower", "geoPower");



%% P04_RxSimulation
%% Reciever Gain
Grx = 10* log10((pi * gsAntenna *fc /c)^2 * eff);
ThermalNoisedBm = 10 * log10(kb * TempK * ChannelBW) +30; % Noise in dBm
%% LEO Power calculations
GtxLEO = 10* log10((pi * leoAntenna *fc /c)^2 * eff);
RhoLEO(ElLEO<0) = Inf;
PathLoss = 20*log10(fc) + 20*log10(RhoLEO) -147.55;
AtmoLLEO = F01_ComputeAtmosphericLoss(fc, ElLEO, Att);
FadingLEO = F02_MultipathFadingLoss(FadingModel, ElLEO);
PrxLEO = leoPower + GtxLEO + Grx - PathLoss - AtmoLLEO - FadingLEO;
% PrxLEO = leoPower + GtxLEO + Grx - PathLoss;
SNRLEO = PrxLEO - ThermalNoisedBm;
%% GEO Power calculations
GtxGEO = 10* log10((pi * geoAntenna *fc /c)^2 * eff);
RhoGEO(ElGEO<0) = Inf;
PathLoss = 20*log10(fc) + 20*log10(RhoGEO) -147.55;
AtmoLGEO = F01_ComputeAtmosphericLoss(fc, ElGEO , Att);
FadingGEO = F02_MultipathFadingLoss(FadingModel, ElGEO);
PrxGEO = geoPower + GtxGEO + Grx - PathLoss - AtmoLGEO - FadingGEO;
% PrxGEO = geoPower + GtxGEO + Grx - PathLoss;
SNRGEO = PrxGEO - ThermalNoisedBm;



%% Define which is RFI based on channel allocation
fprintf('Channel allocation...\n');
% Define number of channels based of number of LEO and GEO users + 5 extra
% Each GEO users will always have its own channel
% LEO users will always share all channel randemoly assigned with unique channels per timestep
numChannels = 5 + NumLeoUser + NumGeoUser;
ChannelListLeo = nan(NumGS, leoNum, length(ts));
ChannelListGeo = nan(NumGS, geoNum, length(ts));
LEOUsers = find(GSLEOFilter);  % e.g., 1:10
GEOUsers = find(GSGEOFilter);  % e.g., 11:20
% Only Assign Channels to Valid Users (LEO or GEO)
for t = 1:length(ts)
    for s = 1:leoNum
        ChannelListLeo(LEOUsers, s, t) = randperm(numChannels, NumLeoUser);
    end
    for g = 1:geoNum
        ChannelListGeo(GEOUsers, g, t) = randperm(NumGeoUser, NumGeoUser)';
    end
end



%% Finding the serving LEO for each LEO GS (20 x 31)
fprintf('Finding the serving LEO for each LEO GS...\n');
ActualPrxLEO = PrxLEO .*GSLEOFilter;
[PservLEO, Serv_idxLEO] = max(ActualPrxLEO, [], 2);  % Max over LEO satellites
PservLEO = squeeze(PservLEO);                        % [NumGS × Time]
Serv_idxLEO = GSLEOFilter .* squeeze(Serv_idxLEO);   % [NumGS × Time]
%% Find the serving GEO for each GEO GS (20 x 31)
ActualPrxGEO = PrxGEO .*GSGEOFilter;
[PservGEO, Serv_idxGEO] = max(ActualPrxGEO, [], 2);  % Max over GEOs (dim 2)
PservGEO = squeeze(PservGEO);                        % [NumGS × Time]
Serv_idxGEO =  GSGEOFilter .* squeeze(Serv_idxGEO);  % [NumGS × Time]
%% Find the final channel allocations per users
FreqAlloc = NaN(NumGS, length(ts));  % Initialize
for t = 1:length(ts)
    for u = 1:NumGS
        if GSLEOFilter(u)
            s_serv = Serv_idxLEO(u, t);
            if s_serv > 0 && ~isnan(s_serv)
                FreqAlloc(u, t) = ChannelListLeo(u, s_serv, t);
            end
        elseif GSGEOFilter(u)
            s_serv = Serv_idxGEO(u, t);
            if s_serv > 0 && ~isnan(s_serv)
                FreqAlloc(u, t) = ChannelListGeo(u, s_serv, t);
            end
        end
    end
end

%% Initialise Interference Calculations
fprintf('Interference calculation step...\n');
T = length(ts);
SINR = NaN(NumGS, T);  % Output SINR matrix [NumGS x T]
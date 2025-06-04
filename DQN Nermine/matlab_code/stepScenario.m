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



% save('Data');
%% Interference Calculations
fprintf('Interference calculation step...\n');
T = length(ts);
SINR = NaN(NumGS, T);  % Output SINR matrix [NumGS x T]
for t = 1:T
    PrxLEOt = PrxLEO(:, :, t);              % [NumGS x LEO]
    PrxGEOt = PrxGEO(:, :, t);              % [NumGS x GEO]
    ChannelListLeot = ChannelListLeo(:, :, t);
    ChannelListGeot = ChannelListGeo(:, :, t);
    PservLEOt = PservLEO(:, t);
    Serv_idxLEOt = Serv_idxLEO(:, t);
    PservGEOt = PservGEO(:, t);
    Serv_idxGEOt = Serv_idxGEO(:, t);
    for userIdx = 1:NumGS
        isLEOUser = GSLEOFilter(userIdx);
        isGEOUser = GSGEOFilter(userIdx);

        if isLEOUser
            s_serv = Serv_idxLEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListLeot(userIdx, s_serv);
            Psig_dBm = PservLEOt(userIdx);
        elseif isGEOUser
            s_serv = Serv_idxGEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListGeot(userIdx, s_serv);
            Psig_dBm = PservGEOt(userIdx);
        else
            continue;  % undefined user
        end
        %% Interference from LEO
        PintLEO_mW = 0;
        for s = 1:leoNum
            if isLEOUser && s == s_serv
                continue;
            end
            for u = LEOUsers
                ch_other = ChannelListLeot(u, s);
                if ch_other == ch_user
                    Pint_dBm = PrxLEOt(userIdx, s);
                    if ~isnan(Pint_dBm) && ~isinf(Pint_dBm)
                        PintLEO_mW = PintLEO_mW + 10^(Pint_dBm / 10);
                    end
                end
            end
        end
        %% Interference from GEO
        PintGEO_mW = 0;
        for g = 1:geoNum
            if isGEOUser && g == s_serv
                continue;
            end
            for u = GEOUsers
                ch_other = ChannelListGeot(u, g);
                if ch_other == ch_user
                    Pint_dBm = PrxGEOt(userIdx, g);
                    if ~isnan(Pint_dBm) && ~isinf(Pint_dBm)
                        PintGEO_mW = PintGEO_mW + 10^(Pint_dBm / 10);
                    end
                end
            end
        end
        %% Final SINR
        Pint_total_mW = PintLEO_mW + PintGEO_mW;
        Psig_mW = 10^(Psig_dBm / 10);
        Noise_mW = 10^(ThermalNoisedBm / 10);
        SINR_mW = Psig_mW / (Pint_total_mW + Noise_mW);
        SINR(userIdx, t) = 10 * log10(SINR_mW);
    end
end


P07_Plotting
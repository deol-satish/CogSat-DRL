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


% P07_Plotting
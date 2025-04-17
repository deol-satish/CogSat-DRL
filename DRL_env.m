% Initial setup
[sc, gsList, geoSats, geoTx, geoTxGimbals, leoSats, leoTx, params] = initializeScenario();

% Sample loop for stepping (example)
for tIdx = 1:length(params.ts)
    t = params.ts(tIdx);
    freqs = stepScenario(leoTx, params.channelFreqs);
    % ... logging, access check, link calculations here
end

% Reset if needed
[sc, gsList, geoSats, geoTx, geoTxGimbals, leoSats, leoTx, params] = resetScenario();

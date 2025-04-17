function currentLEOFreqs = stepScenario(leoTx, channelFreqs)
    % Randomly assign new frequency to each LEO transmitter at each step
    numLEO = numel(leoTx);
    currentLEOFreqs = channelFreqs(randi([1, numel(channelFreqs)], 1, numLEO));
    for i = 1:numLEO
        leoTx{i}.Frequency = currentLEOFreqs(i);
    end
end

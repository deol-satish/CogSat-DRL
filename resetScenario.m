function [sc, gsList, geoSats, geoTx, geoTxGimbals, leoSats, leoTx, params] = resetScenario()
    fprintf("Resetting scenario...\n");
    [sc, gsList, geoSats, geoTx, geoTxGimbals, leoSats, leoTx, params] = initializeScenario();
end

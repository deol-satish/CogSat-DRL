import matlab.engine

# Start MATLAB
eng = matlab.engine.start_matlab()

# Add path if needed
eng.addpath(r'./', nargout=0)

eng.eval("initialiseScenario", nargout=0)

eng.eval("resetScenario", nargout=0)

eng.eval("stepScenario", nargout=0)

eng.eval("SaveData", nargout=0)
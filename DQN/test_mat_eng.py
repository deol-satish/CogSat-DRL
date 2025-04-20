import matlab.engine

# Start MATLAB
eng = matlab.engine.start_matlab()

# Add path if needed
eng.addpath(r'./', nargout=0)

eng.eval("initialiseScenario", nargout=0)

eng.eval("resetScenario", nargout=0)

for i in range(4):
    eng.eval("stepScenario", nargout=0)
    done = eng.workspace['done']
    pystate = eng.workspace['py_state']
    print(f"Step {i+1}, done: {done}")
    if done:
        break

# eng.eval("SaveData", nargout=0)

py_reward = eng.workspace['py_reward']
print(pystate)
% === Reset environment and get initial obs ===
obs_last_py = pyrun("from a2c_agent import reset_env; obs_last = reset_env()", "obs_last");
disp('Observation received from Python (py.list):');
disp(obs_last_py);
% Convert to MATLAB double column vector
obs_last = double(obs_last_py(:));
disp('Observation converted to MATLAB double:');
disp(obs_last);
whos obs_last % Check type and size

% === Choose action ===
disp('Attempting to call get_action with obs_last...');

% --- VERIFY obs_last JUST BEFORE THE CALL ---
if isempty(obs_last) || ~isnumeric(obs_last)
    error('MATLAB variable obs_last is empty or not numeric before calling pyrun!');
end
disp('Value of obs_last being passed to Python:');
disp(obs_last);
% --- END VERIFICATION ---

py_obs = py.list(obs_last); 

code = [
    "def process_data(data):", ...
    "    print('Received:', data)", ...
    "    return sum(data)", ...
    "", ...
    "result = process_data(obs_last)"
];

result = pyrun(strjoin(code, newline), "result", "obs_last", obs_last);
disp(result);

display('=======================NEW=====================')


code = [
    "from a2c_agent import get_action; action = get_action(obs_last)"
];

action = pyrun(strjoin(code, newline), "action", "obs_last", obs_last);
disp(result);

disp('Attempting to call get_action with obs_last...');
% Call pyrun - This is the line that currently causes the error
action = pyrun("from a2c_agent import get_action; action = get_action(py_obs)","action","py_obs",py_obs);


display('=-=-=-=-=-=------=======================NEW=====================')



action = py.a2c_agent.get_action(obs_last);
disp(result);

disp('Attempting to call get_action with obs_last...');
% Call pyrun - This is the line that currently causes the error
action = pyrun("from a2c_agent import get_action; action = get_action(py_obs)","action","py_obs",py_obs);                                


% --- This part will only be reached if pyrun succeeds ---
disp('Action received from Python (py object):');
disp(action)
% Convert the returned Python object (likely numpy array) to MATLAB double
action = double(action);
disp('Action converted to MATLAB double:');
disp(action);
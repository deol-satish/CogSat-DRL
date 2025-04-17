
obs_last = py.a2c_agent.reset_env()
display("obs_last-----")
disp(obs_last);
for i = 1:10


    action = py.a2c_agent.get_action(obs_last)

    display("action-----")
    display(action)
    
    
    display("my_step-----")
    py_tuple = py.a2c_agent.my_step(action)
    
    % Extract elements from the tuple
    next_obs = py_tuple{1};  % Python list
    reward = py_tuple{2};    % Python float
    done = py_tuple{3};      % Python bool
    
    py.a2c_agent.store_transition(reward, done, next_obs)

    obs_last = next_obs
    
    display(py.a2c_agent.step_count)
end






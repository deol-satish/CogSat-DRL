In MATLAB, when you write a function, the local variables defined within that function are generally confined to the function's workspace. However, if you want to make those local variables available in the base workspace (i.e., the command window's workspace), you can use the `assignin` function.

The `assignin` function allows you to assign a variable from the function workspace to the base workspace. For example:

```matlab
function myFunction()
    a = 10;
    b = 20;
    
    % Assign variables to the base workspace
    assignin('base', 'a', a);
    assignin('base', 'b', b);
end
```

In this example, `a` and `b` will be assigned to the base workspace after running `myFunction()`, making them accessible in the command window.

### To automatically push all local variables to the base workspace, you can use the following approach:

```matlab
function myFunction()
    a = 10;
    b = 20;
    c = 30;
    
    % Get a list of all variables in the current function workspace
    vars = who;
    
    % Loop through and assign each variable to the base workspace
    for i = 1:length(vars)
        assignin('base', vars{i}, eval(vars{i}));
    end
end
```

In this code:
- `who` retrieves all the variable names in the current workspace.
- `eval(vars{i})` evaluates the value of each variable.
- `assignin('base', vars{i}, ...)` assigns each variable to the base workspace.

This will push all local variables from the function into the base workspace. However, be cautious about potential name conflicts with existing variables in the base workspace.

Let me know if you need any further clarifications!
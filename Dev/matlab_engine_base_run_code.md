Yes, absolutely! If `init.m` and `step.m` are both **scripts** (not functions), and you're running them through the **MATLAB Engine**, then **you can call them multiple times**, and all the variables they define or modify in the **base workspace** will persist across those calls — just like if you were running them directly in MATLAB.

---

### ✅ How this would work in practice

Let’s say you have two script files:

#### `init.m` (initializes state)

```matlab
startIndex = 1;
endIndex = 5;
currentIndex = startIndex;
done = false;
```

#### `step.m` (steps through the index range)

```matlab
if currentIndex < endIndex
    currentIndex = currentIndex + 1;
    done = false;
else
    done = true;
end
```

---

### 🧪 Python Code Using MATLAB Engine

```python
import matlab.engine

# Start MATLAB
eng = matlab.engine.start_matlab()

# Add path if needed
eng.addpath(r'/path/to/mfiles', nargout=0)

# Run init.m to initialize state
eng.eval("init", nargout=0)

# Step through until done == True
while True:
    eng.eval("step", nargout=0)
    done = eng.workspace['done']
    current = eng.workspace['currentIndex']
    print(f"Current Index: {current}, Done: {done}")
    if done:
        break
```

---

### ✅ Notes

- All variables (`startIndex`, `endIndex`, `currentIndex`, `done`) live in the **base workspace** of the engine session.
- You can access or modify them in Python, and the changes persist across script calls.
- This works best when you treat the engine like a persistent MATLAB session.

---

### 🧠 Pro Tip: Debugging Variables

If you're unsure what’s in the workspace at any point:

```python
print(eng.eval("whos", nargout=1))
```

Let me know if you want help turning that logic into a loop controller class or script manager in Python.
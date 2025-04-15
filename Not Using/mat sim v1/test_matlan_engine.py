import matlab.engine

eng = matlab.engine.start_matlab()
res = eng.add_numbers(5.0, 3.0)
print("Result from MATLAB:", res)
eng.quit()

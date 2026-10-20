@echo off
if not exist bin mkdir bin
echo Compiling Task 1...
g++ -O3 -mavx2 -std=c++17 -DSTANDALONE_BUILD src\task1_series.cpp -o bin\task1_series.exe
if errorlevel 1 goto error

echo Compiling Task 2...
g++ -O3 -march=native -std=c++17 -DSTANDALONE_BUILD src\task2_matrix.cpp -o bin\task2_matrix.exe
if errorlevel 1 goto error

echo Build successful!
exit /b 0

:error
echo Build failed!
exit /b 1

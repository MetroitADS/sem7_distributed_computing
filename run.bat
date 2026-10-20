@echo off
chcp 65001 > nul
echo ========================================================
echo   Запуск заданий Лабораторной работы №1
echo ========================================================

echo.
echo --- ЗАДАНИЕ 1: Сумма ряда (std::thread + SIMD AVX2) ---
bin\task1_series.exe

echo.
echo.
echo --- ЗАДАНИЕ 2: Умножение матриц 4096 x 4096 (std::thread) ---
bin\task2_matrix.exe

echo.
echo Выполнение завершено.

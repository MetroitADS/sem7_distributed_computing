#include <iostream>
#include <windows.h>

int run_task1(int argc = 0, char* argv[] = nullptr);
int run_task2(int argc = 0, char* argv[] = nullptr);
int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    while (true) {
        std::cout << "\n====================================================================\n";
        std::cout << "  Лабораторная работа №1 (Вариант 16: Целочисленный 32 бит)\n";
        std::cout << "  1. Задание 1: Вычисление суммы ряда (std::thread + AVX2 SIMD)\n";
        std::cout << "  2. Задание 2: Умножение матриц 4096 x 4096 (std::thread)\n";
        std::cout << "  3. Выполнить оба задания подряд\n";
        std::cout << "  0. Выход\n";
        std::cout << "====================================================================\n";
        std::cout << "Выберите пункт (0-3): ";

        int choice = -1;
        std::cin >> choice;

        if (choice == 0) {
            break;
        } else if (choice == 1) {
            run_task1();
        } else if (choice == 2) {
            run_task2();
        } else if (choice == 3) {
            run_task1();
            std::cout << "\n\n";
            run_task2();
        } else {
            std::cout << "Неверный выбор. Пожалуйста, введите число от 0 до 3.\n";
        }
    }

    return 0;
}

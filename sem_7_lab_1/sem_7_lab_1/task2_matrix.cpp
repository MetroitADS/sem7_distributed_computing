#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <random>


constexpr int N = 4096;
constexpr int NUM_RUNS = 10;

struct RunStats {
    std::vector<double> times_ms;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms = 0.0;
    double stddev_ms = 0.0;

    void calculate() {
        if (times_ms.empty()) return;
        min_ms = times_ms[0];
        max_ms = times_ms[0];
        double sum = 0.0;
        for (double t : times_ms) {
            if (t < min_ms) min_ms = t;
            if (t > max_ms) max_ms = t;
            sum += t;
        }
        avg_ms = sum / times_ms.size();
        double variance = 0.0;
        for (double t : times_ms) {
            variance += (t - avg_ms) * (t - avg_ms);
        }
        stddev_ms = std::sqrt(variance / times_ms.size());
    }
};

// Функция транспонирования матрицы: поворачивает матрицу B, меняя строки и столбцы местами
// Зачем это нужно: в C++ матрицы хранятся построчно. При обычном умножении мы вынуждены читать
// столбцы матрицы B "прыжками" через каждые n элементов (промахиваясь мимо кэша процессора L1/L2)
// Транспонирование превращает столбцы матрицы B в строки матрицы BT
// В результате оба массива (A и BT) читаются процессором последовательно подряд
void transpose_matrix(const int32_t* src, int32_t* dst, int n) {
    for (int i = 0; i < n; ++i) {        
        for (int j = 0; j < n; ++j) {    
            // Берем элемент из строки i, столбца j и сохраняем его в строку j, столбец i
            dst[j * n + i] = src[i * n + j];
        }
    }
}

// 1. Однопоточное умножение матриц: C = A * B (используем транспонированную BT), для того, чтобы не умножать строки на столбцы, а умнажать строки на строки  просто делая j+1 для смены столбца
void multiply_single_thread(const int32_t* A, const int32_t* BT, int32_t* C, int n) {
    // Цикл 1: Перебираем строки результирующей матрицы C (от 0 до n-1)
    for (int i = 0; i < n; ++i) {
        const int32_t* a_row = &A[i * n];
        int32_t* c_row = &C[i * n]; // Адрес начала строки i матрицы C
        for (int j = 0; j < n; ++j) {
            // Берем адрес j-й строки матрицы BT (j-м столбцом матрицы B)
            const int32_t* bt_row = &BT[j * n];
            int32_t sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a_row[k] * bt_row[k];
            }
            c_row[j] = sum;
        }
    }
}

// Копия multiply_single_thread, но потоки получают свой кусок матрицы
void worker_multiply_rows(const int32_t* A, const int32_t* BT, int32_t* C, int n, int start_row, int end_row) {
    // Поток считает строки матрицы C строго от start_row до end_row-1
    for (int i = start_row; i < end_row; ++i) {
        const int32_t* a_row = &A[i * n];
        int32_t* c_row = &C[i * n];

        for (int j = 0; j < n; ++j) {
            const int32_t* bt_row = &BT[j * n];
            int32_t sum = 0;

            for (int k = 0; k < n; ++k) {
                sum += a_row[k] * bt_row[k];
            }

            c_row[j] = sum;
        }
    }
}

// 2. Многопоточное умножение матриц на основе std::thread
void multiply_multi_thread(const int32_t* A, const int32_t* BT, int32_t* C, int n, unsigned int num_threads) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int rows_per_thread = n / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? n : start_row + rows_per_thread;
        threads.emplace_back(worker_multiply_rows, A, BT, C, n, start_row, end_row);
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}

int run_task2(int argc = 0, char* argv[] = nullptr) {
    std::cout << "====================================================================\n";
    std::cout << "Умножение матриц 4096 x 4096\n";
    std::cout << "====================================================================\n";

    unsigned int hardware_threads = 16;
    std::cout << "Число ядер/потоков процессора: " << hardware_threads << "\n";
    std::cout << "Размерность матриц: " << N << " x " << N << "\n";
    std::cout << "Объем памяти для трех матриц (A, B, C): " 
              << (3LL * N * N * sizeof(int32_t)) / (1024 * 1024) << " МБ\n";
    std::cout << "Количество контрольных запусков: " << NUM_RUNS << "\n\n";

    std::cout << "Выделение памяти и инициализация матриц A и B... " << std::flush;
    std::vector<int32_t> A(static_cast<size_t>(N) * N);
    std::vector<int32_t> B(static_cast<size_t>(N) * N);
    std::vector<int32_t> BT(static_cast<size_t>(N) * N);
    std::vector<int32_t> C_single(static_cast<size_t>(N) * N, 0);
    std::vector<int32_t> C_multi(static_cast<size_t>(N) * N, 0);

    // Случайные числа по Дональда Кнута
    for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
        A[i] = static_cast<int32_t>((i * 1103515245u + 12345u) % 21) - 10;
        B[i] = static_cast<int32_t>((i * 1664525u + 1013904223u) % 21) - 10;
    }

    // Транспонирование B для ускорения последовательного доступа в кэш
    transpose_matrix(B.data(), BT.data(), N);
    std::cout << "Готово.\n\n";

    RunStats stats_single;
    RunStats stats_multi;

    std::cout << "Запуск тестирования производительности (" << NUM_RUNS << " запусков)...\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(6) << "Итер." 
              << std::setw(25) << "Однопоточный (мс)" 
              << std::setw(25) << "std::thread (мс)\n";
    std::cout << std::string(60, '-') << "\n";

    for (int run = 0; run < NUM_RUNS; ++run) {
        // Замер однопоточного выполнения
        auto t0 = std::chrono::high_resolution_clock::now();
        multiply_single_thread(A.data(), BT.data(), C_single.data(), N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt_single = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_single.times_ms.push_back(dt_single);

        // Замер многопоточного выполнения
        auto t2 = std::chrono::high_resolution_clock::now();
        multiply_multi_thread(A.data(), BT.data(), C_multi.data(), N, hardware_threads);
        auto t3 = std::chrono::high_resolution_clock::now();
        double dt_multi = std::chrono::duration<double, std::milli>(t3 - t2).count();
        stats_multi.times_ms.push_back(dt_multi);

        std::cout << std::setw(6) << (run + 1)
                  << std::setw(25) << std::fixed << std::setprecision(1) << dt_single
                  << std::setw(25) << std::fixed << std::setprecision(1) << dt_multi << "\n";
    }
    std::cout << std::string(60, '-') << "\n\n";

    stats_single.calculate();
    stats_multi.calculate();

    // Проверка корректности (поэлементное сравнение)
    std::cout << "Верификация результатов (проверка равенства матриц C_single и C_multi)... ";
    bool match = true;
    size_t mismatch_count = 0;
    for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
        if (C_single[i] != C_multi[i]) {
            match = false;
            mismatch_count++;
            if (mismatch_count <= 5) {
                std::cout << "\n  Несовпадение в элементе " << i << ": single=" 
                          << C_single[i] << " != multi=" << C_multi[i];
            }
        }
    }
    if (match) {
        std::cout << "[OK] Матрицы абсолютно идентичны!\n\n";
    } else {
        std::cout << "\n  [ERROR] Обнаружено " << mismatch_count << " несовпадений!\n\n";
    }

    std::cout << "Сводная таблица производительности умножения матриц:\n";
    std::cout << std::string(78, '=') << "\n";
    std::cout << std::left << std::setw(28) << "Метод"
              << std::right << std::setw(12) << "Мин. (мс)"
              << std::setw(12) << "Макс. (мс)"
              << std::setw(14) << "Среднее (мс)"
              << std::setw(12) << "Ускорение (S)\n";
    std::cout << std::string(78, '-') << "\n";

    double speedup = stats_single.avg_ms / stats_multi.avg_ms;

    std::cout << std::left << std::setw(28) << "1. Однопоточный"
              << std::right << std::setw(12) << std::fixed << std::setprecision(1) << stats_single.min_ms
              << std::setw(12) << stats_single.max_ms
              << std::setw(14) << stats_single.avg_ms
              << std::setw(12) << "1.00x\n";

    std::cout << std::left << std::setw(28) << ("2. std::thread (" + std::to_string(hardware_threads) + " потоков)")
              << std::right << std::setw(12) << std::fixed << std::setprecision(1) << stats_multi.min_ms
              << std::setw(12) << stats_multi.max_ms
              << std::setw(14) << stats_multi.avg_ms
              << std::setw(11) << std::setprecision(2) << speedup << "x\n";
    std::cout << std::string(78, '=') << "\n";

    return 0;
}

#if defined(STANDALONE_BUILD) || defined(BUILD_TASK2)
int main(int argc, char* argv[]) {
    return run_task2(argc, argv);
}
#endif

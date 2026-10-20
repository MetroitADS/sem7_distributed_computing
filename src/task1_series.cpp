#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <numeric>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <immintrin.h>


constexpr size_t N = 1000'000'000;
constexpr int NUM_RUNS = 10;

// Структура для сбора статистики запусков
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

// 1. Однопоточная реализация
int32_t sum_single_thread(const int32_t* data, size_t count) {
    int32_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        total += data[i];
    }
    return total;
}

// 2. Многопоточная реализация на основе std::thread
// Функция для одного рабочего потока
void worker_sum_scalar(const int32_t* data, size_t start, size_t end, int32_t* result_slot) {
    int32_t local_sum = 0;
    for (size_t i = start; i < end; ++i) {
        local_sum += data[i];
    }
    *result_slot = local_sum;
}

int32_t sum_multi_thread(const int32_t* data, size_t count, unsigned int num_threads) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);      // Выделение памяти под 16 потоков

    // Выделяем 64 байта, чтобы была занята полностью кеш линия. (1 кеш линия на 1 поток)
    struct alignas(64) PaddedSlot {
        int32_t value = 0;
    };
    std::vector<PaddedSlot> partial_sums(num_threads);


    // Делаем так, чтобы каждый поток получил кусок массива для вычисления.
    size_t chunk_size = count / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? count : start + chunk_size;
        threads.emplace_back(worker_sum_scalar, data, start, end, &partial_sums[t].value);
    }

    // Ожидание пока потоки ещё работаеют
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    //Складывание
    int32_t total = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        total += partial_sums[t].value;
    }
    return total;
}

// 3. SIMD-векторизованная функция с использованием AVX2 (8 чисел int32 за одну инструкцию)
int32_t sum_simd_avx2_block(const int32_t* data, size_t start, size_t end) {
    size_t count = end - start;
    const int32_t* ptr = data + start;
    __m256i vsum0 = _mm256_setzero_si256();
    __m256i vsum1 = _mm256_setzero_si256();
    __m256i vsum2 = _mm256_setzero_si256();
    __m256i vsum3 = _mm256_setzero_si256();
    size_t i = 0;

    // Обрабатываем блоки по 32 элемента (4 регистра по 8 элементов int32_t)
    for (; i + 32 <= count; i += 32) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i + 8));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i + 16));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i + 24));

        vsum0 = _mm256_add_epi32(vsum0, v0);
        vsum1 = _mm256_add_epi32(vsum1, v1);
        vsum2 = _mm256_add_epi32(vsum2, v2);
        vsum3 = _mm256_add_epi32(vsum3, v3);
    }

    // Складываем все 4 вектора в один
    vsum0 = _mm256_add_epi32(vsum0, vsum1);
    vsum2 = _mm256_add_epi32(vsum2, vsum3);
    vsum0 = _mm256_add_epi32(vsum0, vsum2);

    // Обрабатываем остаток по 8 элементов
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i));
        vsum0 = _mm256_add_epi32(vsum0, v);
    }

    // Горизонтальное суммирование 8 элементов 256-битного регистра
    alignas(32) int32_t buffer[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), vsum0);
    int32_t local_sum = 0;
    for (int k = 0; k < 8; ++k) {
        local_sum += buffer[k];
    }

    // Если осталось число меньше 8, то просто добираем его при помощи сложения
    for (; i < count; ++i) {
        local_sum += ptr[i];
    }

    return local_sum;
}

// Функция для потока с SIMD
void worker_sum_simd(const int32_t* data, size_t start, size_t end, int32_t* result_slot) {
    *result_slot = sum_simd_avx2_block(data, start, end);
}

// 4. Многопоточная реализация со встроенным SIMD (std::thread + AVX2)
int32_t sum_multi_thread_simd(const int32_t* data, size_t count, unsigned int num_threads) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    struct alignas(64) PaddedSlot {
        int32_t value = 0;
    };
    std::vector<PaddedSlot> partial_sums(num_threads);

    size_t chunk_size = count / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? count : start + chunk_size;
        threads.emplace_back(worker_sum_simd, data, start, end, &partial_sums[t].value);
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    int32_t total = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        total += partial_sums[t].value;
    }
    return total;
}

int run_task1(int argc = 0, char* argv[] = nullptr) {
    // Настройка консоли
    std::cout << "-====================================================================-\n";
    std::cout << "  Задание 1. Вычисление суммы ряда\n";
    std::cout << "-====================================================================-\n";

    unsigned int hardware_threads =  16;
    std::cout << "Обнаружено аппаратных потоков процессора: " << hardware_threads << "\n";
    std::cout << "Размер обрабатываемого массива: " << N << " элементов (" << (N * sizeof(int32_t)) / (1024 * 1024) << " МБ)\n";
    std::cout << "Количество контрольных запусков: " << NUM_RUNS << "\n\n";

    // Генерация тестового массива данных
    std::cout << "Инициализация тестовых данных... " << std::flush;
    std::vector<int32_t> data(N);
    // Случайные числа по Дональда Кнута
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int32_t>((i * 1664525u + 1013904223u) % 2001) - 1000;
    }
    std::cout << "Готово.\n\n";

    RunStats stats_single;
    RunStats stats_multi;
    RunStats stats_simd;

    int32_t result_single = 0;
    int32_t result_multi = 0;
    int32_t result_simd = 0;

    std::cout << "Запуск тестов быстродействия (" << NUM_RUNS << " итераций для каждого метода)...\n";
    std::cout << std::string(75, '-') << "\n";
    std::cout << std::setw(6) << "Итер." 
              << std::setw(22) << "Однопоточный (мс)" 
              << std::setw(23) << "std::thread (мс)" 
              << std::setw(24) << "thread + SIMD (мс)\n";
    std::cout << std::string(75, '-') << "\n";

    for (int run = 0; run < NUM_RUNS; ++run) {
        // Однопоточный замер
        auto t0 = std::chrono::high_resolution_clock::now();
        result_single = sum_single_thread(data.data(), N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt_single = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_single.times_ms.push_back(dt_single);

        // Многопоточный замер
        auto t2 = std::chrono::high_resolution_clock::now();
        result_multi = sum_multi_thread(data.data(), N, hardware_threads);
        auto t3 = std::chrono::high_resolution_clock::now();
        double dt_multi = std::chrono::duration<double, std::milli>(t3 - t2).count();
        stats_multi.times_ms.push_back(dt_multi);

        // Многопоточный + SIMD замер
        auto t4 = std::chrono::high_resolution_clock::now();
        result_simd = sum_multi_thread_simd(data.data(), N, hardware_threads);
        auto t5 = std::chrono::high_resolution_clock::now();
        double dt_simd = std::chrono::duration<double, std::milli>(t5 - t4).count();
        stats_simd.times_ms.push_back(dt_simd);

        std::cout << std::setw(6) << (run + 1)
                  << std::setw(22) << std::fixed << std::setprecision(2) << dt_single
                  << std::setw(23) << std::fixed << std::setprecision(2) << dt_multi
                  << std::setw(24) << std::fixed << std::setprecision(2) << dt_simd << "\n";
    }
    std::cout << std::string(75, '-') << "\n\n";

    stats_single.calculate();
    stats_multi.calculate();
    stats_simd.calculate();

    std::cout << "Проверка корректности вычислений:\n";
    std::cout << "  Сумма (однопоточная):      " << result_single << "\n";
    std::cout << "  Сумма (std::thread):       " << result_multi << "\n";
    std::cout << "  Сумма (thread + SIMD):     " << result_simd << "\n";
    if (result_single == result_multi && result_multi == result_simd) {
        std::cout << "  [OK] Все методы сошлись и вернули идентичные суммы!\n\n";
    } else {
        std::cout << "  [ERROR] Результаты методов различаются!\n\n";
    }

    std::cout << "Сводная таблица производительности:\n";
    std::cout << std::string(78, '=') << "\n";
    std::cout << std::left << std::setw(26) << "Метод"
              << std::right << std::setw(12) << "Мин. (мс)"
              << std::setw(12) << "Макс. (мс)"
              << std::setw(14) << "Среднее (мс)"
              << std::setw(14) << "Ускорение (S)\n";
    std::cout << std::string(78, '-') << "\n";

    double speedup_multi = stats_single.avg_ms / stats_multi.avg_ms;
    double speedup_simd = stats_single.avg_ms / stats_simd.avg_ms;

    std::cout << std::left << std::setw(26) << "1. Однопоточный"
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << stats_single.min_ms
              << std::setw(12) << stats_single.max_ms
              << std::setw(14) << stats_single.avg_ms
              << std::setw(14) << "1.00x\n";

    std::cout << std::left << std::setw(26) << "2. std::thread"
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << stats_multi.min_ms
              << std::setw(12) << stats_multi.max_ms
              << std::setw(14) << stats_multi.avg_ms
              << std::setw(13) << speedup_multi << "x\n";

    std::cout << std::left << std::setw(26) << "3. thread + SIMD (AVX2)"
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << stats_simd.min_ms
              << std::setw(12) << stats_simd.max_ms
              << std::setw(14) << stats_simd.avg_ms
              << std::setw(13) << speedup_simd << "x\n";
    std::cout << std::string(78, '=') << "\n";

    return 0;
}

#if defined(STANDALONE_BUILD) || defined(BUILD_TASK1)
int main(int argc, char* argv[]) {
    return run_task1(argc, argv);
}
#endif

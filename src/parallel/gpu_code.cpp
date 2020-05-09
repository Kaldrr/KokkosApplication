#include <array>
#include <random>

#include <Kokkos_Core.hpp>
#include <Kokkos_View.hpp>

#include <parallel/gpu_code.h>

namespace gpu {

namespace matrix {

    Kokkos::View<double**> getRandomMatrix(std::size_t height, std::size_t width, std::string_view name)
    {
        std::mt19937 generator { std::random_device {}() };
        std::uniform_real_distribution<double> rng { 1.0, 100.0 };

        Kokkos::View<double**> matrix { std::string { name }, height, width };

        for (std::size_t i = 0; i < height; ++i) {
            for (std::size_t j = 0; j < width; ++j) {
                matrix(i, j) = rng(generator);
            }
        }

        return matrix;
    }

    bool equals(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs)
    {
        const std::array<std::size_t, 2> leftMatrixSize { lhs.extent(0), lhs.extent(1) };
        const std::array<std::size_t, 2> rightMatrixSize { rhs.extent(0), rhs.extent(1) };

        if (leftMatrixSize != rightMatrixSize) {
            return false;
        }

        constexpr double epsilon = 0.0001;

        for (std::size_t i { 0 }; i < leftMatrixSize[0]; ++i) {
            for (std::size_t j { 0 }; j < leftMatrixSize[1]; ++j) {
                if (std::abs(lhs(i, j) - rhs(i, j)) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

}

std::map<std::string, double> measureMatrixTimes(std::pair<int, int> leftMatrixSize, std::pair<int, int> rightMatrixSize)
{
    using Matrix = Kokkos::View<double**>;

    constexpr int teamSize = 8;

    constexpr auto compareMatrix = [](const Matrix& reference, const Matrix& tested, std::string_view leftName, std::string_view rightName) {
        // if (!matrix::equals(reference, tested)) {
        //     std::cout << leftName << " and " << rightName << " differ!\n";
        // }
    };

    const Matrix x = matrix::getRandomMatrix(leftMatrixSize.first, leftMatrixSize.second);
    const Matrix y = matrix::getRandomMatrix(rightMatrixSize.first, rightMatrixSize.second);

    std::map<std::string, double> parallelTimes {};
    Kokkos::Timer timer {};

    timer.reset();

    Matrix sequentialOutput { "HostMatrix", static_cast<size_t>(leftMatrixSize.first), static_cast<size_t>(rightMatrixSize.second) };
    for (std::size_t i = 0; i < leftMatrixSize.first; ++i) {
        for (std::size_t j = 0; j < rightMatrixSize.second; ++j) {
            sequentialOutput(i, j) = 0;
            for (std::size_t k = 0; k < leftMatrixSize.second; ++k) {
                sequentialOutput(i, j) += x(i, k) * y(k, j);
            }
        }
    }
    parallelTimes.insert({ "Sequential", timer.seconds() });

    timer.reset();
    {
        const Matrix output = matrix::matrixMultiply(x, y);
        parallelTimes.insert({ "MDRangePolicy", timer.seconds() });
        compareMatrix(sequentialOutput, output, "SequentialMultiply", "MDRangePolicy");
    }

    timer.reset();
    {
        const Matrix output = matrix::matrixMultiply<Kokkos::RandomAccess>(x, y);
        parallelTimes.insert({ "MDRangePolicyAndTraits", timer.seconds() });
        compareMatrix(sequentialOutput, output, "SequentialMultiply", "MDRangePolicyAndTraits");
    }

    timer.reset();
    {
        const Matrix output = matrix::multiplyTeamBased(x, y, teamSize);
        parallelTimes.insert({ "TeamPolicy", timer.seconds() });
        compareMatrix(sequentialOutput, output, "SequentialMultiply", "TeamPolicy");
    }

    timer.reset();
    {
        const Matrix output = matrix::multiplyTeamBased<Kokkos::RandomAccess>(x, y, teamSize);
        parallelTimes.insert({ "TeamPolicyWithTraits", timer.seconds() });
        compareMatrix(sequentialOutput, output, "SequentialMultiply", "TeamPolicyWithTraits");
    }

    timer.reset();
    {
        const Matrix output = matrix::multiplyTeamBasedSharedMemory(x, y, teamSize);
        parallelTimes.insert({ "TeamPolicyWithSharedMemory", timer.seconds() });
        compareMatrix(sequentialOutput, output, "SequentialMultiply", "TeamBasedSharedMemory");
    }

    timer.reset();
    {
        const Matrix output = matrix::multiplyTeamBasedSharedMemory<Kokkos::RandomAccess>(x, y, teamSize);
        parallelTimes.insert({ "TeamPolicyWithSharedMemoryAndTraits", timer.seconds() });
        compareMatrix(sequentialOutput, output, "SequentialMultiply", "TeamPolicyWithSharedMemoryAndTraits");
    }

    return parallelTimes;
}

Kokkos::View<int**> gpu::nussinov::sequential(std::string_view rnaChain)
{
    const auto bond = [rnaChain](std::size_t i, std::size_t j) {
        constexpr std::array allowedPairings {
            std::pair { 'A', 'U' },
            std::pair { 'U', 'A' },
            std::pair { 'C', 'G' },
            std::pair { 'G', 'C' },
            std::pair { 'G', 'U' },
            std::pair { 'U', 'G' }
        };

        const auto iterator = std::find(allowedPairings.cbegin(), allowedPairings.cend(),
            std::make_pair(rnaChain[i], rnaChain[j]));
        const bool found = iterator != allowedPairings.cend();
        return found ? 1 : 0;
    };

    const int N = rnaChain.length();

    Kokkos::View<int**> output { "output", static_cast<std::size_t>(N), static_cast<std::size_t>(N) };

    for (int i = N - 1; i >= 0; --i) {
        for (int j = i + 1; j < N; ++j) {
            for (int k = 0; k < j - i; ++k) {
                output(i, j) = std::max(output(i, k + i) + output(k + i + 1, j), output(i, j));
            }
            output(i, j) = std::max(output(i, j), output(i + 1, j - 1) + bond(i, j));
        }
    }

    return output;
}

std::map<std::string, double> measureNussinovTimes(std::string_view rnaSequence)
{
    constexpr std::size_t teamSize = 32;
    using Output = Kokkos::View<int**>;

    Kokkos::Timer timer {};
    std::map<std::string, double> times {};

    timer.reset();
    const Output sequentialOutput = gpu::nussinov::sequential(rnaSequence);
    times.insert({ "Sequential", timer.seconds() });

    {
        timer.reset();
        const Output parallelOutput = gpu::nussinov::teamBased(teamSize, rnaSequence);
        times.insert({ "TeamBased", timer.seconds() });
    }
    {
        timer.reset();
        const Output parallelOutput = gpu::nussinov::teamBased<Kokkos::RandomAccess>(teamSize, rnaSequence);
        times.insert({ "TeamBasedTraits", timer.seconds() });
    }
    {
        timer.reset();
        const Output parallelOutput = gpu::nussinov::rangePolicy(rnaSequence);
        times.insert({ "RangePolicy", timer.seconds() });
    }

    for (const auto& [name, time] : times) {
        std::cout << name << ": " << time << '\n';
    }

    return times;
}
}

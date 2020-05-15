#pragma once

#include <Kokkos_Cuda.hpp>
#include <Kokkos_CudaSpace.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_View.hpp>
#include <Kokkos_CopyViews.hpp>

#include <map>
#include <string_view>
#include <utility>

namespace gpu {
namespace matrix {
    Kokkos::View<float**> getRandomMatrix(std::size_t height, std::size_t width,
        std::string_view name = "RandomMatrix");
    bool equals(const Kokkos::View<float**>& lhs,
        const Kokkos::View<float**>& rhs);

    template <unsigned... Traits>
    Kokkos::View<float**> matrixMultiply(const Kokkos::View<float**>& lhs,
        const Kokkos::View<float**>& rhs);

    template <unsigned int... Traits>
    Kokkos::View<float**> multiplyTeamBased(const Kokkos::View<float**>& lhs,
        const Kokkos::View<float**>& rhs,
        std::size_t teamSize);

    template <unsigned int... Traits>
    Kokkos::View<float**>
    multiplyTeamBasedSharedMemory(const Kokkos::View<float**>& lhs,
        const Kokkos::View<float**>& rhs,
        std::size_t teamSize);
} // namespace matrix
namespace nussinov {
    Kokkos::View<int**> sequential(std::string_view rnaChain);

    template <unsigned int... Traits>
    Kokkos::View<int**> teamBased(std::size_t teamSize, std::string_view rnaChain);

    template <unsigned int... Traits>
    Kokkos::View<int**> rangePolicy(std::string_view rnaChain);
} // namespace nussinov

std::map<std::string, double>
measureMatrixTimes(std::pair<int, int> leftMatrixSize,
    std::pair<int, int> rightMatrixSize);

std::map<std::string, double>
measureNussinovTimes(std::string_view rnaSequence);
} // namespace gpu

template <unsigned... Traits>
Kokkos::View<float**>
gpu::matrix::matrixMultiply(const Kokkos::View<float**>& lhs,
    const Kokkos::View<float**>& rhs)
{
    using ViewTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;

    using DataView = Kokkos::View<float**, ViewTraits>;
    using ConstView = Kokkos::View<const float**, ViewTraits>;

    const ConstView leftMatrix = lhs;
    const ConstView rightMatrix = rhs;

    const std::array<std::size_t, 2> leftSize { lhs.extent(0), lhs.extent(1) };
    const std::array<std::size_t, 2> rightSize { rhs.extent(0), rhs.extent(1) };

    const Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy {
        { 0, 0 }, { leftSize[0], rightSize[1] }
    };

    const DataView outputMatrix { "OutputMatrix", leftSize[0], rightSize[1] };

    Kokkos::parallel_for(
        "MatrixMultiply", policy,
        KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < leftSize[1]; ++k) {
                sum += leftMatrix(i, k) * rightMatrix(k, j);
            }
            outputMatrix(i, j) = sum;
        });
    Kokkos::fence();

    return outputMatrix;
}

template <unsigned int... Traits>
Kokkos::View<float**>
gpu::matrix::multiplyTeamBased(const Kokkos::View<float**>& lhs,
    const Kokkos::View<float**>& rhs,
    std::size_t teamSize)
{
    using ViewTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;

    using DataView = Kokkos::View<float**, ViewTraits>;
    using ConstView = Kokkos::View<const float**, ViewTraits>;

    const ConstView lhsConstView = lhs;
    const ConstView rhsConstView = rhs;

    const std::array<std::size_t, 2> leftSize { lhs.extent(0), lhs.extent(1) };
    const std::array<std::size_t, 2> rightSize { rhs.extent(0), rhs.extent(1) };

    const DataView output { "TeamBasedOutput", leftSize[0], rightSize[1] };

    const std::size_t teamsPerRow = (rightSize[1] + teamSize - 1) / teamSize;
    const std::size_t teamsPerColumn = (leftSize[0] + teamSize - 1) / teamSize;
    const std::size_t numberOfTeams = teamsPerRow * teamsPerColumn;

    const auto policy = Kokkos::TeamPolicy<> { static_cast<int>(numberOfTeams), static_cast<int>(teamSize * teamSize) };

    Kokkos::parallel_for(
        "TeamBasedMultiply", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const std::size_t blockRow = (team.league_rank() / teamsPerRow) * teamSize;
            const std::size_t blockCol = (team.league_rank() % teamsPerRow) * teamSize;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, teamSize * teamSize),
                KOKKOS_LAMBDA(int index) {
                    const std::size_t threadRow = blockRow + index / teamSize;
                    const std::size_t threadCol = blockCol + index % teamSize;

                    if (threadRow < leftSize[0] && threadCol < rightSize[1]) {
                        float threadValue = 0.0f;
                        for(int j = 0; j < leftSize[1]; ++j){
                            threadValue += lhsConstView(threadRow, j) * rhsConstView(j, threadCol);
                        }
                        output(threadRow, threadCol) = threadValue;
                    }
                });
        });
    Kokkos::fence();

    return output;
}

template <unsigned int... Traits>
Kokkos::View<float**>
gpu::matrix::multiplyTeamBasedSharedMemory(const Kokkos::View<float**>& lhs,
    const Kokkos::View<float**>& rhs,
    std::size_t teamSize)
{
    using ViewTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;
    using ScratchMemorySpace = Kokkos::Cuda::scratch_memory_space;

    using DataView = Kokkos::View<float**, ViewTraits>;
    using ConstView = Kokkos::View<const float**, ViewTraits>;
    using ScratchMemory = Kokkos::View<float**, ScratchMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    const ConstView lhsConstView = lhs;
    const ConstView rhsConstView = rhs;

    const std::array<std::size_t, 2> leftSize { lhs.extent(0), lhs.extent(1) };
    const std::array<std::size_t, 2> rightSize { rhs.extent(0), rhs.extent(1) };

    const DataView output { "TeamBasedOutput", leftSize[0], rightSize[1] };

    const std::size_t teamsPerRow = (rightSize[1] + teamSize - 1) / teamSize;
    const std::size_t teamsPerColumn = (leftSize[0] + teamSize - 1) / teamSize;
    const std::size_t numberOfTeams = teamsPerRow * teamsPerColumn;

    constexpr std::size_t memoryLevel = 0;
    const std::size_t scratchMemorySize = sizeof(float) * teamSize * teamSize * 2;

    const auto policy = Kokkos::TeamPolicy<> { static_cast<int>(numberOfTeams), static_cast<int>(teamSize * teamSize) }
                            .set_scratch_size(memoryLevel, Kokkos::PerTeam(static_cast<int>(scratchMemorySize)));

    Kokkos::parallel_for(
        "TeamBasedMultiply", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const std::size_t blockRow = (team.league_rank() / teamsPerRow) * teamSize;
            const std::size_t blockCol = (team.league_rank() % teamsPerRow) * teamSize;
            const std::size_t tilesCount = (leftSize[1] + teamSize - 1) / teamSize;

            const ScratchMemory leftTile { team.team_scratch(memoryLevel), teamSize, teamSize };
            const ScratchMemory rightTile { team.team_scratch(memoryLevel), teamSize, teamSize };

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, teamSize * teamSize),
                KOKKOS_LAMBDA(int index) {
                    const std::size_t threadRow = index / teamSize;
                    const std::size_t threadCol = index % teamSize;
                    float threadSum = 0.0f;

                    for (std::size_t currentTile = 0; currentTile < tilesCount; ++currentTile) {
                        const std::size_t tileOffset = currentTile * teamSize;

                        if (threadRow + blockRow < leftSize[0] && threadCol + tileOffset < leftSize[1]) {
                            leftTile(threadRow, threadCol) = lhsConstView(threadRow + blockRow, threadCol + tileOffset);
                        } else {
                            leftTile(threadRow, threadCol) = 0.0f;
                        }

                        if (threadRow + tileOffset < rightSize[0] && threadCol + blockCol < rightSize[1]) {
                            rightTile(threadRow, threadCol) = rhsConstView(threadRow + tileOffset, threadCol + blockCol);
                        } else {
                            rightTile(threadRow, threadCol) = 0.0f;
                        }

                        team.team_barrier();

                        for (int j = 0; j < teamSize; ++j) {
                            threadSum += leftTile(threadRow, j) * rightTile(j, threadCol);
                        }
                        team.team_barrier();
                    }

                    if (threadRow + blockRow < leftSize[0] && threadCol + blockCol < rightSize[1]) {
                        output(threadRow + blockRow, threadCol + blockCol) = threadSum;
                    }
                });
        });
    Kokkos::fence();

    return output;
}

template <unsigned int... Traits>
Kokkos::View<int**> gpu::nussinov::teamBased(std::size_t teamSize,
    std::string_view rnaChain)
{
    ///* Start of CLooG code */
    //    if (N >= 2) {
    //        for (t2=1;t2<=N-1;t2++) {
    //            lbp=t2;
    //            ubp=N-1;
    //#pragma omp parallel for private(t4,t6) shared(t2)
    //            for (t4=t2;t4<=N-1;t4++) {
    //                for (t6=0;t6<=t2-1;t6++) {
    //                    S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t6+(-t2+t4)] +
    //                    S[t6+(-t2+t4)+1][t4], S[(-t2+t4)][t4]);;
    //                }
    //                S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t4], S[(-t2+t4)+1][t4-1])
    //                + MAX((-t2+t4),t4);;
    //            }
    //        }
    //    }
    using Policy = Kokkos::TeamPolicy<>;
    using Team = Policy::member_type;
    using MemoryTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;

    const int N = rnaChain.length();
    Kokkos::View<int**, MemoryTraits> output {
        "output", static_cast<std::size_t>(N), static_cast<std::size_t>(N)
    };

    const Kokkos::View<char*, MemoryTraits> deviceRnaChain { "deviceRna",
        rnaChain.length() };
    const Kokkos::View<Kokkos::pair<char, char>*> allowedPairings {
        "allowedPairings", 6
    };
    allowedPairings(0) = { 'A', 'U' };
    allowedPairings(1) = { 'U', 'A' };
    allowedPairings(2) = { 'C', 'G' };
    allowedPairings(3) = { 'G', 'C' };
    allowedPairings(4) = { 'G', 'U' };
    allowedPairings(5) = { 'U', 'G' };

    std::memcpy(deviceRnaChain.data(), rnaChain.data(),
        rnaChain.length() * sizeof(char));

    const Kokkos::View<const char*, MemoryTraits> constDeviceRnaChain = deviceRnaChain;
    const Kokkos::View<const Kokkos::pair<char, char>*, MemoryTraits>
        constAllowedPairings = allowedPairings;

    for (int t2 = 1; t2 <= N - 1; ++t2) {
        const std::size_t groupsCount = (N - t2 + (teamSize - 1)) / teamSize;

        Kokkos::parallel_for(
            "nussinovFor",
            Policy { static_cast<int>(groupsCount), static_cast<int>(teamSize) },
            KOKKOS_LAMBDA(const Team& team) {
                const auto bond = [=](std::size_t i, std::size_t j) {
                    const Kokkos::pair<char, char> myPair { constDeviceRnaChain(i),
                        constDeviceRnaChain(j) };
                    for (int k = 0; k < constAllowedPairings.extent(0); ++k) {
                        if (myPair == constAllowedPairings(k)) {
                            return 1;
                        }
                    }
                    return 0;
                };

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, teamSize), KOKKOS_LAMBDA(int i) {
                        const int t4 = t2 + (team.league_rank() * teamSize) + i;
                        if (((-t2 + t4) < output.extent(0)) && (t4 < output.extent(0))) {
                            int maxValue = output(-t2 + t4, t4);
                            for(int j = 0; j < t2; ++j){
                                maxValue = std::max(output((-t2 + t4), j + (-t2 + t4)) + output(j + (-t2 + t4) + 1, t4), maxValue);
                            }
                            output((-t2 + t4), t4) = std::max(maxValue, output((-t2 + t4) + 1, t4 - 1) + bond((-t2 + t4), t4));
                        }
                    });
            });

        Kokkos::fence();
    }
    return output;
}

template <unsigned int... Traits>
Kokkos::View<int**> gpu::nussinov::rangePolicy(std::string_view rnaChain)
{
    ///* Start of CLooG code */
    //    if (N >= 2) {
    //        for (t2=1;t2<=N-1;t2++) {
    //            lbp=t2;
    //            ubp=N-1;
    //#pragma omp parallel for private(t4,t6) shared(t2)
    //            for (t4=t2;t4<=N-1;t4++) {
    //                for (t6=0;t6<=t2-1;t6++) {
    //                    S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t6+(-t2+t4)] +
    //                    S[t6+(-t2+t4)+1][t4], S[(-t2+t4)][t4]);;
    //                }
    //                S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t4], S[(-t2+t4)+1][t4-1])
    //                + MAX((-t2+t4),t4);;
    //            }
    //        }
    //    }
    using Policy = Kokkos::RangePolicy<std::size_t>;
    using Team = Policy::member_type;

    const int N = rnaChain.length();
    Kokkos::View<int**> output { "output", static_cast<std::size_t>(N),
        static_cast<std::size_t>(N) };

    const Kokkos::View<char*> deviceRnaChain { "deviceRna", rnaChain.length() };
    const Kokkos::View<Kokkos::pair<char, char>*> allowedPairings {
        "allowedPairings", 6
    };
    allowedPairings(0) = { 'A', 'U' };
    allowedPairings(1) = { 'U', 'A' };
    allowedPairings(2) = { 'C', 'G' };
    allowedPairings(3) = { 'G', 'C' };
    allowedPairings(4) = { 'G', 'U' };
    allowedPairings(5) = { 'U', 'G' };

    std::memcpy(deviceRnaChain.data(), rnaChain.data(),
        rnaChain.length() * sizeof(char));

    for (int t2 = 1; t2 <= N - 1; ++t2) {
        Kokkos::parallel_for(
            "nussinovFor", Policy { 0, static_cast<std::size_t>(N - t2) },
            KOKKOS_LAMBDA(std::size_t index) {
                const auto bond = [=](std::size_t i, std::size_t j) {
                    const Kokkos::pair<char, char> myPair { deviceRnaChain(i),
                        deviceRnaChain(j) };
                    for (int k = 0; k < allowedPairings.extent(0); ++k) {
                        if (myPair == allowedPairings(k)) {
                            return 1;
                        }
                    }
                    return 0;
                };

                const int t4 = t2 + index;
                int maxValue = output(-t2 + t4, t4);
                for (int j = 0; j < t2; ++j) {
                    maxValue = std::max(output((-t2 + t4), j + (-t2 + t4)) + output(j + (-t2 + t4) + 1, t4),
                        maxValue);
                }
                output((-t2 + t4), t4) = std::max(
                    maxValue, output((-t2 + t4) + 1, t4 - 1) + bond((-t2 + t4), t4));
            });
        Kokkos::fence();
    }
    return output;
}
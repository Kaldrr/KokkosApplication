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
    Kokkos::View<double**> getRandomMatrix(std::size_t height, std::size_t width, std::string_view name = "RandomMatrix");
    bool equals(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs);

    template <unsigned... Traits>
    Kokkos::View<double**> matrixMultiply(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs);

    template <unsigned int... Traits>
    Kokkos::View<double**> multiplyTeamBased(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs, std::size_t teamSize);

    template <unsigned int... Traits>
    Kokkos::View<double**> multiplyTeamBasedSharedMemory(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs, std::size_t teamSize);
}
namespace nussinov {
    Kokkos::View<int**> sequential(std::string_view rnaChain);

    template <unsigned int... Traits>
    Kokkos::View<int**> teamBased(std::size_t teamSize, std::string_view rnaChain);

    template <unsigned int... Traits>
    Kokkos::View<int**> rangePolicy(std::string_view rnaChain);
}

std::map<std::string, double> measureMatrixTimes(std::pair<int, int> leftMatrixSize, std::pair<int, int> rightMatrixSize);

std::map<std::string, double> measureNussinovTimes(std::string_view rnaSequence);
}

template <unsigned... Traits>
Kokkos::View<double**> gpu::matrix::matrixMultiply(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs)
{
    using ViewTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;

    using DataView = Kokkos::View<double**, ViewTraits>;
    using ConstView = Kokkos::View<const double**, ViewTraits>;

    const ConstView leftMatrix = lhs;
    const ConstView rightMatrix = rhs;

    const std::array<std::size_t, 2> leftSize { lhs.extent(0), lhs.extent(1) };
    const std::array<std::size_t, 2> rightSize { rhs.extent(0), rhs.extent(1) };

    const Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy { { 0, 0 },
        { leftSize[0], rightSize[1] } };

    DataView outputMatrix { "OutputMatrix", leftSize[0], rightSize[1] };

    Kokkos::parallel_for(
        "MatrixMultiply",
        policy,
        KOKKOS_LAMBDA(const std::size_t i, const std::size_t j) {
            outputMatrix(i, j) = 0.0;
            for (std::size_t k = 0; k < leftSize[1]; ++k) {
                outputMatrix(i, j) += leftMatrix(i, k) * rightMatrix(k, j);
            }
        });
    Kokkos::fence();

    return outputMatrix;
}

template <unsigned int... Traits>
Kokkos::View<double**> gpu::matrix::multiplyTeamBased(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs, std::size_t teamSize)
{
    using ViewTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;

    using DataView = Kokkos::View<double**, ViewTraits>;
    using ConstView = Kokkos::View<const double**, ViewTraits>;

    const ConstView lhs_const_view = lhs;
    const ConstView rhs_const_view = rhs;

    const std::array<std::size_t, 2> leftSize { lhs.extent(0), lhs.extent(1) };
    const std::array<std::size_t, 2> rightSize { rhs.extent(0), rhs.extent(1) };

    DataView output { "TeamBasedOutput", leftSize[0], rightSize[1] };

    const int teamsPerRow = static_cast<int>(rightSize[1] / teamSize + (rightSize[1] % teamSize == 0 ? 0 : 1));
    const int teamsPerColumn = static_cast<int>(leftSize[0] / teamSize + (leftSize[0] % teamSize == 0 ? 0 : 1));
    const int numberOfTeams = static_cast<int>(teamsPerRow * teamsPerColumn);

    const auto policy = Kokkos::TeamPolicy<> { numberOfTeams, static_cast<int>(teamSize * teamSize) };

    Kokkos::parallel_for(
        "TeamBasedMultiply", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int teamNumber = team.league_rank();
            const int row = static_cast<int>((teamNumber / teamsPerRow) * teamSize);
            const int column = static_cast<int>((teamNumber % teamsPerRow) * teamSize);

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, teamSize * teamSize),
                KOKKOS_LAMBDA(int i) {
                    const int threadRow = i / teamSize;
                    const int threadColumn = i % teamSize;
                    if (((threadRow + row) < leftSize[0]) && ((threadColumn + column) < rightSize[1])) {
                        double product = 0.0;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, leftSize[0]),
                            KOKKOS_LAMBDA(int j, double& value) {
                                value += lhs_const_view(threadRow + row, j) * rhs_const_view(j, threadColumn + column);
                            },
                            Kokkos::Sum<double>(product));
                            output(threadRow + row, threadColumn + column) = product;
                    }
                });
        });
    Kokkos::fence();

    return output;
}

template <unsigned int... Traits>
Kokkos::View<double**> gpu::matrix::multiplyTeamBasedSharedMemory(const Kokkos::View<double**>& lhs, const Kokkos::View<double**>& rhs, std::size_t teamSize)
{
    using ViewTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;
    using ScratchMemorySpace = Kokkos::Cuda::scratch_memory_space;

    using DataView = Kokkos::View<double**, ViewTraits>;
    using ConstView = Kokkos::View<const double**, ViewTraits>;
    using ScratchMemory = Kokkos::View<double**, ScratchMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    const ConstView lhs_const_view = lhs;
    const ConstView rhs_const_view = rhs;

    const std::array<std::size_t, 2> leftSize { lhs.extent(0), lhs.extent(1) };
    const std::array<std::size_t, 2> rightSize { rhs.extent(0), rhs.extent(1) };

    const DataView output { "TeamBasedOutput", leftSize[0], rightSize[1] };

    const int teamsPerRow = static_cast<int>(rightSize[1] / teamSize + (rightSize[1] % teamSize == 0 ? 0 : 1));
    const int teamsPerColumn = static_cast<int>(leftSize[0] / teamSize + (leftSize[0] % teamSize == 0 ? 0 : 1));
    const int numberOfTeams = static_cast<int>(teamsPerRow * teamsPerColumn);

    constexpr int memoryLevel = 0;
    const int scratchMemorySize = sizeof(double) * teamSize * teamSize;

    const auto policy = Kokkos::TeamPolicy<> { numberOfTeams, static_cast<int>(teamSize * teamSize) }
      .set_scratch_size(memoryLevel, Kokkos::PerTeam(scratchMemorySize * 2), Kokkos::PerThread(0));

    Kokkos::parallel_for(
        "TeamBasedMultiply", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int teamNumber = team.league_rank();
            const int row = static_cast<int>((teamNumber / teamsPerRow) * (teamSize));
            const int column = static_cast<int>((teamNumber % teamsPerRow) * (teamSize));
            const int tilesCount = leftSize[1] / teamSize + (leftSize[1] % teamSize == 0 ? 0 : 1);

            const ScratchMemory leftMatrixView { team.team_scratch(memoryLevel), teamSize, teamSize };
            const ScratchMemory rightMatrixView { team.team_scratch(memoryLevel), teamSize, teamSize };

            for (int currentTile = 0; currentTile < tilesCount; ++currentTile) {

                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, std::min(teamSize, (int)leftSize[1] - currentTile * teamSize)),
                    KOKKOS_LAMBDA(int i) {
                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorRange(team, std::min(teamSize, (int)leftSize[0] - currentTile * teamSize)),
                            KOKKOS_LAMBDA(int j) {
                                leftMatrixView(i, j) = lhs_const_view(i + row, j + currentTile * teamSize);
                            });
                    });

                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, std::min(teamSize, rightSize[0] - currentTile * teamSize)),
                    KOKKOS_LAMBDA(int i) {
                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorRange(team, std::min(teamSize, rightSize[1] - currentTile * teamSize)),
                            KOKKOS_LAMBDA(int j) {
                                const int myRow = i + currentTile * teamSize;
                                const int myColumn = column + j;
                                rightMatrixView(i, j) = rhs_const_view(myRow, myColumn);
                            });
                    });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, teamSize * teamSize),
                    KOKKOS_LAMBDA(int i) {
                        const int threadRow = i / teamSize;
                        const int threadColumn = i % teamSize;

                        if ((threadRow + row < leftSize[0]) && (threadColumn + column < rightSize[1])) {
                            double product = 0.0;
                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team, std::min(teamSize, leftSize[1] - currentTile * teamSize)),
                                KOKKOS_LAMBDA(int j, double& value) {
                                    value += leftMatrixView(threadRow, j) * rightMatrixView(j, threadColumn);
                                },
                                Kokkos::Sum<double> { product });
                                output(threadRow + row, threadColumn + column) += product;
                        }
                    });
            }
        });
    Kokkos::fence();

    return output;
}

template <unsigned int... Traits>
Kokkos::View<int**> gpu::nussinov::teamBased(std::size_t teamSize, std::string_view rnaChain)
{
    ///* Start of CLooG code */
    //    if (N >= 2) {
    //        for (t2=1;t2<=N-1;t2++) {
    //            lbp=t2;
    //            ubp=N-1;
    //#pragma omp parallel for private(t4,t6) shared(t2)
    //            for (t4=t2;t4<=N-1;t4++) {
    //                for (t6=0;t6<=t2-1;t6++) {
    //                    S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t6+(-t2+t4)] + S[t6+(-t2+t4)+1][t4], S[(-t2+t4)][t4]);;
    //                }
    //                S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t4], S[(-t2+t4)+1][t4-1]) + MAX((-t2+t4),t4);;
    //            }
    //        }
    //    }
    using Policy = Kokkos::TeamPolicy<>;
    using Team = Policy::member_type;
    using MemoryTraits = Kokkos::MemoryTraits<(0u | ... | Traits)>;

    const int N = rnaChain.length();
    Kokkos::View<int**, MemoryTraits> output { "output", static_cast<std::size_t>(N), static_cast<std::size_t>(N) };

    const Kokkos::View<char*, MemoryTraits> deviceRnaChain { "deviceRna", rnaChain.length() };
    const Kokkos::View<Kokkos::pair<char, char>*> allowedPairings { "allowedPairings", 6 };
    allowedPairings(0) = { 'A', 'U' };
    allowedPairings(1) = { 'U', 'A' };
    allowedPairings(2) = { 'C', 'G' };
    allowedPairings(3) = { 'G', 'C' };
    allowedPairings(4) = { 'G', 'U' };
    allowedPairings(5) = { 'U', 'G' };

    std::memcpy(deviceRnaChain.data(), rnaChain.data(), rnaChain.length() * sizeof(char));

    const Kokkos::View<const char*, MemoryTraits> constDeviceRnaChain = deviceRnaChain;
    const Kokkos::View<const Kokkos::pair<char,char>*, MemoryTraits> constAllowedPairings = allowedPairings;

    for (int t2 = 1; t2 <= N - 1; ++t2) {
        const std::size_t groupsCount = (N - t2 + (teamSize - 1)) / teamSize;

        Kokkos::parallel_for(
            "nussinovFor",
            Policy { static_cast<int>(groupsCount), static_cast<int>(teamSize) },
            KOKKOS_LAMBDA(const Team& team) {
                const auto bond = [=](std::size_t i, std::size_t j) {
                    const Kokkos::pair<char, char> myPair { constDeviceRnaChain(i), constDeviceRnaChain(j) };
                    for (int k = 0; k < constAllowedPairings.extent(0); ++k) {
                        if (myPair == constAllowedPairings(k)) {
                            return 1;
                        }
                    }
                    return 0;
                };

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, teamSize),
                    KOKKOS_LAMBDA(int i) {
                        const int t4 = t2 + (team.league_rank() * teamSize) + i;
                        if (((-t2 + t4) < output.extent(0)) && (t4 < output.extent(0))) {
                            int maxValue = output(-t2 + t4, t4);
                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team, t2),
                                KOKKOS_LAMBDA(int j, int& value) {
                                    value = std::max(output((-t2 + t4), j + (-t2 + t4)) + output(j + (-t2 + t4) + 1, t4), value);
                                },
                                Kokkos::Max<int> { maxValue });
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
    //                    S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t6+(-t2+t4)] + S[t6+(-t2+t4)+1][t4], S[(-t2+t4)][t4]);;
    //                }
    //                S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t4], S[(-t2+t4)+1][t4-1]) + MAX((-t2+t4),t4);;
    //            }
    //        }
    //    }
    using Policy = Kokkos::RangePolicy<std::size_t>;
    using Team = Policy::member_type;

    const int N = rnaChain.length();
    Kokkos::View<int**> output { "output", static_cast<std::size_t>(N), static_cast<std::size_t>(N) };

    const Kokkos::View<char*> deviceRnaChain { "deviceRna", rnaChain.length() };
    const Kokkos::View<Kokkos::pair<char, char>*> allowedPairings { "allowedPairings", 6 };
    allowedPairings(0) = { 'A', 'U' };
    allowedPairings(1) = { 'U', 'A' };
    allowedPairings(2) = { 'C', 'G' };
    allowedPairings(3) = { 'G', 'C' };
    allowedPairings(4) = { 'G', 'U' };
    allowedPairings(5) = { 'U', 'G' };

    std::memcpy(deviceRnaChain.data(), rnaChain.data(), rnaChain.length() * sizeof(char));

    for (int t2 = 1; t2 <= N - 1; ++t2) {
        Kokkos::parallel_for(
            "nussinovFor",
            Policy { 0, static_cast<std::size_t>(N - t2) },
            KOKKOS_LAMBDA(std::size_t index) {
                const auto bond = [=](std::size_t i, std::size_t j) {
                    const Kokkos::pair<char, char> myPair { deviceRnaChain(i), deviceRnaChain(j) };
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
                    maxValue = std::max(output((-t2 + t4), j + (-t2 + t4)) + output(j + (-t2 + t4) + 1, t4), maxValue);
                }
                output((-t2 + t4), t4) = std::max(maxValue, output((-t2 + t4) + 1, t4 - 1) + bond((-t2 + t4), t4));
            });
        Kokkos::fence();
    }
    return output;
}
#include <Kokkos_Core.hpp>

#include <gui/mainwindow.h>
#include <parallel/gpu_code.h>

#include <QApplication>

#include <array>

namespace {
void print_gpu_configuration(std::ostream& os = std::cout)
{
    os << "Detected " << Kokkos::Cuda::detect_device_count() << " CUDA devices\n";
    Kokkos::Cuda::print_configuration(os);
}
}

int main(int argc, char** argv)
{
    const Kokkos::ScopeGuard scopeGuard { argc, argv };

    print_gpu_configuration();

    std::cout << "Default execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';
    std::cout << "Default memory space: " << Kokkos::DefaultExecutionSpace::memory_space::name() << '\n';
    QApplication application { argc, argv };

    MainWindow mainWindow {};
    mainWindow.setWindowState(Qt::WindowState::WindowMaximized);
    mainWindow.show();

    return application.exec();
}

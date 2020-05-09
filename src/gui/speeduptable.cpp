#include <gui/speeduptable.h>

#include <algorithm>

SpeedupTable::SpeedupTable()
    : SpeedupTable { nullptr }
{
}

SpeedupTable::SpeedupTable(QWidget* parent)
    : QTableWidget { nullptr }
{
}

void SpeedupTable::createTable(const std::vector<std::pair<std::string, double>>& a_executionTimes)
{
    setRowCount(static_cast<int>(a_executionTimes.size()) + 1);
    setColumnCount(3);

    setItem(0, 0, new QTableWidgetItem { "Version" });
    setItem(0, 1, new QTableWidgetItem { "Time" });
    setItem(0, 2, new QTableWidgetItem { "Speedup" });

    const std::string sequentialKey = "Sequential";

    const auto sequentialTimeIterator = std::find_if(a_executionTimes.cbegin(), a_executionTimes.cend(),
        [&](const std::pair<std::string, double>& pair) { return pair.first == sequentialKey; });

    int index = 1;
    for (const auto& [key, value] : a_executionTimes) {
        setItem(index, 0, new QTableWidgetItem { QString::fromStdString(key) });
        setItem(index, 1, new QTableWidgetItem { QString::number(value) });
        setItem(index, 2, new QTableWidgetItem { QString::number(sequentialTimeIterator->second / value) });
        ++index;
    }
}

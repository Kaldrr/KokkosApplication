#include <gui/speedupgraph.h>

#include <QtCharts/QBarSeries>
#include <QtCharts/QtCharts>

#include <array>

namespace {
const std::array Colors {
    QColor { "Red" },
    QColor { "Green" },
    QColor { "Blue" },
    QColor { "DarkRed" },
    QColor { "Orange" },
    QColor { "Magenta" },
    QColor { "DarkGreen" },
    QColor { "DarkOrange" }
};
}

SpeedupGraph::SpeedupGraph()
    : SpeedupGraph { nullptr }
{
}

SpeedupGraph::SpeedupGraph(QWidget* parent)
    : QtCharts::QChartView { parent }
    , m_chart { std::make_unique<QChart>() }
    , m_axis { new QValueAxis { m_chart.get() } }
{
    m_chart->setTitle("Speedup");
    m_chart->setAnimationOptions(QChart::SeriesAnimations);
    m_chart->legend()->setVisible(true);
    m_chart->legend()->setAlignment(Qt::AlignRight);
    setChart(m_chart.get());

    m_chart->addAxis(m_axis, Qt::AlignLeft);
}

void SpeedupGraph::createChart(const std::map<std::string, double>& a_executionTimes)
{
    if (a_executionTimes.empty()) {
        return;
    }
    m_chart->removeAllSeries();

    auto currentColor = Colors.cbegin();

    auto* const barSeries = new QBarSeries { this };
    for (const auto& [key, value] : a_executionTimes) {
        auto* const barSet = new QBarSet { QString::fromStdString(key), barSeries };
        barSet->setColor(*(currentColor++));
        barSet->append(value);
        barSeries->append(barSet);
    }

    m_chart->addSeries(barSeries);

    using T = std::pair<std::string, double>;

    const auto maxValueIterator = std::max_element(
        a_executionTimes.cbegin(),
        a_executionTimes.cend(),
        [](const T& lhs, const T& rhs) { return lhs.second < rhs.second; });

    assert(maxValueIterator != a_executionTimes.cend());
    m_axis->setRange(0, maxValueIterator->second);
}
#pragma once

#include <QtCharts/QAbstractAxis>
#include <QtCharts/QChartView>

#include <QDebug>

#include <map>
#include <memory>
#include <string>

class SpeedupGraph : public QtCharts::QChartView {
    Q_OBJECT

public:
    SpeedupGraph();
    explicit SpeedupGraph(QWidget* parent);

    void createChart(const std::map<std::string, double>& a_executionTimes);

private:
    std::unique_ptr<QtCharts::QChart> m_chart;
    QtCharts::QAbstractAxis* m_axis { nullptr };
};

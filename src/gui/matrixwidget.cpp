#include "ui_matrixoptions.h"
#include <gui/matrixwidget.h>
#include <parallel/gpu_code.h>

#include <QDebug>
#include <QIntValidator>
#include <QMessageBox>
#include <QString>

#include <limits>
#include <utility>

MatrixWidget::MatrixWidget(OptionsWidget* options)
    : MatrixWidget(options, nullptr)
{}

MatrixWidget::MatrixWidget(OptionsWidget* options, QWidget* parent)
    : SpeedupGraph { parent }
    , m_optionsParent{options}
{
}

QWidget* MatrixWidget::getOptionsPage()
{
    m_optionsWidget = std::make_unique<QWidget>();
    m_optionsUi = std::make_unique<Ui::MatrixOptions>();

    m_optionsUi->setupUi(m_optionsWidget.get());

    leftMatrixWidth = m_optionsUi->leftMatrixLeftSize;
    leftMatrixHeight = m_optionsUi->leftMatrixRightSize;
    rightMatrixWidth = m_optionsUi->rightMatrixLeftSize;
    rightMatrixHeight = m_optionsUi->rightMatrixRightSize;

    leftMatrixWidth->setValidator(new QIntValidator { 0, std::numeric_limits<int>::max(), leftMatrixWidth });
    leftMatrixHeight->setValidator(new QIntValidator { 0, std::numeric_limits<int>::max(), leftMatrixHeight });
    rightMatrixWidth->setValidator(new QIntValidator { 0, std::numeric_limits<int>::max(), rightMatrixWidth });
    rightMatrixHeight->setValidator(new QIntValidator { 0, std::numeric_limits<int>::max(), rightMatrixHeight });

    connect(m_optionsUi->startButton, &QPushButton::clicked,
        this, &MatrixWidget::startClicked);
    connect(m_optionsParent, &OptionsWidget::workFinished,
        this, &MatrixWidget::viewResults);

    return m_optionsWidget.get();
}
void MatrixWidget::startClicked()
{
    const std::pair<int, int> leftMatrixSize { leftMatrixWidth->text().toInt(), leftMatrixHeight->text().toInt() };
    const std::pair<int, int> rightMatrixSize { rightMatrixWidth->text().toInt(), rightMatrixHeight->text().toInt() };

    if (leftMatrixSize.second != rightMatrixSize.first) {
        QMessageBox errorMessage { QMessageBox::Icon::Critical,
            "Matrix size error",
            QString { "Matrix inner dimensions do not match [%1,%2]x[%3,%4]" }.arg(leftMatrixSize.first).arg(leftMatrixSize.second).arg(rightMatrixSize.first).arg(rightMatrixSize.second),
            QMessageBox::StandardButton::Ok };
        errorMessage.exec();
        return;
    }

    m_optionsParent->registerWork(&gpu::measureMatrixTimes, leftMatrixSize, rightMatrixSize);
}

void MatrixWidget::viewResults(const std::map<std::string, double>& times)
{
    for (const auto& [technique, time] : times) {
        std::cout << "Using " << technique << " took " << time << "s\n";
    }
    createChart(times);
}

MatrixWidget::~MatrixWidget() = default;

#include "gui/nussinovwidget.h"
#include "ui_nussinovoptions.h"

#include <parallel/gpu_code.h>

#include <QIntValidator>
#include <random>

NussinovWidget::NussinovWidget(OptionsWidget* optionsWidget)
    : NussinovWidget { optionsWidget, nullptr }
{
}

NussinovWidget::NussinovWidget(OptionsWidget* optionsWidget, QWidget* parent)
    : SpeedupGraph { parent }
    , m_optionsParent { optionsWidget }
{
}

NussinovWidget::~NussinovWidget() = default;

QWidget* NussinovWidget::getOptionsPage()
{
    m_optionsWidget = std::make_unique<QWidget>();
    m_options = std::make_unique<Ui::NussinovOptions>();
    m_options->setupUi(m_optionsWidget.get());

    connect(m_options->randomSequenceButton, &QRadioButton::toggled,
        [&](bool enabled) { m_options->rnaSequenceInput->setEnabled(!enabled); });
    connect(m_options->inputSequenceButton, &QRadioButton::toggled,
        [&](bool enabled) { m_options->randomSequenceLength->setEnabled(!enabled); });
    connect(m_options->startButton, &QPushButton::clicked,
        this, &NussinovWidget::startClicked);
    connect(m_optionsParent, &OptionsWidget::workFinished,
        [&](const std::map<std::string, double>& times){ createChart(times); });

    m_options->randomSequenceLength->setValidator(new QIntValidator { this });
    m_options->rnaSequenceInput->setEnabled(false);

    return m_optionsWidget.get();
}

void NussinovWidget::startClicked()
{
    const std::string rnaSequence = getRnaSequence();

    m_optionsParent->registerWork([](std::string rna) { return gpu::measureNussinovTimes(rna); }, rnaSequence);
}

std::string NussinovWidget::getRnaSequence()
{
    if (m_options->inputSequenceButton->isChecked()) {
        return m_options->rnaSequenceInput->text().toStdString();
    } else if (m_options->randomSequenceButton->isChecked()) {
        constexpr std::array nucleotide {
            'A',
            'G',
            'U',
            'C'
        };
        const int length = m_options->randomSequenceLength->text().toInt();

        std::mt19937 dev { std::random_device {}() };
        std::uniform_int_distribution<int> rng { 0, nucleotide.size() - 1 };

        std::string rnaChain {};
        rnaChain.reserve(length);
        for (std::size_t i = 0; i < length; ++i) {
            rnaChain += nucleotide.at(rng(dev));
        }
        return rnaChain;
    }
    return {};
}
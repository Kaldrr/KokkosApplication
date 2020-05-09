#include "ui_optionswidget.h"

#include <gui/matrixwidget.h>
#include <gui/nussinovwidget.h>
#include <gui/optionswidget.h>

OptionsWidget::OptionsWidget()
    : OptionsWidget { nullptr }
{
}

OptionsWidget::OptionsWidget(QWidget* parent)
    : QWidget { parent }
    , m_ui { new Ui::OptionsWidget }
    , m_optionsWidgetLayout { new QVBoxLayout }
{
    m_ui->setupUi(this);
    m_ui->optionsGroupBox->setHidden(true);
    m_ui->optionsGroupBox->setLayout(m_optionsWidgetLayout);
    connect(m_ui->matrixButton, &QPushButton::clicked,
        [this] {
            m_ui->tabWidget->setUpdatesEnabled(false);

            auto* const newWidget = new MatrixWidget { this };
            addNewWidget(newWidget, "Matrix");
            addOptionsPage(newWidget->getOptionsPage(), "Matrix options");

            m_ui->tabWidget->setUpdatesEnabled(true);
        });
    connect(m_ui->nussinovButton, &QPushButton::clicked,
        [this] {
            m_ui->tabWidget->setUpdatesEnabled(false);

            auto* const newWidget = new NussinovWidget { this };
            addNewWidget(newWidget, "Nussinov");
            addOptionsPage(newWidget->getOptionsPage(), "Nussinov options");

            m_ui->tabWidget->setUpdatesEnabled(true);
        });

    connect(&m_futureWatcher, &QFutureWatcher<std::map<std::string, double>>::finished,
        this, &OptionsWidget::futureFinished);
    connect(&m_futureWatcher, &QFutureWatcher<std::map<std::string, double>>::started,
        []() { QApplication::setOverrideCursor(Qt::WaitCursor); });
}

OptionsWidget::~OptionsWidget()
{
    delete m_ui;
}

void OptionsWidget::addNewWidget(QWidget* widget, const QString& title)
{
    const int tabPagesCount = m_ui->tabWidget->count();
    for (int i = tabPagesCount - 1; i >= 0; --i) {
        m_ui->tabWidget->widget(i)->deleteLater();
        m_ui->tabWidget->removeTab(i);
    }
    m_ui->tabWidget->addTab(widget, title);

    m_ui->tabWidget->setUpdatesEnabled(true);
}

void OptionsWidget::addOptionsPage(QWidget* optionsPage, const QString& title)
{
    m_ui->optionsGroupBox->setHidden(false);
    m_optionsWidgetLayout->addWidget(optionsPage);
}

void OptionsWidget::futureFinished()
{
    QApplication::restoreOverrideCursor();
    emit workFinished(m_futureWatcher.result());
}

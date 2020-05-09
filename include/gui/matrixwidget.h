#pragma once

#include <QLineEdit>
#include <QWidget>

#include <gui/speedupgraph.h>
#include <gui/optionswidget.h>

#include <memory>

namespace Ui {
class MatrixOptions;
}

class MatrixWidget : public SpeedupGraph {
    Q_OBJECT

public:
    explicit MatrixWidget(OptionsWidget* options);
    MatrixWidget(OptionsWidget* options, QWidget* parent);
    ~MatrixWidget() override;

    QWidget* getOptionsPage();


private slots:
    void startClicked();
    void viewResults(const std::map<std::string, double>& times);

private:
    std::unique_ptr<QWidget> m_optionsWidget {};
    std::unique_ptr<Ui::MatrixOptions> m_optionsUi {};
    QLineEdit* leftMatrixWidth { nullptr };
    QLineEdit* leftMatrixHeight { nullptr };
    QLineEdit* rightMatrixWidth { nullptr };
    QLineEdit* rightMatrixHeight { nullptr };
    OptionsWidget* m_optionsParent{nullptr};
};
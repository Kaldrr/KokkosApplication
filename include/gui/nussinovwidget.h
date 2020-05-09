#pragma once

#include <gui/optionswidget.h>
#include <gui/speedupgraph.h>

#include <memory>

namespace Ui {
class NussinovOptions;
}

class NussinovWidget : public SpeedupGraph {
public:
    explicit NussinovWidget(OptionsWidget* optionsWidget);
    NussinovWidget(OptionsWidget* optionsWidget, QWidget* parent);
    ~NussinovWidget() override;

    QWidget* getOptionsPage();

private:
    std::string getRnaSequence();

private slots:
    void startClicked();

private:
    std::unique_ptr<QWidget> m_optionsWidget {};
    std::unique_ptr<Ui::NussinovOptions> m_options;
    OptionsWidget* m_optionsParent { nullptr };
};

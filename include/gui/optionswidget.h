#pragma once

#include <QFuture>
#include <QFutureWatcher>
#include <QString>
#include <QToolBox>
#include <QWidget>
#include <QtConcurrent/QtConcurrentRun>
#include <QVBoxLayout>

#include <iostream>

namespace Ui {
class OptionsWidget;
}

class OptionsWidget : public QWidget {
    Q_OBJECT

public:
    OptionsWidget();
    explicit OptionsWidget(QWidget* parent);
    ~OptionsWidget() override;

    template <typename Callable, typename... Args>
    void registerWork(Callable callable, Args... args);

signals:
    void workFinished(std::map<std::string, double>);

private slots:
    void addNewWidget(QWidget* widget, const QString& title);
    void addOptionsPage(QWidget* optionsPage, const QString& title);
    void futureFinished();

private:
    Ui::OptionsWidget* m_ui { nullptr };
    QFutureWatcher<std::map<std::string, double>> m_futureWatcher {};
    QVBoxLayout *m_optionsWidgetLayout;
};

template <typename Callable, typename... Args>
void OptionsWidget::registerWork(Callable callable, Args... args)
{
    auto future = QtConcurrent::run([=]() {
        try {
            return callable(args...);
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
        } catch (...) {
            std::cerr << "Unknown exception...\n";
        }
        std::terminate();
    });
    m_futureWatcher.setFuture(future);
}

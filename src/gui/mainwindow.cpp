#include "ui_mainwindow.h"

#include <gui/mainwindow.h>
#include <gui/optionswidget.h>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setCentralWidget(new OptionsWidget);
}

MainWindow::~MainWindow()
{
    delete ui;
}
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1165, 774)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Hauptlayout erstellen
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setObjectName("main_layout")

        # Margins des Hauptlayouts minimieren
        self.main_layout.setContentsMargins(1, 1, 1, 1)
        self.main_layout.setSpacing(2)

        # Widget für die Colorbar
        self.colorbar_widget = QtWidgets.QWidget()
        self.colorbar_widget.setFixedWidth(90)
        self.main_layout.addWidget(self.colorbar_widget)

        # Vertikales Layout für Colorbar und Button
        colorbar_layout = QtWidgets.QVBoxLayout(self.colorbar_widget)
        colorbar_layout.setContentsMargins(1, 1, 1, 10)
        colorbar_layout.setSpacing(4)

        
        # Widget für die Colorbar
        self.colorbar_plot = QtWidgets.QWidget()
        colorbar_layout.addWidget(self.colorbar_plot, 0, QtCore.Qt.AlignTop)
        
        # Push-Button
        self.StartSkript = QtWidgets.QPushButton("Calculate")
        self.StartSkript.setFixedWidth(90)
        self.StartSkript.setFixedHeight(25)
        self.StartSkript.setShortcut("c")
        colorbar_layout.addWidget(self.StartSkript, 0, QtCore.Qt.AlignTop)
        colorbar_layout.addStretch(1)

        # Layout für die Plots
        self.plots_widget = QtWidgets.QWidget()
        self.plots_layout = QtWidgets.QVBoxLayout(self.plots_widget)
        self.plots_layout.setContentsMargins(1, 1, 1, 1)  # Minimale Ränder
        self.plots_layout.setSpacing(1)  # Minimaler Abstand
        self.main_layout.addWidget(self.plots_widget)

        # Vertikaler Splitter für die Plots
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_splitter.setHandleWidth(1)  # Dünnere Splitter
        self.plots_layout.addWidget(self.vertical_splitter)

        # Oberer und unterer horizontaler Splitter
        self.horizontal_splitter_top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.horizontal_splitter_top.setHandleWidth(1)
        
        self.horizontal_splitter_bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.horizontal_splitter_bottom.setHandleWidth(1)

        self.vertical_splitter.addWidget(self.horizontal_splitter_top)
        self.vertical_splitter.addWidget(self.horizontal_splitter_bottom)

        # Plot-Widgets erstellen
        self.SPLPlot = QtWidgets.QWidget()
        self.SPLPlot_YAxis = QtWidgets.QWidget()
        self.SPLPlot_XAxis = QtWidgets.QWidget()
        self.Plot_PolarPattern = QtWidgets.QWidget()

        # Layouts für die Plot-Widgets mit minimalen Rändern
        for widget in [self.SPLPlot, self.SPLPlot_YAxis, self.SPLPlot_XAxis, self.Plot_PolarPattern]:
            layout = QtWidgets.QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)  # Keine Ränder
            layout.setSpacing(0)  # Kein Abstand

        # Widgets zu Splittern hinzufügen
        self.horizontal_splitter_top.addWidget(self.SPLPlot)
        self.horizontal_splitter_top.addWidget(self.SPLPlot_YAxis)
        self.horizontal_splitter_bottom.addWidget(self.SPLPlot_XAxis)
        self.horizontal_splitter_bottom.addWidget(self.Plot_PolarPattern)

        # Minimale Größen anpassen (optional)
        min_size = QtCore.QSize(50, 50)  # Kleinere Mindestgröße
        for widget in [self.SPLPlot, self.SPLPlot_YAxis, self.SPLPlot_XAxis, self.Plot_PolarPattern]:
            widget.setMinimumSize(min_size)

        # Größenverhältnisse für die Splitter setzen
        self.vertical_splitter.setSizes([194, 130])
        self.horizontal_splitter_top.setSizes([770, 250])
        self.horizontal_splitter_bottom.setSizes([770, 250])  # Beide Werte gleich, damit das Polardiagramm links beginnt



        # Optional: Minimale Breiten setzen
        self.Plot_PolarPattern.setMinimumWidth(250)  # Minimale Breite für das Polardiagramm
        
        MainWindow.setCentralWidget(self.centralwidget)

        # Statusbar und Menubar Setup
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1165, 24))
        self.menubar.setObjectName("menubar")

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        self.menuWindow = QtWidgets.QMenu(self.menubar)
        self.menuWindow.setObjectName("menuWindow")
        self.menuCalculation = QtWidgets.QMenu(self.menubar)
        self.menuCalculation.setObjectName("menuCalculation")

        MainWindow.setMenuBar(self.menubar)

        # Aktionen erstellen und zum Menü hinzufügen
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtWidgets.QAction(MainWindow)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionReset_to_Factory_Settings = QtWidgets.QAction(MainWindow)
        self.actionReset_to_Factory_Settings.setCheckable(False)
        self.actionReset_to_Factory_Settings.setObjectName("actionReset_to_Factory_Settings")
        self.actionCourves_Window = QtWidgets.QAction(MainWindow)
        self.actionCourves_Window.setObjectName("actionCourves_Window")
        self.actionSpeakerSpecs_Window = QtWidgets.QAction(MainWindow)
        self.actionSpeakerSpecs_Window.setObjectName("actionSpeakerSpecs_Window")

        self.actionImpulse_Window = QtWidgets.QAction(MainWindow)
        self.actionImpulse_Window.setObjectName("actionImpulse_Window")
        self.actionSourceLayout_Window = QtWidgets.QAction(MainWindow)
        self.actionSourceLayout_Window.setObjectName("actionSourceLayout_Window")
        self.actionSurface_Window = QtWidgets.QAction(MainWindow)
        self.actionSurface_Window.setObjectName("actionSurface_Window")
        self.actionSettings_window = QtWidgets.QAction(MainWindow)
        self.actionSettings_window.setCheckable(False)
        self.actionSettings_window.setObjectName("actionSettings_window")
        self.actionPlotMode_SPL = QtWidgets.QAction(MainWindow)
        self.actionPlotMode_SPL.setObjectName("actionPlotMode_SPL")
        self.actionPlotMode_Phase = QtWidgets.QAction(MainWindow)
        self.actionPlotMode_Phase.setObjectName("actionPlotMode_Phase")

        # Aktionen zu den Menüs hinzufügen
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_as)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad)
        self.menuSettings.addAction(self.actionSettings_window)
        self.menuSettings.addSeparator()
        self.menuSettings.addAction(self.actionReset_to_Factory_Settings)
        self.menuWindow.addAction(self.actionSurface_Window)
        self.menuWindow.addSeparator()
        self.menuWindow.addAction(self.actionSpeakerSpecs_Window)
        self.menuWindow.addSeparator()
        self.menuWindow.addAction(self.actionCourves_Window)
        self.menuWindow.addAction(self.actionImpulse_Window)
        self.menuWindow.addSeparator()
        self.menuWindow.addAction(self.actionSourceLayout_Window)
        self.menuWindow.addAction(self.actionSurface_Window)
        self.menuCalculation.addAction(self.actionPlotMode_SPL)
        self.menuCalculation.addAction(self.actionPlotMode_Phase)

        # Füge neue Actions für die View-Kontrolle hinzu
        self.actionFocus_SPL = QtWidgets.QAction(MainWindow)
        self.actionFocus_SPL.setObjectName("actionFocus_SPL")
        self.actionFocus_Yaxis = QtWidgets.QAction(MainWindow)
        self.actionFocus_Yaxis.setObjectName("actionFocus_Yaxis")
        self.actionFocus_Xaxis = QtWidgets.QAction(MainWindow)
        self.actionFocus_Xaxis.setObjectName("actionFocus_Xaxis")
        self.actionFocus_Polar = QtWidgets.QAction(MainWindow)
        self.actionFocus_Polar.setObjectName("actionFocus_Polar")
        self.actionDefault_View = QtWidgets.QAction(MainWindow)
        self.actionDefault_View.setObjectName("actionDefault_View")
        
        # Füge die Actions zum Window-Menü hinzu
        self.menuWindow.addSeparator()
        self.menuWindow.addAction(self.actionFocus_SPL)
        self.menuWindow.addAction(self.actionFocus_Yaxis)
        self.menuWindow.addAction(self.actionFocus_Xaxis)
        self.menuWindow.addAction(self.actionFocus_Polar)
        self.menuWindow.addAction(self.actionDefault_View)

        # Neue Action für Manage Speaker
        self.actionManage_Speaker = QtWidgets.QAction(MainWindow)
        self.actionManage_Speaker.setObjectName("actionManage_Speaker")
        
        # Action zum Settings-Menü hinzufügen
        self.menuSettings.addSeparator()
        self.menuSettings.addAction(self.actionManage_Speaker)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menuWindow.menuAction())
        self.menubar.addAction(self.menuCalculation.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Low Frequency Optimizer"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuSettings.setTitle(_translate("MainWindow", "Setup"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))
        self.menuCalculation.setTitle(_translate("MainWindow", "Calculation"))
        self.actionLoad.setText(_translate("MainWindow", "Open"))
        self.actionLoad.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionNew.setShortcut(_translate("MainWindow", "Ctrl+Shift+N"))
        self.actionSave_as.setText(_translate("MainWindow", "Save as"))
        self.actionSave_as.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.actionReset_to_Factory_Settings.setText(_translate("MainWindow", "Load Factory Settings"))
        self.actionReset_to_Factory_Settings.setShortcut(_translate("MainWindow", "Ctrl+Shift+R"))
        self.actionSpeakerSpecs_Window.setText(_translate("MainWindow", "Sources"))
        self.actionSpeakerSpecs_Window.setShortcut(QtGui.QKeySequence("Ctrl+Alt+S"))
        #elf.actionSpeakerSpecs_Window.setShortcut(_translate("MainWindow", "Alt+1"))  
        self.actionCourves_Window.setText(_translate("MainWindow", "Snapshots"))
        self.actionCourves_Window.setShortcut(QtGui.QKeySequence("Ctrl+Alt+C"))
        self.actionImpulse_Window.setText(_translate("MainWindow", "Impulse"))
        self.actionImpulse_Window.setShortcut(QtGui.QKeySequence("Ctrl+Alt+I"))
        self.actionSourceLayout_Window.setText(_translate("MainWindow", "Source Layout"))
        self.actionSourceLayout_Window.setShortcut(QtGui.QKeySequence("Ctrl+Alt+L"))
        self.actionSurface_Window.setText(_translate("MainWindow", "Surface"))
        self.actionSurface_Window.setShortcut(QtGui.QKeySequence("Ctrl+Alt+P"))
        self.actionSettings_window.setText(_translate("MainWindow", "Preferences"))
        self.actionSettings_window.setShortcut(_translate("MainWindow", "Ctrl+-"))
        self.actionPlotMode_SPL.setText(_translate("MainWindow", "SPL Plot"))
        self.actionPlotMode_Phase.setText(_translate("MainWindow", "Phase alignment"))

        # Setze die Texte und Shortcuts für die neuen Actions
        self.actionFocus_SPL.setText(_translate("MainWindow", "Focus SPL"))
        self.actionFocus_SPL.setShortcut(_translate("MainWindow", "Ctrl+1"))
        self.actionFocus_Yaxis.setText(_translate("MainWindow", "Focus Yaxis"))
        self.actionFocus_Yaxis.setShortcut(_translate("MainWindow", "Ctrl+2"))
        self.actionFocus_Xaxis.setText(_translate("MainWindow", "Focus Xaxis"))
        self.actionFocus_Xaxis.setShortcut(_translate("MainWindow", "Ctrl+3"))
        self.actionFocus_Polar.setText(_translate("MainWindow", "Focus Polar"))
        self.actionFocus_Polar.setShortcut(_translate("MainWindow", "Ctrl+4"))
        self.actionDefault_View.setText(_translate("MainWindow", "Default View"))
        self.actionDefault_View.setShortcut(_translate("MainWindow", "Ctrl+5"))

        # Text und Shortcut für Manage Speaker
        self.actionManage_Speaker.setText(_translate("MainWindow", "Manage Speaker"))
        self.actionManage_Speaker.setShortcut(QtGui.QKeySequence("Ctrl+Alt+M"))

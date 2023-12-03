import sys, os
import re
import shutil
import zhconv
import json
import ntpath
from threading import Thread
from pathlib import Path
from scipy.io.wavfile import write

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from qfluentwidgets import PushButton, ComboBox, CaptionLabel, IndeterminateProgressBar, PlainTextEdit, DoubleSpinBox, ToolButton, InfoBar, InfoBarPosition
from qfluentwidgets.multimedia import SimpleMediaPlayBar, MediaPlayBarButton
from qfluentwidgets.common.icon import FluentIcon
from qfluentwidgets.components.dialog_box.mask_dialog_base import MaskDialogBase
from qfluentwidgets.components.widgets.combo_box import ComboBoxBase
from qfluentwidgets.common.style_sheet import FluentStyleSheet

import utils
import torch
from infer import infer, get_net_g
import numpy as np
from gradio.processing_utils import convert_to_16_bit_wav

regex = r'([\u0041-\u005a\u0061-\u007a ]+)'

if __name__ == "__main__":
    hps = utils.get_hparams_from_file("./configs/config.json")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    app_config = "./configs/app_config.json"
    with open(app_config, 'r') as f:
        data = json.load(f)
        models = data['models']
        current_settings = data['current_settings']
        download_path = data['download_path']
    
    versions = ["1.0", "1.1.0", "1.1.1-fix", "2.0"]
    languages = ["ZH", "JP", "EN"]
    speakers = list(hps.data.spk2id.keys())
    
    class NewComboBox(ComboBox):
        def __init__(self, parent):
            super().__init__(parent)
            
        def _onItemClicked(self, index):
            self.setCurrentIndex(index)
            self.currentTextChanged.emit(self.currentText())
            self.currentIndexChanged.emit(index)
            
        def removeItem(self, index: int):
            if not 0 <= index < len(self.items):
                return

            self.items.pop(index)

            if index < self.currentIndex():
                self.setCurrentIndex(self._currentIndex - 1)
            elif index == self.currentIndex():
                if index > 0:
                    self.setCurrentIndex(self._currentIndex - 1)
                else:
                    self.setCurrentIndex(0)

            if self.count() == 0:
                self.clear()
    
    class GenerateMessageBox(MaskDialogBase):

        def __init__(self, parent):
            super().__init__(parent)
            self.vBoxLayout = QVBoxLayout(self.widget)
            self.titleLabel = QLabel("Generating audio...", self.widget)
            self.contentLabel = QLabel("It takes longer time to generate the first audio", self.widget)
            self.progressBar = IndeterminateProgressBar(self.widget)
            self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignTop)
            self.vBoxLayout.addWidget(self.contentLabel, 0, Qt.AlignTop)
            self.vBoxLayout.addSpacing(8)
            self.vBoxLayout.addWidget(self.progressBar, 0 ,Qt.AlignTop)
            self.vBoxLayout.addSpacing(8)
            
            self.windowMask.setObjectName('windowMask')
            self.titleLabel.setObjectName('titleLabel')
            self.contentLabel.setObjectName('contentLabel')
            FluentStyleSheet.MESSAGE_DIALOG.apply(self)
            
            self.widget.setFixedSize(320,96)
        
    class Ui(QWidget):
        def __init__(self):
            super(Ui, self).__init__()
            self.resize(720, 540)
            self.setMinimumSize(QSize(600, 540))
            self.setMaximumSize(QSize(1440, 1080))
            
            font = QFont()
            font.setFamily(u"\u5fae\u8edf\u6b63\u9ed1\u9ad4")
            font.setPointSize(10)
            
            self.gridLayout_main = QGridLayout(self)  
            
            self.groupBox_input = QGroupBox(self)
            self.groupBox_input.setFont(font)
            self.groupBox_input.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
            self.groupBox_input.setTitle(u"Text Input / \u6587\u672c\u8f38\u5165")
            self.gridLayout_input= QGridLayout(self.groupBox_input)
            self.PlainTextEdit = PlainTextEdit(self.groupBox_input)

            self.gridLayout_input.addWidget(self.PlainTextEdit, 0, 0, 1, 1)
            self.gridLayout_main.addWidget(self.groupBox_input, 0, 0, 1, 1)

            self.groupBox_settings = QGroupBox(self)
            self.groupBox_settings.setFont(font)
            self.groupBox_settings.setTitle(u"Settings / \u8a2d\u5b9a")
            self.formLayout_settings = QFormLayout(self.groupBox_settings)
            
            self.CaptionLabel_7 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_7.setFont(font)
            self.CaptionLabel_7.setText(u"Model")
            self.formLayout_settings.setWidget(1, QFormLayout.LabelRole, self.CaptionLabel_7)
            
            self.widgetBox_1 = QWidget(self.groupBox_settings)
            self.layout_1 = QHBoxLayout(self.widgetBox_1)
            self.layout_1.setContentsMargins(0,0,0,0)
            self.layout_1.setSpacing(4)
            self.ComboBox_3 = ComboBox(self.widgetBox_1)
            self.ComboBox_3.setMinimumWidth(96)
            self.ComboBox_3.setMaximumWidth(128)
            for model in models:
                self.ComboBox_3.addItem(ntpath.basename(model))
            self.ComboBox_3.setCurrentIndex(int(current_settings[0]))
            self.layout_1.addWidget(self.ComboBox_3)
            
            self.btn_add_1 = ToolButton(self.widgetBox_1)
            self.btn_add_1.setMaximumWidth(32)
            self.btn_add_1.setIcon(FluentIcon.ADD)
            self.btn_add_1.setToolTip(self.tr('Add Models'))
            self.btn_add_1.clicked.connect(self.add_models)
            self.layout_1.addWidget(self.btn_add_1)
            
            self.btn_remove_1 = ToolButton(self.widgetBox_1)
            self.btn_remove_1.setMaximumWidth(32)
            self.btn_remove_1.setIcon(FluentIcon.REMOVE)
            self.btn_remove_1.setToolTip(self.tr('Remove Model'))
            self.btn_remove_1.clicked.connect(self.remove_model)
            self.layout_1.addWidget(self.btn_remove_1)
            
            self.formLayout_settings.setWidget(1, QFormLayout.FieldRole, self.widgetBox_1)
            
            self.CaptionLabel_8 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_8.setFont(font)
            self.CaptionLabel_8.setText(u"Version")
            self.formLayout_settings.setWidget(2, QFormLayout.LabelRole, self.CaptionLabel_8)
            
            self.ComboBox_4 = ComboBox(self.groupBox_settings)
            self.ComboBox_4.addItems(versions)
            self.ComboBox_4.setCurrentIndex(int(current_settings[1]))
            self.formLayout_settings.setWidget(2, QFormLayout.FieldRole, self.ComboBox_4)
            
            self.CaptionLabel_1 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_1.setFont(font)
            self.CaptionLabel_1.setText(u"Speaker")
            self.formLayout_settings.setWidget(3, QFormLayout.LabelRole, self.CaptionLabel_1)

            self.ComboBox_1 = ComboBox(self.groupBox_settings)
            self.ComboBox_1.addItems(speakers)
            self.ComboBox_1.setCurrentIndex(int(current_settings[2]))
            self.formLayout_settings.setWidget(3, QFormLayout.FieldRole, self.ComboBox_1)

            self.CaptionLabel_2 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_2.setFont(font)
            self.CaptionLabel_2.setText(u"Language")
            self.formLayout_settings.setWidget(4, QFormLayout.LabelRole, self.CaptionLabel_2)

            self.ComboBox_2 = ComboBox(self.groupBox_settings)
            self.ComboBox_2.addItems(languages)
            self.ComboBox_2.setCurrentIndex(int(current_settings[3]))
            self.formLayout_settings.setWidget(4, QFormLayout.FieldRole, self.ComboBox_2)

            self.CaptionLabel_3 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_3.setFont(font)
            self.CaptionLabel_3.setText(u"SDP/DP")
            self.formLayout_settings.setWidget(5, QFormLayout.LabelRole, self.CaptionLabel_3)

            self.DoubleSpinBox_1 = DoubleSpinBox(self.groupBox_settings)
            self.DoubleSpinBox_1.setRange(0, 1)
            self.DoubleSpinBox_1.setSingleStep(0.1)
            self.DoubleSpinBox_1.setValue(current_settings[4])
            self.DoubleSpinBox_1.setDecimals(1)
            self.formLayout_settings.setWidget(5, QFormLayout.FieldRole, self.DoubleSpinBox_1)

            self.CaptionLabel_4 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_4.setFont(font)
            self.CaptionLabel_4.setText(u"Emotion")
            self.formLayout_settings.setWidget(6, QFormLayout.LabelRole, self.CaptionLabel_4)

            self.DoubleSpinBox_2 = DoubleSpinBox(self.groupBox_settings)
            self.DoubleSpinBox_2.setRange(0.1, 2)
            self.DoubleSpinBox_2.setSingleStep(0.1)
            self.DoubleSpinBox_2.setValue(current_settings[5])
            self.DoubleSpinBox_2.setDecimals(1)
            self.formLayout_settings.setWidget(6, QFormLayout.FieldRole, self.DoubleSpinBox_2)

            self.CaptionLabel_5 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_5.setFont(font)
            self.CaptionLabel_5.setText(u"Length")
            self.formLayout_settings.setWidget(7, QFormLayout.LabelRole, self.CaptionLabel_5)

            self.DoubleSpinBox_3 = DoubleSpinBox(self.groupBox_settings)
            self.DoubleSpinBox_3.setRange(0.1, 2)
            self.DoubleSpinBox_3.setSingleStep(0.1)
            self.DoubleSpinBox_3.setValue(current_settings[6])
            self.DoubleSpinBox_3.setDecimals(1)
            self.formLayout_settings.setWidget(7, QFormLayout.FieldRole, self.DoubleSpinBox_3)

            self.CaptionLabel_6 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_6.setFont(font)
            self.CaptionLabel_6.setText(u"Speed")
            self.formLayout_settings.setWidget(8, QFormLayout.LabelRole, self.CaptionLabel_6)

            self.DoubleSpinBox_4 = DoubleSpinBox(self.groupBox_settings)
            self.DoubleSpinBox_4.setRange(0.1, 2)
            self.DoubleSpinBox_4.setSingleStep(0.1)
            self.DoubleSpinBox_4.setValue(current_settings[7])
            self.DoubleSpinBox_4.setDecimals(1)
            self.formLayout_settings.setWidget(8, QFormLayout.FieldRole, self.DoubleSpinBox_4)
            
            self.CaptionLabel_9 = CaptionLabel(self.groupBox_settings)
            self.CaptionLabel_9.setFont(font)
            self.CaptionLabel_9.setText(u"Presets")
            self.formLayout_settings.setWidget(9, QFormLayout.LabelRole, self.CaptionLabel_9)
            
            self.widgetBox_2 = QWidget(self.groupBox_settings)
            self.layout_2 = QHBoxLayout(self.widgetBox_2)
            self.layout_2.setContentsMargins(0,0,0,0)
            self.layout_2.setSpacing(4)
            self.ComboBox_5 = NewComboBox(self.widgetBox_2)
            self.ComboBox_5.setFixedWidth(96)
            self.ComboBox_5.setMaximumWidth(128)
            self.ComboBox_5.addItems(data['presets'].keys())
            self.ComboBox_5.currentIndexChanged.connect(self.change_current_presets)
            self.layout_2.addWidget(self.ComboBox_5)
            
            self.btn_add_2 = ToolButton(self.widgetBox_2)
            self.btn_add_2.setMaximumWidth(32)
            self.btn_add_2.setIcon(FluentIcon.ADD)
            self.btn_add_2.setToolTip(self.tr('Save Preset'))
            self.btn_add_2.clicked.connect(self.save_preset)
            self.layout_2.addWidget(self.btn_add_2)
            
            self.btn_remove_2 = ToolButton(self.widgetBox_2)
            self.btn_remove_2.setMaximumWidth(32)
            self.btn_remove_2.setIcon(FluentIcon.REMOVE)
            self.btn_remove_2.setToolTip(self.tr('Remove Preset'))
            self.btn_remove_2.clicked.connect(self.remove_preset)
            self.layout_2.addWidget(self.btn_remove_2)
            
            self.formLayout_settings.setWidget(9, QFormLayout.FieldRole, self.widgetBox_2)
            
            self.gridLayout_main.addWidget(self.groupBox_settings, 0, 1, 1, 1)

            self.groupBox_output = QGroupBox(self)
            self.groupBox_output.setFont(font)
            self.groupBox_output.setTitle(u"Audio Output / \u97f3\u983b\u8f38\u51fa")
            
            self.horizontalLayout_output = QHBoxLayout(self.groupBox_output)
            self.simplePlayBar = SimpleMediaPlayBar(self.groupBox_output)
            self.simplePlayBar.player.setNotifyInterval(10)
            self.simplePlayBar.setVolume(50)
            
            self.downloadButton = MediaPlayBarButton(self.simplePlayBar)
            self.downloadButton.setIcon(FluentIcon.DOWNLOAD)
            self.downloadButton.setToolTip(self.tr('Download'))
            self.downloadButton.clicked.connect(self.download)
            self.simplePlayBar.addButton(self.downloadButton)
            self.horizontalLayout_output.addWidget(self.simplePlayBar)
            self.url = None
            
            self.PushButton_1 = PushButton(self.groupBox_output)
            self.PushButton_1.setText(u"Generate")
            self.PushButton_1.clicked.connect(self.generate)
            self.horizontalLayout_output.addWidget(self.PushButton_1)
            
            self.gridLayout_main.addWidget(self.groupBox_output, 1, 0, 1, 2)
            
            #Text: self.PlainTextEdit.setPlainText()
            #Speaker: self.ComboBox_1.currentText()
            #Language: self.ComboBox_2.currentText()
            #Model: self.ComboBox_3.currentText()
            #Version: self.ComboBox_4.currentText()
            #Parameters: self.DoubleSpinBox_1.value()
        
        def generate_audio(self):
            net_g = get_net_g(
                model_path=models[self.ComboBox_3.currentIndex()], version=self.ComboBox_4.currentText(), device=device, hps=hps
            )
            text = self.PlainTextEdit.toPlainText()
            audio_list = []
            if self.ComboBox_2.currentText() == 'ZH':
                text = zhconv.convert(text, 'zh-cn')

            with torch.no_grad():             
                try:
                    audio = infer(
                            text,
                            self.DoubleSpinBox_1.value(),
                            self.DoubleSpinBox_2.value(),
                            self.DoubleSpinBox_3.value(),
                            self.DoubleSpinBox_4.value(),
                            self.ComboBox_1.currentText(), #Speaker
                            self.ComboBox_2.currentText(), #Language
                            hps,
                            net_g,
                            device,
                            self.ComboBox_4.currentText(), #Version
                        )
                    audio_list.append(convert_to_16_bit_wav(audio))
                    write("output.wav", 44100, np.concatenate(audio_list).astype(np.int16))
                    self.url = QUrl.fromLocalFile(str(Path('output.wav').absolute()))
                    self.simplePlayBar.player.setSource(self.url)
                    print("Success")
                    self.Box.accept() #Close the dialog
                        
                except RuntimeError:
                    self.Box.reject()
                    self.Message = "RuntimeError: Your text input is too long"
                    
                except IndexError:
                    self.Box.reject()
                    self.Message = "IndexError: Please check the value of n_speakers"
                    
        def generate(self):
            if self.ComboBox_3.currentIndex() == -1:
                InfoBar.error(
                    title='Error',
                    content="Please add a model",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=5000,
                    parent=self
                )         
                print("No Model")
                return
            
            if not Path(models[self.ComboBox_3.currentIndex()]).is_file():
                InfoBar.error(
                    title='Error',
                    content= self.ComboBox_3.currentText() + " does NOT exist",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=5000,
                    parent=self
                )
                print("File Not Exist")
                return
            
            print("Generating...")
            self.url = None
            self.simplePlayBar.player.setSource(self.url)
            t = Thread(target=self.generate_audio)
            t.start()
            self.Box = GenerateMessageBox(self)
            if self.Box.exec():
                InfoBar.success(
                    title='Success',
                    content="Audio is successfully generated",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
            else:
                InfoBar.error(
                    title='Failure',
                    content=self.Message,
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=10000,
                    parent=self
                )

        def download(self):
            if self.url:
                global download_path
                d_path, _ = QFileDialog.getSaveFileName(self, 'Save File', download_path, 'Audio File (*.wav)')
                if d_path:
                    download_path = d_path
                    shutil.copy2("./output.wav", download_path)
                    InfoBar.success(
                        title='Success',
                        content= ntpath.basename(download_path) + " is successfully downloaded",
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP,
                        duration=3000,
                        parent=self
                    )          
                    
        def add_models(self):
            model_paths, _ = QFileDialog.getOpenFileNames(self, 'Add Files', './', 'Files (*.pth)')
            if model_paths:
                for model in model_paths:
                    name = ntpath.basename(model)
                    if self.ComboBox_3.findText(name) == -1:
                        models.append(model)
                        self.ComboBox_3.addItem(name)
                        self.ComboBox_3.setCurrentIndex(self.ComboBox_3.count() - 1)
                        data['models'] = models
                        os.remove(app_config)
                        with open(app_config, 'w') as f:
                            json.dump(data, f, indent=4)
                    
        def remove_model(self):
            index = self.ComboBox_3.currentIndex()
            if index >= 0:
                del models[index]
                self.ComboBox_3.removeItem(index)
                data['models'] = models
                os.remove(app_config)
                with open(app_config, 'w') as f:
                    json.dump(data, f, indent=4)
        
        def change_current_presets(self):
            presets = data['presets'][self.ComboBox_5.currentText()]
            self.DoubleSpinBox_1.setValue(presets[0])  
            self.DoubleSpinBox_2.setValue(presets[1])  
            self.DoubleSpinBox_3.setValue(presets[2])  
            self.DoubleSpinBox_4.setValue(presets[3])
            
        def save_settings(self):
            data['current_settings'] = [self.ComboBox_3.currentIndex(),
                                        self.ComboBox_4.currentIndex(),
                                        self.ComboBox_1.currentIndex(),
                                        self.ComboBox_2.currentIndex(),
                                        self.DoubleSpinBox_1.value(),
                                        self.DoubleSpinBox_2.value(),
                                        self.DoubleSpinBox_3.value(),
                                        self.DoubleSpinBox_4.value()
                                        ]
            data['download_path'] = download_path
            os.remove(app_config)
            with open(app_config, 'w') as f:
                json.dump(data, f, indent=4)
        
        def save_preset(self):
            text, ok = QInputDialog().getText(self, 'Save Preset', 'Name:', text = self.ComboBox_5.text())
            if text and ok:
                if self.ComboBox_5.findText(text) == -1:
                    self.ComboBox_5.addItem(text)
                    self.ComboBox_5.setCurrentIndex(self.ComboBox_5.count() - 1)
                data['presets'][text] = [self.DoubleSpinBox_1.value(),
                                         self.DoubleSpinBox_2.value(),
                                         self.DoubleSpinBox_3.value(),
                                         self.DoubleSpinBox_4.value()]
                os.remove(app_config)
                with open(app_config, 'w') as f:
                    json.dump(data, f, indent=4)
        
        def remove_preset(self):
            index = self.ComboBox_5.currentIndex()
            if index >= 0 and self.ComboBox_5.currentText() != "Default":
                del data['presets'][self.ComboBox_5.currentText()]
                self.ComboBox_5.removeItem(index)
                os.remove(app_config)
                with open(app_config, 'w') as f:
                    json.dump(data, f, indent=4)        
            
    print("App Launched")
    
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication([])
    app.setAttribute(Qt.ApplicationAttribute.AA_DisableWindowContextHelpButton)
    ui = Ui()
    ui.show()
    app.aboutToQuit.connect(ui.save_settings)
    sys.exit(app.exec())

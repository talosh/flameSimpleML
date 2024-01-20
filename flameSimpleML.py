import os
import sys
import importlib
from pprint import pprint

try:
    from PySide6 import QtWidgets, QtCore, QtGui
    using_pyside6 = True
except ImportError:
    from PySide2 import QtWidgets, QtCore, QtGui
    using_pyside6 = False

import flameSimpleML_inference
importlib.reload(flameSimpleML_inference)

from flameSimpleML_inference import flameSimpleMLInference

settings = {
    'menu_group_name': 'Simple ML',
    'debug': False,
    'app_name': 'flameSimpleML',
    'prefs_folder': os.getenv('FLAMESMPLMsL_PREFS'),
    'bundle_folder': os.getenv('FLAMESMPLML_BUNDLE'),
    'packages_folder': os.getenv('FLAMESMPLML_PACKAGES'),
    'temp_folder': os.getenv('FLAMESMPLML_TEMP'),
    'requirements': [
        'numpy>=1.16',
        'torch>=1.12.0'
    ],
    'version': 'v0.0.2',
}

class ApplyModelDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Line Edit for path
        self.pathLineEdit = QtWidgets.QLineEdit(self)

        # Choose button
        self.chooseButton = QtWidgets.QPushButton("Choose", self)
        self.chooseButton.clicked.connect(self.chooseFile)

        # Create and Cancel buttons
        self.createButton = QtWidgets.QPushButton("Create", self)
        self.cancelButton = QtWidgets.QPushButton("Cancel", self)

        # Setting up layouts
        self.layout = QtWidgets.QVBoxLayout(self)
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.pathLayout = QtWidgets.QHBoxLayout()

        # Adding widgets to the path layout
        self.pathLayout.addWidget(self.pathLineEdit)
        self.pathLayout.addWidget(self.chooseButton)

        # Adding widgets to the button layout
        self.buttonLayout.addWidget(self.createButton)
        self.buttonLayout.addWidget(self.cancelButton)

        # Adding layouts to the main layout
        self.layout.addLayout(self.pathLayout)
        self.layout.addLayout(self.buttonLayout)

        # Connect the create and cancel buttons
        self.createButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

    def chooseFile(self):
        # Open file dialog and update path line edit
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose File")
        if file_name:
            self.pathLineEdit.setText(file_name)


def get_media_panel_custom_ui_actions():
    def scope_clip(selection):
        try:
            import flame
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    return True
        except Exception as e:
            print (f'[{settings["app_name"]}]: Exception: {e}')
        return False
    
    def apply_model(selection):
        apply_dialog = ApplyModelDialog()
        if apply_dialog.exec():
            print ('hello')
        else:
            print ('cancel')

        return
        '''
        return flameSimpleMLInference(
            selection=selection,
            settings=settings
            )
        '''

    def train_model(selection):
        import flame
        flame_version = flame.get_version()
        python_executable_path = f'/opt/Autodesk/python/{flame_version}/bin/python'
        script_folder = os.path.abspath(os.path.dirname(__file__))
        app_name = settings.get('app_name')
        version = settings.get('version')
        msg = f'GUI for model training is not yet implemented in {app_name} {version}\n'
        msg += f'Training is currently possible with a command-line script.\n'
        msg += f'Please run\n"{python_executable_path} {script_folder}/train.py --help"'
        dialog = flame.messages.show_in_dialog(
            title ='Train Model GUI is not yet implemented',
            message = msg,
            type = 'info',
            buttons = ['Copy', 'Ok'])
        if dialog == 'Copy':
            try:
                from PySide6.QtWidgets import QApplication
            except ImportError:
                from PySide2.QtWidgets import QApplication

            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            clipboard = app.clipboard()
            clipboard.setText(f'{python_executable_path} {script_folder}/train.py --help')

    def about_dialog():
        pass

    menu = [
        {
            'name': settings['menu_group_name'],
            'actions': [
                {
                    'name': 'Apply SimpleML model',
                    'execute': apply_model,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': 'Train SimpleML model',
                    'execute': train_model,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': f'Version: {settings["version"]}',
                    'execute': about_dialog,
                    'isVisible': scope_clip,
                    'isEnabled': False,
                },
            ],
        }
    ]

    return menu

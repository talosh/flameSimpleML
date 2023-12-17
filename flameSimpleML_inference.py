import os
import sys

from flameSimpleML_framework import flameAppFramework

try:
    from PySide6 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide2 import QtWidgets, QtCore, QtGui

from pprint import pprint, pformat

class flameSimpleMLInference(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        
        self.settings = kwargs.get('settings', dict())
        self.framework = flameAppFramework(settings = self.settings)
        self.version = self.settings.get('version', 'UnknownVersion')
        self.prefs['version'] = self.version
        self.framework.save_prefs()
        
        '''

        # Module defaults
        self.new_speed = 1
        self.dedup_mode = 0
        self.cpu = False
        self.flow_scale = self.prefs.get('flow_scale', 1.0)

        self.flow_res = {
            1.0: 'Use Full Resolution',
            0.5: 'Use 1/2 Resolution',
            0.25: 'Use 1/4 Resolution',
            0.125: 'Use 1/8 Resolution',
        }

        self.modes = {
            1: 'Normal',
            2: 'Faster',
            3: 'Slower',
            4: 'CPU - Normal',
            5: 'CPU - Faster',
            6: 'CPU - Slower'
        }

        self.current_mode = self.prefs.get('current_mode', 1)

        self.trained_models_path = os.path.join(
            self.framework.bundle_folder,
            'trained_models', 
            'default',
        )

        self.model_path = os.path.join(
            self.trained_models_path,
            'v4.6.model'
            )

        self.flownet_model_path = os.path.join(
            self.trained_models_path,
            'v2.4.model',
            'flownet.pkl'
        )

        self.current_models = {}

        self.progress = None
        self.torch = None
        self.threads = True
        self.temp_library = None
        
        self.torch_device = None

        self.requirements = requirements

        # this enables fallback to CPU on Macs
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        '''

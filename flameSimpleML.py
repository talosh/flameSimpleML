import os
import sys
import importlib
from pprint import pprint

import flameSimpleML_inference
importlib.reload(flameSimpleML_inference)

from flameSimpleML_inference import flameSimpleMLInference

settings = {
    'menu_group_name': 'Simple ML',
    'debug': True,
    'app_name': 'flameSimpleML',
    'prefs_folder': os.getenv('FLAMESMPLMsL_PREFS'),
    'bundle_folder': os.getenv('FLAMESMPLML_BUNDLE'),
    'packages_folder': os.getenv('FLAMESMPLML_PACKAGES'),
    'temp_folder': os.getenv('FLAMESMPLML_TEMP'),
    'requirements': [
        'numpy>=1.16',
        'torch>=1.12.0'
    ],
    'version': 'v0.0.1.dev.011',
}

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
        return flameSimpleMLInference(
            selection=selection,
            settings=settings
            ) 

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
            buttons = ['Copy command to Clipboard', 'Ok'])
        # if dialog == 'Copy command to Clipboard':

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

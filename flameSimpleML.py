import os
import sys

from utils.flameSimpleML_common import flameAppFramework

# Configurable settings
menu_group_name = 'Simple ML'
DEBUG = False
app_name = 'flameSimpleML'
prefs_folder = os.getenv('FLAMESMPLML_PREFS')
bundle_folder = os.getenv('FLAMESMPLML_BUNDLE')
packages_folder = os.getenv('FLAMESMPLML_PACKAGES')
temp_folder = os.getenv('FLAMESMPLML_TEMP')
requirements = [
    'numpy>=1.16',
    'torch>=1.3.0'
]
__version__ = 'v0.0.1.dev.001'


def get_media_panel_custom_ui_actions():

    def about_dialog():
        pass

    menu = [
        {
            'name': app_name,
            'actions': [
                {
                    'name': f'{app_name}: {__version__}',
                    'execute': about_dialog,
                    #
                    # Since this action require user input, we want to
                    # show the normal cursor for the duration of the action.
                    #
                    'waitCursor': False,
                },
            ],
        }
    ]

    return menu

import os
import sys

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

    menu = []
    selection = []

    try:
        import flame
        selection = flame.media_panel.selected_entries
    except:
        pass
    
    for app in apps:
        if app.__class__.__name__ == 'flameTimewarpML':
            app_menu = []
            app_menu = app.build_menu()
            if app_menu:
                menu.append(app_menu)
    return menu

import os
import sys
import time
import queue
import threading
import traceback
import atexit
import hashlib
import pickle

from pprint import pprint, pformat

class flameAppFramework(object):
    # flameAppFramework class takes care of preferences

    class prefs_dict(dict):
        # subclass of a dict() in order to directly link it 
        # to main framework prefs dictionaries
        # when accessed directly it will operate on a dictionary under a 'name'
        # key in master dictionary.
        # master = {}
        # p = prefs(master, 'app_name')
        # p['key'] = 'value'
        # master - {'app_name': {'key', 'value'}}
            
        def __init__(self, master, name, **kwargs):
            self.name = name
            self.master = master
            if not self.master.get(self.name):
                self.master[self.name] = {}
            self.master[self.name].__init__()

        def __getitem__(self, k):
            return self.master[self.name].__getitem__(k)
        
        def __setitem__(self, k, v):
            return self.master[self.name].__setitem__(k, v)

        def __delitem__(self, k):
            return self.master[self.name].__delitem__(k)
        
        def get(self, k, default=None):
            return self.master[self.name].get(k, default)
        
        def setdefault(self, k, default=None):
            return self.master[self.name].setdefault(k, default)

        def pop(self, k, v=object()):
            if v is object():
                return self.master[self.name].pop(k)
            return self.master[self.name].pop(k, v)
        
        def update(self, mapping=(), **kwargs):
            self.master[self.name].update(mapping, **kwargs)
        
        def __contains__(self, k):
            return self.master[self.name].__contains__(k)

        def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
            return type(self)(self)
        
        def keys(self):
            return list(self.master[self.name].keys())

        @classmethod
        def fromkeys(cls, keys, v=None):
            return self.master[self.name].fromkeys(keys, v)
        
        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, self.master[self.name].__repr__())

        def master_keys(self):
            return list(self.master.keys())

    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__
        self.settings = kwargs.get('settings', dict())
        self.app_name = self.settings.get('app_name', 'flameApp')
        self.bundle_name = self.sanitize_name(self.app_name)
        self.version = self.settings.get('version', 'Unknown version')
        self.debug = self.settings.get('debug', False)
        self.requirements = self.settings.get('requirements', list())

        self.log_debug(f'settings: {self.settings}')

        # self.prefs scope is limited to flame project and user
        self.prefs = {}
        self.prefs_user = {}
        self.prefs_global = {}
        
        try:
            import flame
            self.flame = flame
            self.flame_project_name = self.flame.project.current_project.name
            self.flame_user_name = flame.users.current_user.name
        except:
            self.flame = None
            self.flame_project_name = 'UnknownFlameProject'
            self.flame_user_name = 'UnknownFlameUser'
        
        try:
            import socket
            self.hostname = socket.gethostname()
        except:
            self.hostname = 'UnknownHostName'

        if self.settings.get('prefs_folder'):
            self.prefs_folder = self.settings['prefs_folder']        
        elif sys.platform == 'darwin':
            self.prefs_folder = os.path.join(
                os.path.expanduser('~'),
                 'Library',
                 'Preferences',
                 self.bundle_name)
        elif sys.platform.startswith('linux'):
            self.prefs_folder = os.path.join(
                os.path.expanduser('~'),
                '.config',
                self.bundle_name)

        self.prefs_folder = os.path.join(
            self.prefs_folder,
            self.hostname,
        )

        self.log_debug('[%s] waking up' % self.__class__.__name__)
        self.load_prefs()
        
        if self.settings.get('bundle_folder'):
            self.bundle_folder = self.settings['bundle_folder']
        else:
            self.bundle_folder = os.path.realpath(
                os.path.dirname(__file__)
            )

        if self.settings.get('packages_folder'):
            self.packages_folder = self.settings['packages_folder']
        else:
            self.packages_folder = os.path.join(
                self.bundle_folder,
                'packages'
            )

        self.site_packages_folder = os.path.join(
            self.packages_folder,
            '.lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages'
        )

        self.log_debug(f'site-packages folder: {self.site_packages_folder}')

        if self.settings.get('temp_folder'):
            self.temp_folder = self.settings['temp_folder']
        else:
            self.temp_folder = os.path.join(
            '/var/tmp',
            self.bundle_name,
            'temp'
        )
        
        self.log_debug(f'temp folder: {self.temp_folder}')

        '''
        self.bundle_path = os.path.join(
            self.bundle_folder,
            self.bundle_name
        )

        if not self.check_bundle_id():
            threading.Thread(
                target=self.unpack_bundle,
                args=(os.path.dirname(self.site_packages_folder), )
            ).start()
        '''

    def log(self, message):
        try:
            print ('[%s] %s' % (self.bundle_name, str(message)))
        except:
            pass

    def log_debug(self, message):
        if self.debug:
            try:
                print ('[DEBUG %s] %s' % (self.bundle_name, str(message)))
            except:
                pass

    def load_prefs(self):
        import json
        
        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs.json'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs.json'
        prefs_global_file_path = prefix + '.prefs.json'

        try:
            with open(prefs_file_path, 'r') as prefs_file:
                self.prefs = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs, indent=4))
        except Exception as e:
            self.log_debug('unable to load preferences from %s' % prefs_file_path)
            self.log_debug(e)

        try:
            with open(prefs_user_file_path, 'r') as prefs_file:
                self.prefs_user = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_user_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_user, indent=4))
        except Exception as e:
            self.log_debug('unable to load preferences from %s' % prefs_user_file_path)
            self.log_debug(e)

        try:
            with open(prefs_global_file_path, 'r') as prefs_file:
                self.prefs_global = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_global_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_global, indent=4))
        except Exception as e:
            self.log_debug('unable to load preferences from %s' % prefs_global_file_path)
            self.log_debug(e)

        return True

    def save_prefs(self):
        import json

        if not os.path.isdir(self.prefs_folder):
            try:
                os.makedirs(self.prefs_folder)
            except:
                self.log('unable to create folder %s' % self.prefs_folder)
                return False

        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs.json'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs.json'
        prefs_global_file_path = prefix + '.prefs.json'

        try:
            with open(prefs_file_path, 'w') as prefs_file:
                json.dump(self.prefs, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_file_path)
            self.log_debug(e)

        try:
            with open(prefs_user_file_path, 'w') as prefs_file:
                json.dump(self.prefs_user, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_user_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_user, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_user_file_path)
            self.log_debug(e)

        try:
            with open(prefs_global_file_path, 'w') as prefs_file:
                json.dump(self.prefs_global, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_global_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_global, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_global_file_path)
            self.log_debug(e)
            
        return True

    def check_bundle_id(self):
        bundle_id_file_path = os.path.join(
            os.path.dirname(self.site_packages_folder),
            'bundle_id'
            )

        bundle_id = self.version

        if (os.path.isdir(self.bundle_folder) and os.path.isfile(bundle_id_file_path)):
            self.log('checking existing bundle id %s' % bundle_id_file_path)
            try:
                with open(bundle_id_file_path, 'r') as bundle_id_file:
                    if bundle_id_file.read() == bundle_id:
                        self.log('bundle folder exists with id matching current version')
                        bundle_id_file.close()
                        return True
                    else:
                        self.log('existing env bundle id does not match current one')
                        return False
            except Exception as e:
                self.log(pformat(e))
                return False
        elif not os.path.isdir(self.bundle_folder):
            self.log('bundle folder does not exist: %s' % self.bundle_folder)
            return False
        elif not os.path.isfile(bundle_id_file_path):
            self.log('bundle id file does not exist: %s' % bundle_id_file_path)
            return False

    def unpack_bundle(self, bundle_path):
        start = time.time()
        script_file_name, ext = os.path.splitext(os.path.realpath(__file__))
        script_file_name += '.py'
        # self.log('script file: %s' % script_file_name)
        script = None
        payload = None

        try:
            with open(script_file_name, 'r+') as scriptfile:
                script = scriptfile.read()
                start_position = script.rfind('# bundle payload starts here')
                
                if script[start_position -1: start_position] != '\n':
                    scriptfile.close()
                    return False

                start_position += 33
                payload = script[start_position:-4]
                # scriptfile.truncate(start_position - 34)
                scriptfile.close()
        except Exception as e:
            self.log_exception(e)
            return False
        
        del script
        if not payload:
            return False
        
        if payload.startswith('BUNDLE_PAYLO'):
            self.log(f'No payload attached to {__file__}')
            self.log('Nothing to unpack')
            return False

        bundle_backup_folder = ''
        if os.path.isdir(bundle_path):
            bundle_backup_folder = os.path.realpath(bundle_path + '.previous')
            if os.path.isdir(bundle_backup_folder):
                try:
                    cmd = 'rm -rf "' + os.path.realpath(bundle_backup_folder) + '"'
                    self.log('removing previous backup folder')
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)
                except Exception as e:
                    self.log_exception(e)
                    return False
            try:
                cmd = 'mv "' + os.path.realpath(bundle_path) + '" "' + bundle_backup_folder + '"'
                self.log('backing up existing bundle folder')
                self.log('Executing command: %s' % cmd)
                os.system(cmd)
            except Exception as e:
                self.log_exception(e)
                return False

        try:
            self.log('creating new bundle folder: %s' % bundle_path)
            os.makedirs(bundle_path)
        except Exception as e:
            self.log_exception(e)
            return False

        payload_dest = os.path.join(
            bundle_path, 
            self.sanitize_name(self.bundle_name + '.' + __version__ + '.bundle.tar.gz')
            )
        
        try:
            import base64
            self.log('unpacking payload: %s' % payload_dest)
            with open(payload_dest, 'wb') as payload_file:
                payload_file.write(base64.b64decode(payload))
                payload_file.close()
            cmd = 'tar xf "' + payload_dest + '" -C "' + bundle_path + '/"'
            self.log('Executing command: %s' % cmd)
            status = os.system(cmd)
            self.log('exit status %s' % os.WEXITSTATUS(status))

            # self.log('cleaning up %s' % payload_dest, logfile)
            # os.remove(payload_dest)
        
        except Exception as e:
            self.log_exception(e)
            return False

        delta = time.time() - start
        self.log('bundle extracted to %s' % bundle_path)
        self.log('extracting bundle took %s sec' % '{:.1f}'.format(delta))

        del payload
        try:
            os.remove(payload_dest)
        except Exception as e:
            self.log_exception(e)

        try:
            with open(os.path.join(bundle_path, 'bundle_id'), 'w') as bundle_id_file:
                bundle_id_file.write(self.version)
        except Exception as e:
            self.log_exception(e)
            return False
        
        return True

    def log_exception(self, e):
        self.log(pformat(e))
        self.log_debug(pformat(traceback.format_exc()))

    def sanitize_name(self, name_to_sanitize):
        import re
        if name_to_sanitize is None:
            return None
        
        stripped_name = name_to_sanitize.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        result = exp.sub('_', stripped_name)
        return re.sub('_\_+', '_', result)

    def sanitized(self, text):
        import re

        if text is None:
            return None
        
        text = text.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        if isinstance(text, str):
            result = exp.sub('_', text)
        else:
            decoded = text.decode('utf-8')
            result = exp.sub('_', decoded).encode('utf-8')

        return re.sub('_\_+', '_', result)

    def create_timestamp_uid(self):
        import random
        import uuid
        from datetime import datetime

        def number_to_letter(number):
            # Map each digit to a letter
            mapping = {
                '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
                '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'
            }
            return ''.join(mapping.get(char, char) for char in number)

        uid = ((str(uuid.uuid4()).replace('-', '')).upper())
        uid = ''.join(random.sample(number_to_letter(uid), 4))
        timestamp = (datetime.now()).strftime('%Y%b%d_%H%M').upper()
        return f'{timestamp}_{uid}'

    def normalize_values(self, image_array, torch = None):
        if torch is None:
            import torch

        def custom_bend(x):
            linear_part = x
            exp_positive = torch.pow( x, 1 / 4 )
            exp_negative = -torch.pow( -x, 1 / 4 )
            return torch.where(x > 1, exp_positive, torch.where(x < -1, exp_negative, linear_part))

        # transfer (0.0 - 1.0) onto (-1.0 - 1.0) for tanh
        image_array = (image_array * 2) - 1
        # bend values below -1.0 and above 1.0 exponentially so they are not larger then (-4.0 - 4.0)
        image_array = custom_bend(image_array)
        # bend everything to fit -1.0 - 1.0 with hyperbolic tanhent
        image_array = torch.tanh(image_array)
        # move it to 0.0 - 1.0 range
        image_array = (image_array + 1) / 2

        return image_array
    
    def restore_normalized_values(self, image_array, torch = None):
        if torch is None:
            import torch

        def custom_de_bend(x):
            linear_part = x
            inv_positive = torch.pow( x, 4 )
            inv_negative = -torch.pow( -x, 4 )
            return torch.where(x > 1, inv_positive, torch.where(x < -1, inv_negative, linear_part))

        epsilon = torch.tensor(4e-8, dtype=torch.float32).to(image_array.device)
        # clamp image befor arctanh
        image_array = torch.clamp((image_array * 2) - 1, -1.0 + epsilon, 1.0 - epsilon)
        # restore values from tanh  s-curve
        image_array = torch.arctanh(image_array)
        # restore custom bended values
        image_array = custom_de_bend(image_array)
        # move it to 0.0 - 1.0 range
        image_array = ( image_array + 1.0) / 2.0

        return image_array

    def check_requirements(self, requirements):
        sys.path_importer_cache.clear()

        def import_required_packages(requirements, cleanup = False):
            import re

            packages_by_name = {re.split(r'[!<>=]', req)[0]: req for req in requirements}
            missing_requirements = []

            for package_name in packages_by_name.keys():
                # try:
                #    self.message_queue.put(
                #        {'type': 'info', 'message': f'Checking requirements... importing {package_name}'}
                #    )
                # except:
                #    pass
                try:
                    sys.path_importer_cache.clear()                   
                    __import__(package_name)
                    
                    if cleanup:
                        if package_name in sys.modules:
                            del sys.modules[package_name]
                            sys.path_importer_cache.clear()                   

                    # try:
                    #    self.message_queue.put(
                    #        {'type': 'info', 'message': f'Checking requirements... successfully imported {package_name}'}
                    #    )
                    # except:
                    #    pass
                except:
                    missing_requirements.append(packages_by_name.get(package_name))
            return missing_requirements
        
        missing_requirements = import_required_packages(requirements)

        if missing_requirements:
            # try to add bundled packafe folder into sys.path and check if it is possible to import
            if not self.site_packages_folder in sys.path:
                sys.path.append(self.site_packages_folder)
            missing_requirements = import_required_packages(requirements, cleanup = False)
            
            # missing_requirements = import_required_packages(requirements, cleanup = True)
            # cleanup sys path and import cache afterwards
            # if self.site_packages_folder in sys.path:
            #    sys.path.remove(self.site_packages_folder)
            # sys.path_importer_cache.clear()

            return missing_requirements
        else:
            return []

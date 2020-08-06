import os.path
import sys
from shutil import copyfile

class ScriptsManager:

    def __init__(self):
        self.pkg_path = sys.modules['efsassembler'].__path__[0] + "/"


    def add_fs_algorithm(self, script_path):
        user_scripts_path = self.pkg_path + "fs_algorithms/user_algorithms/"
        dest = user_scripts_path + script_path.split('/')[-1]
        copyfile(script_path, dest)
        # TO-DO: Add some verification operations (repeated file etc) and messages
        return


    def remove_fs_algorithm(self, script_name):
        user_scripts_path = self.pkg_path + "fs_algorithms/user_algorithms/"
        error404 = "Couldn't find provided file: " + script_name

        if '.' in script_name: #if user provided the extension
            file_path = user_scripts_path + script_name
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                raise(error404)
        else: 

            if os.path.isfile(user_scripts_path + script_name + ".py"):
                os.remove(user_scripts_path + script_name + ".py")
            elif os.path.isfile(user_scripts_path + script_name + ".r"):
                os.remove(user_scripts_path + script_name + ".r")
            else:
                raise(error404)
        return
        

    def add_aggregation_algorithm(self, script_path):
        user_scripts_path = self.pkg_path + "aggreg_algorithms/user_algorithms/"
        dest = user_scripts_path + script_path.split('/')[-1]
        copyfile(script_path, dest)
        # TO-DO: Add some verification operations (repeated file etc) and messages
        return


    def remove_aggregation_algorithm(self, script_name):
        user_scripts_path = self.pkg_path + "aggreg_algorithms/user_algorithms/"
        error404 = "Couldn't find provided file: " + script_name

        if '.' in script_name: #if user provided the extension
            file_path = user_scripts_path + script_name
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                raise(error404)
        else: 
            if os.path.isfile(user_scripts_path + script_name + ".py"):
                os.remove(user_scripts_path + script_name + ".py")
            else:
                raise(error404)
        return
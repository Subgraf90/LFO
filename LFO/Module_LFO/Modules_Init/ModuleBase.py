from Module_LFO.Modules_Calculate.Functions import FunctionToolbox

class ModuleBase:

    def __init__(self, settings):
 
        # Funktionen aus Toolboxwerden initialisiert
        self.functions = FunctionToolbox(settings)
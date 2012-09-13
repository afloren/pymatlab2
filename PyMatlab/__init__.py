import engine

class Session:
    def __init__(self, startcmd=None):
        self._ep = engine.open(startcmd);

    def __del__(self):
        engine.close(self._ep);

    def getVariable(self, name):
        return engine.getVariable(self._ep,name);

    def putVariable(self, name, value):
        engine.putVariable(self._ep,name,value);

    def evalString(self,string):
        return engine.evalString(self._ep,string);

    def getVisible(self):
        return engine.getVisible(self._ep);

    def setVisible(self,value):
        engine.setVisible(self._ep,value);

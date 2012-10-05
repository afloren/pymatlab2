import engine

class MatlabFunc:
    def __init__(self,session,name):
        self._session = session
        self._name = name

    def __call__(self,*args,**kwds):
        return self._session.call(self._name,*args,**kwds)

class Session:
    _evalstr = '''
try
  pymatlab2_err=0;
  {0};
catch e
  disp(e);
  pymatlab2_exception=e;
  pymatlab2_err=1;
end
'''

    def __init__(self,startcmd=None):
        self.__dict__['_ep'] = engine.open(startcmd);
        self.pymatlab2_err=0

    def __del__(self):
        engine.close(self._ep);

    def __getattr__(self,name):
        t = self.call('exist',name)
        if t in (1,):
            return self.get(name)
        if t in (2,3,4,5,6):
            return MatlabFunc(self,name)
        raise Exception('{0} is not a variable or a function'.format(name))

    def __setattr__(self,name,value):
        self.put(name,value)

    def __getitem__(self,name):
        return self.__getattr__(name)

    def __setitem__(self,name,value):
        self.__setattr__(name,value)

    def get(self,name):
        return engine.getVariable(self._ep,name);

    def put(self,name,value):
        engine.putVariable(self._ep,name,value);

    def eval(self,string):
        rstr = engine.evalString(self._ep,self._evalstr.format(string));
        if self.get('pymatlab2_err') == 1:
            raise Exception(self.get('pymatlab2_exception'))
        return rstr

    def getVisible(self):
        return engine.getVisible(self._ep);

    def setVisible(self,value):
        engine.setVisible(self._ep,value);

    def call(self,func,*args,**kwds):
        rvalues = kwds['rvalues'] if 'rvalues' in kwds.keys() else 1
        arglst = ['pymatlab2_a{0}'.format(i) for i in range(len(args))]
        for a,v in zip(arglst,args):
            self.put(a,v)
        retlst = ['pymatlab2_r{0}'.format(i) for i in range(rvalues)]
        self.eval('[{0}]={1}({2})'.format(str.join(',',retlst),func,str.join(',',arglst)))
        ret = tuple([self.get(r) for r in retlst]) if rvalues > 1 else self.get(retlst[0])
        self.eval('clear {0} {1}'.format(str.join(' ',arglst),str.join(' ',retlst)))
        return ret

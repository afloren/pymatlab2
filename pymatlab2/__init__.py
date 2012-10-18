import engine
import sys
import dis

def expecting():
    f = sys._getframe().f_back.f_back
    i = f.f_lasti + 3
    bytecode = f.f_code.co_code
    instruction = ord(bytecode[i])
    while True:
        if instruction == dis.opmap['DUP_TOP']:
            if ord(bytecode[i+1]) == dis.opmap['UNPACK_SEQUENCE']:
                return ord(bytecode[i+2])
            i += 4
            instruction = ord(bytecode[i])
            continue
        if instruction == dis.opmap['STORE_NAME']:
            return 1
        if instruction == dis.opmap['UNPACK_SEQUENCE']:
            return ord(bytecode[i+1])
        return 0

class MatlabFunc:
    def __init__(self,session,name):
        self._session = session
        self._name = name

    def __call__(self,*args,**kwds):
        if 'rvalues' not in kwds.keys():
            kwds['rvalues'] = expecting()
        return self._session.call(self._name,*args,**kwds)

class Session:
    _evalstr = """
try
  pymatlab2_err=0;
  pymatlab2_errstr='';
  {0};
catch e
  pymatlab2_err=1;
  pymatlab2_errstr=getReport(e,'extended');
end
"""

    def __init__(self,startcmd=None):
        self.__dict__['_ep'] = engine.open(startcmd);
        self.pymatlab2_err=0

    def __del__(self):
        engine.close(self._ep);

    def __getattr__(self,name):
        t = self.call('exist',name,rvalues=1)
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
        rstr = engine.evalString(self._ep,self._evalstr.format(string))
        if self.get('pymatlab2_err') is 1:
            raise Exception(self.get('pymatlab2_errstr'))
        return rstr

    def getVisible(self):
        return engine.getVisible(self._ep);

    def setVisible(self,value):
        engine.setVisible(self._ep,value);

    def call(self,func,*args,**kwds):
        avalues = len(args)
        arglst = ['pymatlab2_a{0}'.format(i) for i in range(avalues)]        
        rvalues = kwds['rvalues'] if 'rvalues' in kwds.keys() else expecting()
        retlst = ['pymatlab2_r{0}'.format(i) for i in range(rvalues)]

        for a,v in zip(arglst,args):
            self.put(a,v)
        
        rstr = ''

        if rvalues > 0:
            rstr = self.eval('[{0}]={1}({2})'.format(','.join(retlst),func,','.join(arglst)))
        else:
            rstr = self.eval('{0}({1})'.format(func,','.join(arglst)))
        
        print(rstr)

        ret = None
        if rvalues > 1:
            ret = tuple([self.get(r) for r in retlst]) 
        elif rvalues is 1:
            ret = self.get(retlst[0])
        
        if (avalues+rvalues) > 0:
            self.eval('clear {0} {1}'.format(' '.join(arglst),' '.join(retlst)))
        
        return ret

//
//  engine.c
//  PyMatlab.engine
//
//  Created by Andrew Floren on 9/11/12.
//
//

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <matrix.h>
#include <engine.h>

const size_t bufferLength = 4096;

static PyObject *engineError;

static PyObject * engine_open(PyObject *self, PyObject *args)
{
  const char *startcmd = NULL;
  Engine *ep;

  if(!PyArg_ParseTuple(args, "|z", &startcmd))
    return NULL;
  
  ep = engOpen(startcmd);

  if(ep == NULL) {
    PyErr_SetString(engineError, "engOpen function returned NULL");
    return NULL;
  }

  return Py_BuildValue("l",ep);
}

static PyObject * engine_close(PyObject *self, PyObject *args)
{
  Engine *ep;
  int sts;

  if(!PyArg_ParseTuple(args, "l", &ep))
    return NULL;

  sts = engClose(ep);

  if(sts != 0) {
    PyErr_SetString(engineError, "Failed to close engine handle");
    return NULL;
  }

  Py_RETURN_NONE;
}

int mxtonpy[] = {
  NPY_USERDEF,
  NPY_USERDEF,
  NPY_USERDEF,
  NPY_BOOL,
  NPY_CHAR,
  NPY_USERDEF,
  NPY_DOUBLE,
  NPY_FLOAT,
  NPY_SHORT, NPY_USHORT,
  NPY_INT, NPY_UINT,
  NPY_LONG, NPY_ULONG,
  NPY_LONGLONG, NPY_ULONGLONG,
  NPY_USERDEF
};

typedef struct {
  PyObject_HEAD
} StructObject;

static PyTypeObject StructObjectType = {
  PyObject_HEAD_INIT(NULL)
  0,
  "PyMatlab.Struct",
  sizeof(StructObject),
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  Py_TPFLAGS_DEFAULT,
  "Matlab struct object"
};

static PyObject * mxToPy(mxArray *mx)
{
  int classid = mxGetClassID(mx);
  const size_t nd = mxGetNumberOfDimensions(mx);
  const size_t *dims = mxGetDimensions(mx);
  int type_num = mxtonpy[classid];
  void *data = mxGetData(mx);
  int itemsize = mxGetElementSize(mx);
  
  switch(classid) {
  case mxLOGICAL_CLASS: 
  case mxDOUBLE_CLASS:
  case mxSINGLE_CLASS:
  case mxINT8_CLASS:
  case mxUINT8_CLASS:
  case mxINT16_CLASS:
  case mxUINT16_CLASS:
  case mxINT32_CLASS:
  case mxUINT32_CLASS:
  case mxINT64_CLASS:
  case mxUINT64_CLASS:
    return PyArray_New(&PyArray_Type,nd,dims,type_num,NULL,data,itemsize,NPY_F_CONTIGUOUS,NULL);
  case mxCHAR_CLASS:
    return PyString_FromString(mxArrayToString(mx));
  case mxSTRUCT_CLASS:
    {
      PyObject *obj = PyObject_Call((PyObject *) &StructObjectType, PyTuple_New(0), NULL);
      int i,numFields = mxGetNumberOfFields(mx);
      for(i=0;i<numFields;i++) {
	const char *fieldName = mxGetFieldByNumber(mx,0,i);
	PyObject_SetAttrString(obj,fieldName,mxToPy(mxGetField(mx,0,fieldName)));
      }
    }
  case mxCELL_CLASS:
  case mxFUNCTION_CLASS:
    PyErr_SetString(engineError, "Not yet implemented");
    return NULL;
  case mxVOID_CLASS:
  case mxUNKNOWN_CLASS:
    PyErr_SetString(engineError, "Bad  mx type");
    return NULL;
  }

  PyErr_SetString(engineError, "Unknown mx type");
  return NULL;
}

static PyObject * engine_getVariable(PyObject *self, PyObject *args)
{
  Engine *ep;
  const char *name;
  mxArray *mx;

  if(!PyArg_ParseTuple(args, "ls", &ep, &name))
    return NULL;

  mx = engGetVariable(ep,name);

  return mxToPy(mx);
}

static mxClassID sizeToMx(size_t sz, bool s)
{
  switch(sz) {
  case 1:
    return s ? mxINT8_CLASS : mxUINT8_CLASS;
  case 2:
    return s ? mxINT16_CLASS : mxUINT16_CLASS;
  case 4:
    return s ? mxINT32_CLASS : mxUINT32_CLASS;
  case 8:
    return s ? mxINT64_CLASS : mxUINT64_CLASS;
  }
  return mxUNKNOWN_CLASS;
}

size_t zerodim[] = {1};

static mxArray * createNumericArray(size_t nd, size_t *dims, mxClassID classid, void *data, size_t sz)
{
  if(nd==0) {
    nd = 1;
    dims = zerodim;
  }
  mxArray *mx = mxCreateNumericArray(nd,dims,classid,mxREAL);
  void *dst = mxGetData(mx);
  memcpy(dst,data,sz);
  return mx;
}

static mxArray * createStringArray(size_t nd, size_t *dims, mxClassID classid, void *data, size_t sz)
{
  if(nd==0) {
    char *str = malloc(sz+1);
    memcpy(str,data,sz);
    str[sz] = NULL;
    return mxCreateString(str);
  }
  PyErr_SetString(engineError,"Not yet implemented");
  return NULL;
}

static mxClassID npy2mx(int type_num)
{
  switch(type_num) {
  case NPY_BOOL:
    return mxLOGICAL_CLASS;
  case NPY_BYTE:
    return mxINT8_CLASS;
  case NPY_UBYTE:
    return mxUINT8_CLASS;
  case NPY_FLOAT:
    return mxSINGLE_CLASS;
  case NPY_DOUBLE:
    return mxDOUBLE_CLASS;
  case NPY_SHORT:
    return sizeToMx(sizeof(npy_short),true);
  case NPY_USHORT:
    return sizeToMx(sizeof(npy_ushort),false);
  case NPY_INT:
    return sizeToMx(sizeof(npy_int),true);
  case NPY_UINT:
    return sizeToMx(sizeof(npy_uint),false);
  case NPY_LONG:
    return sizeToMx(sizeof(npy_long),true);
  case NPY_ULONG:
    return sizeToMx(sizeof(npy_ulong),false);
  case NPY_LONGLONG:
    return sizeToMx(sizeof(npy_longlong),true);
  case NPY_ULONGLONG:
    return sizeToMx(sizeof(npy_ulonglong),false);
  }
  return mxUNKNOWN_CLASS;
}

static mxArray * pyToMx(PyObject *py)
{
  PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OF(py,NPY_F_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE);
  int type_num = PyArray_TYPE(arr);
  size_t nd = PyArray_NDIM(arr);
  size_t *dims = PyArray_DIMS(arr);
  void *data = PyArray_DATA(arr);
  size_t sz = PyArray_NBYTES(arr);
  mxClassID classid = mxUNKNOWN_CLASS;
  mxArray *mx = NULL;

  switch(type_num) {
  case NPY_BOOL:
  case NPY_BYTE:
  case NPY_UBYTE:
  case NPY_FLOAT:
  case NPY_DOUBLE:
  case NPY_SHORT:
  case NPY_USHORT:
  case NPY_INT:
  case NPY_UINT:
  case NPY_LONG:
  case NPY_ULONG:
  case NPY_LONGLONG:
  case NPY_ULONGLONG:
    return createNumericArray(nd,dims,npy2mx(type_num),data,sz);
  case NPY_STRING:
  case NPY_UNICODE:
  case NPY_CHAR:
    return createStringArray(nd,dims,npy2mx(type_num),data,sz);
  case NPY_CFLOAT:
  case NPY_CDOUBLE:
  case NPY_OBJECT:
  case NPY_LONGDOUBLE:
  case NPY_CLONGDOUBLE:
    PyErr_SetString(engineError, "Not yet implemented");
    return NULL;
  case NPY_VOID:
  case NPY_NTYPES:
  case NPY_NOTYPE:
  case NPY_USERDEF:
    PyErr_SetString(engineError, "Bad type_num");
    return NULL;
  }
      
  PyErr_SetString(engineError, "Unknown type_num");
  return NULL;
}

static PyObject * engine_putVariable(PyObject *self, PyObject *args)
{
  Engine *ep;
  const char *name;
  PyObject *py;
  mxArray *pm;
  int sts;

  if(!PyArg_ParseTuple(args, "lsO", &ep, &name, &py))
    return NULL;

  pm = pyToMx(py);

  sts = engPutVariable(ep,name,pm);

  if(sts != 0) {
    PyErr_SetString(engineError, "Unable to put variable.");
  }

  Py_RETURN_NONE;
}

static PyObject * engine_evalString(PyObject *self, PyObject *args)
{
  Engine *ep;
  const char *string;
  char buffer[bufferLength];
  int sts;

  if(!PyArg_ParseTuple(args, "ls", &ep, &string))
    return NULL;

  sts = engOutputBuffer(ep,buffer,bufferLength);

  if(sts != 0) {
    PyErr_SetString(engineError, "The engine pointer is invalid or NULL");
    return NULL;
  }

  sts = engEvalString(ep,string);

  if(sts != 0) {
    PyErr_SetString(engineError, "The engine pointer is invalid or NULL");
    return NULL;
  }

  return Py_BuildValue("s",buffer);
}

static PyObject * engine_getVisible(PyObject *self, PyObject *args)
{
  Engine *ep;
  bool value;
  int sts;

  if(!PyArg_ParseTuple(args, "l", &ep))
    return NULL;

  sts = engGetVisible(ep,&value);

  if(sts != 0) {
    PyErr_SetString(engineError, "Failed to get visibility of Matlab engine session");
  }

  return PyBool_FromLong(value);
}

static PyObject * engine_setVisible(PyObject *self, PyObject *args)
{
  Engine *ep;
  bool value;
  int sts;

  if(!PyArg_ParseTuple(args, "li", &ep, &value))
    return NULL;

  sts = engSetVisible(ep,value);

  if(sts != 0) {
    PyErr_SetString(engineError, "Failed to set visibility of Matlab engine session");
  }

  Py_RETURN_NONE;
}

static PyMethodDef engineMethods[] = {
  {"open", engine_open, METH_VARARGS, "Open a Matlab engine handle."},
  {"close", engine_close, METH_VARARGS, "Close a Matlab engine handle."},
  {"getVariable", engine_getVariable, METH_VARARGS, "Get a variable from the Matlab engine session."},
  {"putVariable", engine_putVariable, METH_VARARGS, "Put a variable in the Matlab engine session."},
  {"evalString", engine_evalString, METH_VARARGS, "Evaluate a string in the Matlab engine session."},
  {"getVisible", engine_getVisible, METH_VARARGS, "Get the visibility of the Matlab engine session."},
  {"setVisible", engine_setVisible, METH_VARARGS, "Set the visibility of the Matlab engine session."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initengine(void)
{
  PyObject *m;
  import_array();
  m = Py_InitModule("engine", engineMethods);
  if (m == NULL)
    return;
  
  engineError = PyErr_NewException("engine.Error", NULL, NULL);
  Py_IncRef(engineError);
  PyModule_AddObject(m, "Error", engineError);
}




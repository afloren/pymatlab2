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

static PyObject * mxToPy(mxArray *mx)
{
  int classid = mxGetClassID(mx);
  switch(classid) {
  case mxLOGICAL_CLASS: 
  case mxCHAR_CLASS:
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
    {
      const size_t nd = mxGetNumberOfDimensions(mx);
      const size_t *dims = mxGetDimensions(mx);
      int type_num = mxtonpy[classid];
      void *data = mxGetData(mx);
      int itemsize = mxGetElementSize(mx);
      
      return PyArray_New(&PyArray_Type, nd, dims, type_num, NULL, data, itemsize, NPY_F_CONTIGUOUS, NULL);
    }
    break;
  case mxSTRUCT_CLASS:
  case mxCELL_CLASS:
  case mxFUNCTION_CLASS:
    PyErr_SetString(engineError, "Not yet implemented");
    return NULL;
  case mxVOID_CLASS:
  case mxUNKNOWN_CLASS:
  default:
    PyErr_SetString(engineError, "Unknown or bad  mx type");
    return NULL;
  }

  PyErr_SetString(engineError, "Something went very wrong");
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

mxClassID npytomx[] = { 
  mxLOGICAL_CLASS,
  mxINT8_CLASS, mxUINT8_CLASS,
  mxINT8_CLASS, mxUINT8_CLASS,
  mxINT16_CLASS, mxUINT16_CLASS,
  mxINT32_CLASS, mxUINT32_CLASS,
  mxINT64_CLASS, mxUINT64_CLASS,
  mxSINGLE_CLASS, mxDOUBLE_CLASS, mxUNKNOWN_CLASS,
  mxUNKNOWN_CLASS, mxUNKNOWN_CLASS, mxUNKNOWN_CLASS,
  mxUNKNOWN_CLASS,
  mxCHAR_CLASS, mxCHAR_CLASS,
  mxUNKNOWN_CLASS,
  mxUNKNOWN_CLASS,
  mxCHAR_CLASS,
};

static mxArray * pyToMx(PyObject *py)
{
  PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OF(py,NPY_F_CONTIGUOUS);
  int type_num = PyArray_TYPE(arr);
  switch(type_num) {
  case NPY_BOOL:
  case NPY_BYTE:
  case NPY_UBYTE:
  case NPY_SHORT:
  case NPY_USHORT:
  case NPY_INT:
  case NPY_UINT:
  case NPY_LONG:
  case NPY_ULONG:
  case NPY_LONGLONG:
  case NPY_ULONGLONG:
  case NPY_FLOAT:
  case NPY_DOUBLE:
  case NPY_LONGDOUBLE:
  case NPY_STRING:
  case NPY_UNICODE:
  case NPY_CHAR:
    {
      size_t nd = PyArray_NDIM(arr);
      size_t *dims = PyArray_DIMS(arr);
      mxClassID classid = npytomx[type_num];
      mxArray *mx = mxCreateNumericArray(nd,dims,classid,mxREAL);
      void *dst = mxGetData(mx);
      void *src = PyArray_DATA(arr);
      size_t sz = PyArray_NBYTES(arr);
      memcpy(dst,src,sz);
      return mx;
    }
    break;
  case NPY_CFLOAT:
  case NPY_CDOUBLE:
  case NPY_CLONGDOUBLE:
  case NPY_OBJECT:
    PyErr_SetString(engineError, "Not yet implemented");
    return NULL;
  case NPY_VOID:
  case NPY_NTYPES:
  case NPY_NOTYPE:
  case NPY_USERDEF:
  default:
    PyErr_SetString(engineError, "Unknown or bad type_num");
    return NULL;
  }

  PyErr_SetString(engineError, "Something went very wrong");
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




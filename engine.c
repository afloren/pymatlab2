//
//  engine.c
//  PyMatlab.engine
//
//  Created by Andrew Floren on 9/11/12.
//
//

#include <Python/Python.h>
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

static PyObject * mxToPy(mxArray *mx)
{
  switch(mxGetClassID(mx)) {
  case mxLOGICAL_CLASS:
  case mxCHAR_CLASS:
  case mxVOID_CLASS:
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
  case mxSTRUCT_CLASS:
  case mxCELL_CLASS:
  case mxFUNCTION_CLASS:
    PyErr_SetString(engineError, "Not yet implemented");
    return NULL;
  case mxUNKNOWN_CLASS:
  default:
    PyErr_SetString(engineError, "Unknown mx type");
    return NULL:
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

static mxArray * pyToMx(PyObject *py)
{
  return NULL;
}

static PyObject * engine_putVariable(PyObject *self, PyObject *args)
{
  Engine *ep;
  const char *name;
  PyObject *py;
  mxArray *pm;
  int sts;

  if(!PyArg_ParseTuple(args, "lso", &ep, &name, &py))
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
  
  m = Py_InitModule("engine", engineMethods);
  if (m == NULL)
    return;
  
  engineError = PyErr_NewException("engine.Error", NULL, NULL);
  Py_IncRef(engineError);
  PyModule_AddObject(m, "Error", engineError);
}




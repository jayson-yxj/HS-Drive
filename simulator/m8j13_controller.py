"""Run exact official PID methods in the existing CARLA-only environment.

The source imports torch but never uses it in these classes. AST extraction
retains their method bodies unchanged, without installing torch into CARLA.
"""
import ast
from pathlib import Path
import numpy as np
from datasets.m8j12_contract import DryRunController


class ArrayTensor:
    def __init__(self,value):self.value=np.asarray(value)
    def numpy(self):return self.value
    def __getitem__(self,key):return ArrayTensor(self.value[key])


def native_controller(source):
    tree=ast.parse(Path(source).read_text())
    body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('PID','PIDController')]
    from collections import deque
    scope={'np':np,'deque':deque}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),scope)
    wrapper=DryRunController.__new__(DryRunController)
    wrapper.controller=scope['PIDController'](None)
    return wrapper


def native_command(controller,path,speeds,speed,target):
    return controller.command(dict(lat_reg_final=ArrayTensor(path),lon_reg_final=ArrayTensor(speeds)),speed,target)

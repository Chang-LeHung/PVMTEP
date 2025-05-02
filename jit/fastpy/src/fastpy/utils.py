"""."""

import ast
import inspect
from collections.abc import Callable
from typing import Any
import threading
import ctypes

from llvmlite import binding  # type: ignore[import]

from .ir import (
    FastPyCodeGenerator,
    FastPyFunction,
)
from .visitor import FastPyVisitor


class FastPyCompiler:
    """."""

    def __init__(self, func: Callable[..., Any]):
        """."""
        self.visitor = FastPyVisitor(func)
        self.func = func
        self.compiled = False
        self.res: None | FastPyFunction = None
        self.generator: None | FastPyCodeGenerator = None
        self.tree: None | ast.Module = None

    def build(self) -> FastPyFunction:
        """."""
        if self.compiled:
            assert self.res is not None
            return self.res
        source = inspect.getsource(self.func)
        self.tree = ast.parse(source)
        self.visitor.visit(self.tree)
        self.res = self.visitor.get_func_ir()
        self.generator = FastPyCodeGenerator(self.func, self.visitor.all_basic_blocks)
        self.res.code_gen(self.generator)
        self.compiled = True
        return self.res

    def get_res(self) -> FastPyFunction:
        """."""
        return self.build()

    def dump(self) -> str:
        """."""
        if self.generator is not None:
            return self.generator.asm_code
        self.build()
        assert self.generator is not None
        return self.generator.asm_code


binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()


def create_execution_engine():
    """."""
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = binding.parse_assembly("")
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)
    return engine


root_engine = create_execution_engine()
lock = threading.RLock()


def compile_ir(engine: binding.ExecutionEngine, llvm_ir: str, opt_level: int = 2):
    """Compile LLVM IR string with optimization and add it to the execution engine.

    :param engine: An initialized ExecutionEngine
    :param llvm_ir: LLVM IR in string form
    :param opt_level: Optimization level (0 to 3)
    :return: Optimized LLVM module
    """
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    pmb = binding.PassManagerBuilder()
    pmb.opt_level = opt_level

    pm = binding.ModulePassManager()
    pmb.populate(pm)

    pm.run(mod)

    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    return mod


def get_ctypes_type(ty: type):
    """."""
    if ty is int:
        return ctypes.c_int
    elif ty is float:
        return ctypes.c_float
    elif ty is bool:
        return ctypes.c_bool
    elif ty is str:
        return ctypes.c_char_p
    elif ty is inspect._empty:
        return ctypes.c_void_p
    else:
        raise ValueError(f"unsupported type {ty}")


class HelperCaller:
    """."""

    def __init__(self, func: Callable[..., Any], log: bool = False):
        """."""
        print(func.__globals__.keys())
        source = inspect.getsource(func)
        self.tree = ast.parse(source)
        self.compiler = FastPyCompiler(func)
        self.compiler.build()
        assert self.compiler.tree is not None
        if log:
            print(self.compiler.dump())
        with lock:
            _ = compile_ir(root_engine, self.compiler.dump())
        self.func = func

    @property
    def ir(self):
        """."""
        return self.compiler.dump()

    @property
    def ast_tree(self):
        """."""
        return ast.dump(self.compiler.tree, indent=4)

    def __call__(self, *args, **kwds):
        """."""
        assert len(kwds) == 0
        func = root_engine.get_function_address(self.func.__name__)
        sig = inspect.signature(self.func)
        ret = get_ctypes_type(sig.return_annotation)
        args_type = [get_ctypes_type(arg.annotation) for arg in sig.parameters.values()]
        return ctypes.CFUNCTYPE(ret, *args_type)(func)(*args)


def jit(func: Callable):
    """."""
    return HelperCaller(func)

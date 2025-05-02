"""."""

import inspect
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from threading import RLock
from typing import Any, Literal

from llvmlite import binding, ir  # type: ignore[import]
from llvmlite.ir import AllocaInstr, Argument  # type: ignore[import]
from llvmlite.ir.instructions import Instruction  # type: ignore[import]
from llvmlite.ir.values import Constant  # type: ignore[import]

i1 = ir.IntType(1)
i32 = ir.IntType(32)
i64 = ir.IntType(64)
f32 = ir.FloatType()
f64 = ir.DoubleType()
void = ir.VoidType()

binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()


type FastPyArg = Argument | Constant | Instruction | AllocaInstr
type FastPyType = Literal["int", "float", "str", "bool", "void"]

_NAME = "FastPy"
root_module = ir.Module(name=_NAME)
lock = RLock()


def get_matched_type_by_name(arg: FastPyArg):
    """."""
    match arg:
        case "int":
            return i32
        case "float":
            return f32
        case "bool":
            return i1
        case "void":
            return void
        case _:
            raise ValueError(f"unsupported type {arg}")


def get_matched_type_by_class(arg: type):
    """."""
    if arg is int:
        return i32
    elif arg is float:
        return f32
    elif arg is bool:
        return i1
    elif arg is inspect._empty:
        return void
    else:
        raise ValueError(f"unsupported type {arg}")


class FastPyCodeGenerator:
    """A stack-based code generator."""

    def __init__(self, func: Callable[..., Any], blocks: list[str]):
        """."""
        self.module = root_module
        self.py_func = func
        self.params: list[str] = []
        self.params_type: list[type] = []
        self.return_type: type | None = None
        self._parse_func()
        with lock:
            self.func = ir.Function(
                self.module,
                ir.FunctionType(self.return_type, self.params_type),
                name=self.py_func.__name__,
            )
        self.cfunc_args = self.func.args
        self.stack: list[FastPyArg] = []
        self.vars: dict[str, FastPyArg] = dict()
        self.blocks: dict[str, ir.Block] = dict()
        for bb in blocks:
            self._new_block(bb)
        self.cur_block = self._get_block(blocks[0])
        self.builder = ir.IRBuilder(self.cur_block)

        self._init_args()

    def get_return_type(self) -> FastPyType:
        """."""
        sig = inspect.signature(self.py_func)
        t = sig.return_annotation
        if t is inspect._empty:
            return "void"
        else:
            return t.__name__

    def _init_args(self):
        for idx, parm in enumerate(self.cfunc_args):
            parm.name = self.params[idx]
            self.vars[parm.name] = parm

    def _parse_func(self):
        sig = inspect.signature(self.py_func)
        for param in sig.parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                self.params.append(param.name)
                self.params_type.append(get_matched_type_by_class(param.annotation))
            else:
                raise ValueError(f"unsupported parameter {param.kind} in {self.py_func.__name__}")
        ret = sig.return_annotation
        self.return_type = get_matched_type_by_class(ret)

    def _insert_block(self, name: str, bb: ir.Block):
        """."""
        self.blocks[name] = bb

    def _new_block(self, name: str):
        """."""
        bb = self.func.append_basic_block(name)
        self._insert_block(name, bb)
        return bb

    def is_terminated(self, bb_name: str) -> bool:
        """."""
        return bb_name in self.blocks and self.blocks[bb_name].is_terminated

    @property
    def cur_block_name(self) -> str:
        """."""
        return self.cur_block.name

    def _get_block(self, name: str) -> "FastPyBasicBlock":
        """."""
        assert name in self.blocks
        return self.blocks[name]

    def branch(self, bb_name: str):
        """."""
        self.builder.branch(self._get_block(bb_name))

    def return_val(self):
        """."""
        self.builder.ret(self.pop())

    def return_void(self):
        """."""
        self.builder.ret_void()

    def call(self, func_name: str, args: list[FastPyArg]):
        """."""
        assert func_name == self.py_func.__name__
        ret = self.builder.call(self.func, args)
        if self.return_type != void:
            self.push(ret)

    def cbranch(self, then: str, else_: str):
        """."""
        self.builder.cbranch(self.pop(), self._get_block(then), self._get_block(else_))

    def change_cur_block(self, bb_name: str):
        """."""
        self.cur_block = self._get_block(bb_name)
        self.builder.position_at_start(self.cur_block)

    def add(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.add(lhs, rhs)
        self.push(res)

    def sub(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.sub(lhs, rhs)
        self.push(res)

    def mul(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.mul(lhs, rhs)
        self.push(res)

    def div(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.sdiv(lhs, rhs)
        self.push(res)

    def lshift(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.shl(lhs, rhs)
        self.push(res)

    def rshift(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.ashr(lhs, rhs)
        self.push(res)

    def mod(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.srem(lhs, rhs)
        self.push(res)

    def pow(self):
        """."""
        raise ValueError("unsupported operation")

    def bitand(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.and_(lhs, rhs)
        self.push(res)

    def eq(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.icmp_signed("==", lhs, rhs)
        self.push(res)

    def neq(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.icmp_signed("!=", lhs, rhs)
        self.push(res)

    def gt(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.icmp_signed(">", lhs, rhs)
        self.push(res)

    def lte(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.icmp_signed("<=", lhs, rhs)
        self.push(res)

    def gte(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.icmp_signed(">=", lhs, rhs)
        self.push(res)

    def lt(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.icmp_signed("<", lhs, rhs)
        self.push(res)

    def bitor(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.or_(lhs, rhs)
        self.push(res)

    def bitxor(self):
        """."""
        lhs = self.pop()
        rhs = self.pop()
        res = self.builder.xor(lhs, rhs)
        self.push(res)

    def neg(self):
        """."""
        lhs = self.pop()
        res = self.builder.neg(lhs)
        self.push(res)

    def push(self, val: FastPyArg):
        """."""
        self.stack.append(val)

    def get_var(self, name: str) -> FastPyArg:
        """."""
        return self.vars[name]

    def pop(self) -> FastPyArg:
        """."""
        return self.stack.pop()

    def alloc(self, ty: FastPyType, name: str = ""):
        """."""
        match ty:
            case "int":
                return self.builder.alloca(i32, name=name)
            case "float":
                return self.builder.alloca(f64, name=name)
            case "str":
                # TODO: support str
                return self.builder.alloca(i32, name=name)
            case _:
                raise ValueError("unsupported type")

    def prepare_var(self, name: str, ty: FastPyType):
        """Create a new variable if not exists."""
        if name not in self.vars:
            var = self.alloc(ty, name=name)
            self.vars[name] = var

    def store(self, name: str):
        """."""
        assert name in self.vars
        val = self.pop()
        if isinstance(val, Constant) or isinstance(val, Argument) or isinstance(val, Instruction):
            self.builder.store(val, self.vars[name])
            return
        self.builder.store(self.builder.load(val), self.vars[name])

    @property
    def asm_code(self):
        """."""
        return str(self.func)


class FastPyIR(ABC):
    """The base class of FastPy types."""

    @abstractmethod
    def is_consntant(self):
        """Return True if self is a constant."""
        pass

    @abstractmethod
    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        pass

    @abstractmethod
    def type(self) -> FastPyType:
        """."""
        pass

    @abstractmethod
    def verify(self) -> bool:
        """."""
        pass


class FastPyConstant(FastPyIR):
    """."""

    def __init__(self, val: int | float | str):
        """val: the constant value."""
        super().__init__()
        self.val = val

    def is_consntant(self) -> bool:
        """."""
        return True

    def code_gen(self, ctx: FastPyCodeGenerator) -> FastPyArg:
        """."""
        match self.val:
            case int() as v:
                ctx.push(i32(v))
                return i32(v)
            case float() as v:
                ctx.push(f64(v))
                return f64(v)
            case str() as v:
                # TODO: support str
                return i32(len(v))
        raise ValueError("unsupported constant type")

    def type(self) -> FastPyType:
        """."""
        match self.val:
            case int():
                return "int"
            case float():
                return "float"
            case str():
                return "str"

    def verify(self):
        """."""
        return True


class FastPyBinaryOp(FastPyIR):
    """."""

    def __init__(self, op: str, lhs: FastPyIR, rhs: FastPyIR):
        """."""
        super().__init__()
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def is_consntant(self):
        """."""
        return self.rhs.is_consntant() and self.lhs.is_consntant()

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        self.rhs.code_gen(ctx)
        self.lhs.code_gen(ctx)
        match self.op:
            case "+":
                ctx.add()
            case "-":
                ctx.sub()
            case "*":
                ctx.mul()
            case "/":
                ctx.div()
            case "<<":
                ctx.lshift()
            case ">>":
                ctx.rshift()
            case "%":
                ctx.mod()
            case "**":
                ctx.pow()
            case "&":
                ctx.bitand()
            case "|":
                ctx.bitor()
            case "^":
                ctx.bitxor()

    def type(self) -> FastPyType:
        """."""
        return self.lhs.type()

    def verify(self):
        """."""
        return self.lhs.verify() and self.rhs.verify() and self.lhs.type() == self.rhs.type()


class FastPyVar(FastPyIR):
    """."""

    def __init__(self, var_name: str, var_type: FastPyType | None = None):
        """."""
        super().__init__()
        self.var_name = var_name
        self.var_type = var_type

    @property
    def ptr(self):
        """."""
        return self._ptr

    def type(self) -> FastPyType:
        """."""
        if self.var_type is None:
            raise ValueError("var type is not set")
        return self.var_type

    def set_type(self, var_type: FastPyType) -> None:
        """."""
        self.var_type = var_type

    def valid(self) -> bool:
        """."""
        return self.var_type is not None

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        assert self.valid()
        val = ctx.get_var(self.var_name)
        if isinstance(val, ir.AllocaInstr):
            ctx.push(ctx.builder.load(val))
        else:
            ctx.push(val)

    def verify(self):
        """."""
        return True

    def is_consntant(self):
        """."""
        return False


class FastPyAssign(FastPyIR):
    """."""

    def __init__(self, name: str, value: FastPyIR, var: FastPyVar):
        """."""
        super().__init__()
        self.name = name
        self.value = value
        self.var = var

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        ctx.prepare_var(self.var.var_name, self.var.type())
        self.value.code_gen(ctx)
        ctx.store(self.name)

    def type(self) -> FastPyType:
        """."""
        return self.value.type()

    def verify(self) -> bool:
        """."""
        self.value.verify()
        return True

    def is_consntant(self) -> bool:
        """."""
        return self.value.is_consntant()


class FastPyBasicBlock(FastPyIR):
    """."""

    cnt = 0
    lock = threading.RLock()

    def __init__(self, name: str = ""):
        """."""
        super().__init__()
        self.bb_name: str = name
        if name == "":
            with lock:
                self.bb_name = f"bb_{FastPyBasicBlock.cnt}"
                FastPyBasicBlock.cnt += 1
        self.stmts: list[FastPyIR] = []
        self.nxt_block: FastPyBasicBlock | None = None

    def set_nxt_block(self, nxt_block: "FastPyBasicBlock"):
        """."""
        assert self.nxt_block is None
        self.nxt_block = nxt_block

    @property
    def name(self):
        """."""
        return self.bb_name

    @name.setter
    def name(self, val: str):
        self.bb_name = val

    def append(self, stmt: FastPyIR) -> None:
        """."""
        self.stmts.append(stmt)

    def pop(self) -> FastPyIR:
        """."""
        return self.stmts.pop()

    def is_consntant(self):
        """."""
        return False

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        ctx.change_cur_block(self.bb_name)
        for stmt in self.stmts:
            stmt.code_gen(ctx)
        if self.nxt_block is not None and not ctx.is_terminated(self.bb_name):
            ctx.branch(self.nxt_block.name)

    def type(self):
        """."""
        raise NotImplementedError()

    def verify(self) -> bool:
        """."""
        res = True
        for stmt in self.stmts:
            res = res and stmt.verify()
        return res

    def __iter__(self):
        """."""
        return iter(self.stmts)


class FastPyFunction(FastPyIR):
    """."""

    def __init__(self, func: Callable[..., Any]):
        """."""
        super().__init__()
        self._blocks: list[FastPyBasicBlock] = []
        self._func = func

    @property
    def func(self):
        """."""
        return self._func

    @property
    def basic_blocks(self) -> list[FastPyBasicBlock]:
        """."""
        return self._blocks

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        for block in self._blocks:
            block.code_gen(ctx)

    def type(self):
        """."""
        raise NotImplementedError()

    def is_consntant(self):
        """."""
        return False

    def verify(self) -> bool:
        """."""
        for block in self._blocks:
            if not block.verify():
                return False
        return True

    def append(self, bb: FastPyBasicBlock):
        """."""
        self._blocks.append(bb)


class FastPyIfIR(FastPyIR):
    """."""

    def __init__(
        self,
        condition: FastPyIR,
        true_block: FastPyBasicBlock,
        false_block: FastPyBasicBlock,
        gen_true: bool = True,
        gen_false: bool = True,
    ):
        """."""
        super().__init__()
        self.condition = condition
        self.true_block = true_block
        self.false_block = false_block
        self.gen_true = gen_true
        self.gen_false = gen_false

    def type(self):
        """."""
        raise NotImplementedError()

    def verify(self):
        """."""
        return self.condition.verify() and self.true_block.verify() and self.false_block.verify()

    def is_consntant(self):
        """."""
        raise NotImplementedError()

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        self.condition.code_gen(ctx)
        ctx.cbranch(self.true_block.name, self.false_block.name)

        if self.gen_true:
            ctx.change_cur_block(self.true_block.name)
            self.true_block.code_gen(ctx)

        if self.gen_false:
            ctx.change_cur_block(self.false_block.name)
            self.false_block.code_gen(ctx)


class FastPyCompareIR(FastPyIR):
    """."""

    def __init__(self, op: str, left: FastPyIR, right: FastPyIR):
        """."""
        super().__init__()
        self.op = op
        self.lhs = left
        self.rhs = right

    def type(self) -> FastPyType:
        """."""
        return "bool"

    def is_consntant(self):
        """."""
        return self.lhs.is_consntant() and self.rhs.is_consntant()

    def verify(self):
        """."""
        return self.lhs.verify() and self.rhs.verify() and self.lhs.type() == self.rhs.type()

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        self.rhs.code_gen(ctx)
        self.lhs.code_gen(ctx)
        match self.op:
            case "==":
                ctx.eq()
            case "!=":
                ctx.neq()
            case ">":
                ctx.gt()
            case ">=":
                ctx.gte()
            case "<":
                ctx.lt()
            case "<=":
                ctx.lte()
            case _:
                raise ValueError(f"Unknown operator {self.op}")


class FastPyReturn(FastPyIR):
    """."""

    def __init__(self, val: FastPyIR | None = None):
        """."""
        super().__init__()
        self.val = val

    def type(self) -> FastPyType:
        """."""
        return "void" if self.val is None else self.val.type()

    def is_consntant(self):
        """."""
        return self.val is None or self.val.is_consntant()

    def verify(self) -> bool:
        """."""
        return self.val is None or self.val.verify()

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        if self.val is None:
            return ctx.return_void()
        self.val.code_gen(ctx)
        assert (self.val is None and ctx.get_return_type() == "void") or (
            self.val is not None and self.val.type() == ctx.get_return_type()
        ), "return type mismatch"
        ctx.return_val()


class FastPyCallCFunc(FastPyIR):
    """."""

    def __init__(self, func_name: str, args: list[FastPyIR], return_type: FastPyType):
        """."""
        super().__init__()
        self.func_name = func_name
        self.args = args
        self.return_type = return_type

    def type(self) -> FastPyType:
        """."""
        return self.return_type

    def is_consntant(self):
        """."""
        return False

    def verify(self):
        """."""
        for arg in self.args:
            arg.verify()

    def code_gen(self, ctx: FastPyCodeGenerator):
        """."""
        args = []
        for arg in self.args:
            arg.code_gen(ctx)
            args.append(ctx.pop())
        ctx.call(self.func_name, args)

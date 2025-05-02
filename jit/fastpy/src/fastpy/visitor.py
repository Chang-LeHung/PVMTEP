"""A python hotspot jit compiler."""

import ast
import inspect
from collections.abc import Callable
from typing import Any, cast

from .ir import (
    FastPyAssign,
    FastPyBasicBlock,
    FastPyBinaryOp,
    FastPyCallCFunc,
    FastPyCompareIR,
    FastPyConstant,
    FastPyFunction,
    FastPyIfIR,
    FastPyReturn,
    FastPyVar,
    FastPyType,
)


class FastPyVisitor(ast.NodeVisitor):
    """A visitor to generator binary code via visiting python ast."""

    def __init__(self, func: Callable[..., Any]):
        """."""
        super().__init__()
        self.py_func = func
        self.func = FastPyFunction(func)
        self.cur_bb: FastPyBasicBlock = FastPyBasicBlock()
        self._append_code_gen_block(self.cur_bb)
        self.vars: dict[str, FastPyVar] = dict()
        self._all_blocks: list[str] = []
        self._all_blocks.append(self.cur_bb.bb_name)
        self._cur_if_rest_block: FastPyBasicBlock | None = None
        self._cur_while_rest_block: FastPyBasicBlock | None = None
        self.has_ret = False

    @property
    def all_basic_blocks(self) -> list[str]:
        """."""
        return self._all_blocks

    def get_func_ir(self) -> FastPyFunction:
        """."""
        return self.func

    def _append_code_gen_block(self, bb: FastPyBasicBlock):
        self.func.append(bb)

    def visit_List(self, node):
        """."""
        self.generic_visit(node)
        print(node.__dict__)

    def visit_Constant(self, node):
        """."""
        self.generic_visit(node)
        self.cur_bb.append(FastPyConstant(node.value))

    def visit_BinOp(self, node: ast.BinOp):
        """."""
        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)

    def visit_Add(self, node: ast.Add):
        """."""
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("+", lhs, rhs))

    def visit_Sub(self, node: ast.Sub):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("-", lhs, rhs))

    def visit_Mult(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("*", lhs, rhs))

    def visit_Div(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("/", lhs, rhs))

    def visit_LShift(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("<<", lhs, rhs))

    def visit_RShift(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp(">>", lhs, rhs))

    def visit_Mod(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("%", lhs, rhs))

    def visit_Pow(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("**", lhs, rhs))

    def visit_BitAnd(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("&", lhs, rhs))

    def visit_BitOr(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("|", lhs, rhs))

    def visit_BitXor(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("^", lhs, rhs))

    def visit_FloorDiv(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("//", lhs, rhs))

    def visit_Invert(self, node):
        """."""
        self.generic_visit(node)
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyBinaryOp("~", lhs, rhs))

    def visit_Name(self, node):
        """."""
        assert node.id in self.vars
        self.cur_bb.append(self.vars[node.id])

    def visit_Assign(self, node):
        """."""
        target = node.targets[0]
        value = node.value
        if isinstance(target, ast.Name):
            self.visit_single(target, value)
        elif isinstance(target, ast.Tuple):
            size = len(target.elts)
            if not isinstance(value, ast.Tuple):
                raise ValueError("unsupported target, require more values")
            if size != len(value.elts):
                raise ValueError(f"require {size} values, but got {len(value.elts)}")
            for i in range(size):
                self.visit_single(target.elts[i], value.elts[i])
        else:
            raise ValueError("unsupported target")

    def visit_AugAssign(self, node: ast.AugAssign):
        """."""
        self.visit(node.target)
        self.visit(node.value)
        self.visit(node.op)
        top = self.cur_bb.pop()
        assert isinstance(node.target, ast.Name)
        var = self.vars[node.target.id]
        self.cur_bb.append(FastPyAssign(node.target.id, top, var))

    def visit_single(self, target: ast.Name, value: ast.expr):
        """."""
        if target.id not in self.vars:
            self.vars[target.id] = FastPyVar(target.id)
        self.visit(value)
        top = self.cur_bb.pop()
        var = self.vars[target.id]
        if not var.valid():
            var.set_type(top.type())
        self.cur_bb.append(FastPyAssign(target.id, top, var))

    def visit_arguments(self, node: ast.arguments):
        """."""
        for arg in node.args:
            self.vars[arg.arg] = FastPyVar(arg.arg)
            if arg.annotation is None:
                raise ValueError(f"Params must have type annotation {arg.arg}")
            assert hasattr(arg.annotation, "id")
            self.vars[arg.arg].set_type(arg.annotation.id)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """."""
        self.visit(node.args)
        for stmt in node.body:
            self.visit(stmt)
        if not self.has_ret:
            self.has_ret = True
            self.cur_bb.append(FastPyReturn())

    def visit_Return(self, node: ast.Return):
        """."""
        self.has_ret = True
        if node.value is not None:
            self.visit(node.value)
            self.cur_bb.append(FastPyReturn(self.cur_bb.pop()))
        else:
            self.cur_bb.append(FastPyReturn())

    def visit(self, node):
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        if method not in FastPyVisitor.__dict__.keys():
            raise NotImplementedError(
                "FastPyVisitor does not support visit_" + node.__class__.__name__
            )
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def visit_For(self, node: ast.For):
        """."""
        return super().visit_For(node)

    def visit_While(self, node: ast.While):
        """."""
        # gen body block
        body = FastPyBasicBlock()
        body.name = body.name + "_while_body"
        self._all_blocks.append(body.name)

        # gen rest block
        nested = False
        while_rest_block = FastPyBasicBlock()
        while_rest_block.name = while_rest_block.name + "_while_rest"
        self._all_blocks.append(while_rest_block.name)
        self._append_code_gen_block(while_rest_block)
        if self._cur_while_rest_block is None:
            self._cur_while_rest_block = while_rest_block
        else:
            nested = True
            old_while_rest_block = self._cur_while_rest_block
            while_rest_block.set_nxt_block(old_while_rest_block)

        self.visit(node.test)
        self.cur_bb.append(
            FastPyIfIR(self.cur_bb.pop(), body, while_rest_block, gen_true=False, gen_false=False)
        )

        self.cur_bb.set_nxt_block(body)

        self.cur_bb = body
        for stmt in node.body:
            self.visit(stmt)

        self.visit(node.test)
        self.cur_bb.append(
            FastPyIfIR(self.cur_bb.pop(), body, while_rest_block, gen_true=False, gen_false=False)
        )

        self._append_code_gen_block(body)

        self.cur_bb = while_rest_block

        if nested:
            self._cur_while_rest_block = old_while_rest_block

    def visit_Module(self, node):
        """."""
        return self.generic_visit(node)

    def visit_Expr(self, node):
        """."""
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """."""
        func = cast(ast.Name, node.func)
        args = []
        for arg in node.args:
            self.visit(arg)
            args.append(self.cur_bb.pop())
        sig = inspect.signature(self.py_func)
        return_type: FastPyType
        if sig.return_annotation is inspect._empty:
            return_type = "void"
        else:
            return_type = sig.return_annotation.__name__
        self.cur_bb.append(FastPyCallCFunc(func.id, args, return_type))

    def _new_basic_block(self):
        """."""
        new_bb = FastPyBasicBlock()
        self._all_blocks.append(new_bb.name)
        return new_bb

    def visit_If(self, node: ast.If):
        """."""
        self.visit(node.test)
        old = self.cur_bb

        bb_body = FastPyBasicBlock()
        bb_body.name = bb_body.name + "_body"
        self._all_blocks.append(bb_body.name)

        bb_orelse = FastPyBasicBlock()
        bb_orelse.name = bb_orelse.name + "_orelse"
        self._all_blocks.append(bb_orelse.name)

        if_rest_bb = FastPyBasicBlock()
        if_rest_bb.name = if_rest_bb.name + "_if_rest"
        self._all_blocks.append(if_rest_bb.name)
        self._append_code_gen_block(if_rest_bb)

        nested = False
        if self._cur_if_rest_block is None:
            self._cur_if_rest_block = if_rest_bb
        else:
            nested = True
            old_if_rest_block = self._cur_if_rest_block
            if_rest_bb.set_nxt_block(old_if_rest_block)
            self._cur_if_rest_block = if_rest_bb

        self.cur_bb = bb_body
        for stmt in node.body:
            self.visit(stmt)

        self.cur_bb = bb_orelse
        for stmt in node.orelse:
            self.visit(stmt)

        self.cur_bb = old
        self.cur_bb.append(FastPyIfIR(self.cur_bb.pop(), bb_body, bb_orelse))

        self.cur_bb = if_rest_bb
        bb_body.set_nxt_block(if_rest_bb)
        bb_orelse.set_nxt_block(if_rest_bb)

        if nested:
            self._cur_if_rest_block = old_if_rest_block

    def visit_Compare(self, node: ast.Compare):
        """."""
        self.visit(node.left)
        self.visit(node.comparators[0])
        self.visit(node.ops[0])

    def visit_Gt(self, node: ast.Gt):
        """."""
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyCompareIR(">", lhs, rhs))

    def visit_Lt(self, node: ast.Lt):
        """."""
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyCompareIR("<", lhs, rhs))

    def visit_GtE(self, node: ast.GtE):
        """."""
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyCompareIR(">=", lhs, rhs))

    def visit_LtE(self, node: ast.LtE):
        """."""
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyCompareIR("<=", lhs, rhs))

    def visit_Eq(self, node: ast.Eq):
        """."""
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyCompareIR("==", lhs, rhs))

    def visit_NotEq(self, node: ast.NotEq):
        """."""
        rhs = self.cur_bb.pop()
        lhs = self.cur_bb.pop()
        self.cur_bb.append(FastPyCompareIR("!=", lhs, rhs))

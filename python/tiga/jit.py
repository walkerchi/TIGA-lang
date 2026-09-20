"""Orchestration-level AST capture of Python loops into gf_control ops.

``@tg.jit`` rewrites ``for i in range(k)`` and bare ``while cond:`` loops in
the decorated function into :func:`tiga.control.repeat` /
:func:`tiga.control.while_loop` calls at capture time, and activates an
automatic :class:`tiga.GraphProgram` context so sibling kernel calls in
the straight-line body fuse without an explicit ``@tg.program``. It is a
spelling layer only: the same primitives, lowering, VJP and fail-closed
contracts apply, and MessagePassing UDF regions are never AST-transformed.

Supported subset (every violation fails closed with a source location):

- ``for i in range(...)`` with Python-int bounds lowers to ``gf_control.repeat``.
  Reading ``i`` desugars it into an extra carried rank-zero int64 state.
- ``while <tensor condition>`` lowers to ``gf_control.while`` and requires the
  decorator resource bound ``@tg.jit(max_iterations=k)``.
- ``if <cond>: break`` as the *first* statement of a ``for`` body is an early
  exit and lowers to ``gf_control.while`` bounded by the range length.
- Loop-carried variables must be Tensors assigned before the loop; variables
  created inside the loop body stay local to one iteration.
- ``continue``, mid-body ``break``, ``while True``, loop ``else``, non-range
  iteration and data-dependent Python ``if`` are rejected.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from functools import wraps

from . import control as _control
from .tensor import int64 as _int64
from .tensor import tensor as _make_tensor


class _Unsupported(SyntaxError):
    """One unsupported construct inside an @tg.jit function."""


def _fail(node: ast.AST, message: str) -> None:
    raise _Unsupported(f"tg.jit: line {node.lineno}: {message}")


def _reads(stmts: list[ast.stmt]) -> set[str]:
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                names.add(node.id)

    for stmt in stmts:
        Visitor().visit(stmt)
    return names


def _writes(stmts: list[ast.stmt]) -> list[str]:
    """Assigned names in first-assignment order (nested blocks included)."""
    ordered: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store) and node.id not in ordered:
                ordered.append(node.id)
            self.generic_visit(node)

    for stmt in stmts:
        Visitor().visit(stmt)
    return ordered


def _control_escape(stmts: list[ast.stmt]) -> str | None:
    """Find break/continue/return, skipping nested function/class bodies."""

    class Visitor(ast.NodeVisitor):
        found: str | None = None

        def visit_Break(self, node: ast.Break) -> None:
            self.found = self.found or "break"

        def visit_Continue(self, node: ast.Continue) -> None:
            self.found = self.found or "continue"

        def visit_Return(self, node: ast.Return) -> None:
            self.found = self.found or "return"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            pass

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            pass

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            pass

        def visit_Lambda(self, node: ast.Lambda) -> None:
            pass

    visitor = Visitor()
    for stmt in stmts:
        visitor.visit(stmt)
    return visitor.found


def _index_state(start: int, like) -> object:
    """Rank-zero int64 loop-index state on the carried state's device."""
    if like is None:
        return _make_tensor(start, dtype=_int64)
    return _make_tensor(start, dtype=_int64, device=like.device)


class _LoopRewriter:
    """Rewrite one function body statement list, tracking defined names."""

    def __init__(self, max_iterations: int | None) -> None:
        self.max_iterations = max_iterations
        self.counter = 0

    def rewrite_function(self, definition: ast.FunctionDef) -> None:
        defined = {arg.arg for arg in (
            *definition.args.posonlyargs, *definition.args.args,
            *definition.args.kwonlyargs)}
        for arg in (definition.args.vararg, definition.args.kwarg):
            if arg is not None:
                defined.add(arg.arg)
        definition.body = self._block(definition.body, defined)

    def _block(self, stmts: list[ast.stmt], defined: set[str]) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        for index, stmt in enumerate(stmts):
            if isinstance(stmt, (ast.For, ast.While)):
                emitted, states = self._loop(stmt, defined, stmts[index + 1:])
                rewritten.extend(emitted)
                defined.update(states)
            else:
                self._nested(stmt, defined)
                rewritten.append(stmt)
                defined.update(_writes([stmt]))
        return rewritten

    def _nested(self, node: ast.AST, defined: set[str]) -> None:
        """Recurse into statement lists of if/with/try/match blocks."""
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(
                    isinstance(item, ast.stmt) for item in value):
                block_defined = set(defined)
                value[:] = self._block(value, block_defined)
                defined.update(block_defined)
            elif isinstance(value, ast.AST) and not isinstance(
                    value, (ast.expr, ast.Name)):
                self._nested(value, defined)

    def _loop(
        self,
        stmt: ast.For | ast.While,
        defined: set[str],
        rest: list[ast.stmt],
    ) -> tuple[list[ast.stmt], list[str]]:
        if stmt.orelse:
            _fail(stmt, "loop else clauses are not supported")
        is_while = isinstance(stmt, ast.While)
        if is_while:
            if self.max_iterations is None:
                _fail(stmt, "while loops require "
                      "@tg.jit(max_iterations=...) as a resource bound")
            if isinstance(stmt.test, ast.Constant) and stmt.test.value is True:
                _fail(stmt, "while True is not supported; put the exit "
                      "condition in the while test")
            iterations_src: str | None = None
        else:
            if not isinstance(stmt.target, ast.Name):
                _fail(stmt, "for targets must be a single name")
            iterator = stmt.iter
            if not (isinstance(iterator, ast.Call)
                    and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == "range"
                    and not iterator.keywords
                    and 1 <= len(iterator.args) <= 3):
                _fail(stmt, "for loops must iterate range(...) with "
                      "Python-int bounds")
            iterations_src = ast.unparse(stmt.iter)

        body = self._block(stmt.body, set(defined))
        break_if: ast.If | None = None
        if not is_while and body and isinstance(body[0], ast.If):
            candidate = body[0]
            if (not candidate.orelse and len(candidate.body) == 1
                    and isinstance(candidate.body[0], ast.Break)):
                break_if = candidate
                body = body[1:]
        escape = _control_escape(body)
        if escape is not None:
            _fail(stmt, f"{escape} inside a staged loop is not supported; "
                  "only 'if <cond>: break' as the first loop statement is")

        assigned = _writes(body)
        carried = [name for name in assigned if name in defined]
        used_after = _reads(rest)
        late = [name for name in assigned
                if name not in defined and name in used_after]
        if late:
            _fail(stmt, f"loop-assigned variable(s) {', '.join(late)} must "
                  "be initialized before the loop to be loop-carried")

        target: str | None = None
        if not is_while:
            assert isinstance(stmt.target, ast.Name)
            candidate = stmt.target.id
            reads = _reads(body)
            if break_if is not None:
                reads |= _reads([break_if.test])
            if candidate != "_" and (candidate in reads
                                     or candidate in used_after):
                if candidate in assigned:
                    _fail(stmt, f"loop variable {candidate!r} must not be "
                          "reassigned inside the loop")
                target = candidate
        states = carried + ([target] if target is not None else [])
        if not states:
            _fail(stmt, "the loop carries no Tensor state; it would not "
                  "produce staged IR")

        self.counter += 1
        stamp = self.counter
        params = ", ".join(states)
        parts = list(carried)
        emit: list[ast.stmt] = []

        if target is not None:
            assert isinstance(stmt, ast.For)
            args = stmt.iter.args
            start_src = ast.unparse(args[0]) if len(args) > 1 else "0"
            like = carried[0] if carried else "None"
            emit.extend(self._parse(
                f"__gf_index_{stamp} = __gf_index_state({start_src}, {like})"))
            body = [*body, *self._parse(
                f"{target} = {target} + ({self._step_src(stmt)})")]
            parts.append(f"__gf_index_{stamp}")

        initials = parts[0] if len(parts) == 1 else f"({', '.join(parts)})"
        returns = states[0] if len(states) == 1 else f"({', '.join(states)})"
        body_src = textwrap.indent(
            "\n".join(ast.unparse(item) for item in body), "    ")
        emit.extend(self._parse(
            f"def __gf_body_{stamp}({params}):\n{body_src}\n"
            f"    return {returns}"))

        if break_if is not None:
            emit.extend(self._parse(
                f"def __gf_cond_{stamp}({params}):\n"
                f"    return {self._inverted(break_if.test)}"))
            emit.extend(self._call(
                states, initials, stamp, f"len({iterations_src})",
                while_=True))
        elif is_while:
            emit.extend(self._parse(
                f"def __gf_cond_{stamp}({params}):\n"
                f"    return {ast.unparse(stmt.test)}"))
            emit.extend(self._call(
                states, initials, stamp, repr(self.max_iterations),
                while_=True))
        else:
            emit.extend(self._call(
                states, initials, stamp, f"len({iterations_src})",
                while_=False))
        return emit, states

    def _call(
        self,
        states: list[str],
        initials: str,
        stamp: int,
        budget: str,
        *,
        while_: bool,
    ) -> list[ast.stmt]:
        targets = ", ".join(states)
        if while_:
            source = (f"{targets} = __gf_while({initials}, "
                      f"__gf_cond_{stamp}, __gf_body_{stamp}, "
                      f"max_iterations={budget})")
        else:
            source = (f"{targets} = __gf_repeat({initials}, "
                      f"__gf_body_{stamp}, iterations={budget})")
        return self._parse(source)

    @staticmethod
    def _step_src(stmt: ast.For) -> str:
        args = stmt.iter.args
        return ast.unparse(args[2]) if len(args) > 2 else "1"

    @staticmethod
    def _inverted(test: ast.expr) -> str:
        spelling = {
            ast.Lt: ">=", ast.LtE: ">", ast.Gt: "<=", ast.GtE: "<",
            ast.Eq: "!=", ast.NotEq: "==",
        }
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and type(test.ops[0]) in spelling):
            op = spelling[type(test.ops[0])]
            return (f"({ast.unparse(test.left)}) {op} "
                    f"({ast.unparse(test.comparators[0])})")
        return f"({ast.unparse(test)}) != True"

    @staticmethod
    def _parse(source: str) -> list[ast.stmt]:
        return ast.parse(textwrap.dedent(source)).body


def jit(function=None, *, max_iterations: int | None = None):
    """Capture Python loops in the decorated function as gf_control ops.

    Usable as ``@tg.jit`` (only ``for i in range(k)`` loops) or as
    ``@tg.jit(max_iterations=k)`` (also bare ``while cond:`` loops, where the
    decorator argument is the mandatory per-loop resource bound). See the
    module docstring for the supported subset.
    """
    if function is None:
        return lambda target: jit(target, max_iterations=max_iterations)
    if max_iterations is not None and (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool) or max_iterations < 0):
        raise TypeError("tg.jit max_iterations must be a non-negative integer")
    # @tg.jit may sit outside another capture decorator (e.g. @tg.program);
    # unwrap to the source-owning function so globals and source line up.
    function = inspect.unwrap(function)
    if not inspect.isfunction(function):
        raise TypeError("tg.jit expects a plain Python function")
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as error:
        raise TypeError(
            "tg.jit needs source access; lambdas, REPL and exec-defined "
            "functions are not supported") from error
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TypeError("tg.jit expects a single function definition")
    definition = tree.body[0]
    definition.decorator_list = []
    _LoopRewriter(max_iterations).rewrite_function(definition)
    tree.body.insert(0, ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    env = dict(function.__globals__)
    # Closure variables are snapshotted by value into the staged globals.
    closure = function.__closure__ or ()
    for name, cell in zip(function.__code__.co_freevars, closure, strict=True):
        env[name] = cell.cell_contents
    env.update({
        "__gf_repeat": _control.repeat,
        "__gf_while": _control.while_loop,
        "__gf_index_state": _index_state,
    })
    filename = inspect.getsourcefile(function) or "<tg.jit>"
    exec(compile(tree, filename, "exec"), env)  # noqa: S102 - staging by design
    transformed = env[definition.name]
    # @tg.jit is also the composition boundary: kernel calls in the
    # straight-line body capture into an automatic GraphProgram (fusion),
    # exactly as an explicit @tg.program would. With no kernel calls the
    # boundary is a no-op and behavior is unchanged.
    from .program import capture_boundary

    return wraps(function)(capture_boundary(transformed))


__all__ = ["jit"]

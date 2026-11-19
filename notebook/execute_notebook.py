"""Executes Multi_Disease_ML_Framework.ipynb cell by cell in a plain Python
process (no Jupyter kernel / zmq involved) and writes the real stdout and
trailing-expression outputs back into the notebook file, so it opens already
showing genuine results.
"""
import ast
import contextlib
import io
import os

import nbformat

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Multi_Disease_ML_Framework.ipynb")

nb = nbformat.read(NB_PATH, as_version=4)
namespace = {}
exec_count = 0

for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    exec_count += 1
    src = cell.source
    tree = ast.parse(src)

    trailing_expr = None
    body = tree.body
    if body and isinstance(body[-1], ast.Expr):
        trailing_expr = body[-1]
        body = body[:-1]

    stdout_buf = io.StringIO()
    outputs = []
    error_output = None

    try:
        with contextlib.redirect_stdout(stdout_buf):
            if body:
                exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), namespace)
            last_value = None
            if trailing_expr is not None:
                last_value = eval(
                    compile(ast.Expression(body=trailing_expr.value), "<cell>", "eval"), namespace
                )
    except Exception as e:
        error_output = e
        last_value = None

    stdout_text = stdout_buf.getvalue()
    if stdout_text:
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stdout_text))

    if error_output is not None:
        outputs.append(
            nbformat.v4.new_output(
                "error",
                ename=type(error_output).__name__,
                evalue=str(error_output),
                traceback=[str(error_output)],
            )
        )
        print(f"[cell {exec_count}] ERROR: {error_output}")
        cell["outputs"] = outputs
        cell["execution_count"] = exec_count
        nbformat.write(nb, NB_PATH)
        raise SystemExit(f"Stopped at cell {exec_count} due to error above. Notebook saved with partial outputs.")

    if trailing_expr is not None and last_value is not None:
        data = {"text/plain": repr(last_value)}
        if hasattr(last_value, "to_html"):
            try:
                data["text/html"] = last_value.to_html()
            except Exception:
                pass
        outputs.append(
            nbformat.v4.new_output("execute_result", data=data, execution_count=exec_count)
        )

    cell["outputs"] = outputs
    cell["execution_count"] = exec_count
    print(f"[cell {exec_count}] OK")

nbformat.write(nb, NB_PATH)
print("notebook executed and saved:", NB_PATH)

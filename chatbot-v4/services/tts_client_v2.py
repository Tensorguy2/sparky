"""Auto-restored from compiled bytecode (source file was missing)."""
from pathlib import Path as _Path
import marshal as _marshal

_pyc = _Path(__file__).resolve().parent / "__pycache__" / '_tts_client_v2_bytecode.pyc'
with _pyc.open("rb") as _f:
    _f.read(16)
    _code = _marshal.loads(_f.read())
del _Path, _marshal, _f, _pyc
exec(_code, globals())

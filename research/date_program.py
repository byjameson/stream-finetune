"""The 92-byte student: a program, not a neural net, for date normalization.

The task is deterministic; a date parser is the right representation. This script measures
the exact match and the bytes, with and without the shared dateutil library.
"""
import json
import os

import dateutil
from dateutil.parser import parse

CODE = (
    "from dateutil.parser import parse\n"
    "\n"
    "def parse_date(s):\n"
    "    return parse(s).date().isoformat()\n"
)

D = json.load(open("research/results/date_data.json", encoding="utf-8"))
te = D["test"]
ok = 0
bad = []
for r in te:
    out = parse(r["text"]).date().isoformat()
    if out == r["iso"]:
        ok += 1
    elif len(bad) < 10:
        bad.append((r["text"], out, r["iso"]))

lib_dir = os.path.dirname(dateutil.__file__)
lib_bytes = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(lib_dir) for f in files)
code_bytes = len(CODE.encode("utf-8"))
teacher_params = 1.5e9
result = {
    "exact": round(ok / len(te), 4),
    "n_test": len(te),
    "errors": bad,
    "code_bytes": code_bytes,
    "library_bytes": lib_bytes,
    "bits_per_teacher_param_code_only": round(code_bytes * 8 / teacher_params, 12),
    "bits_per_teacher_param_with_library": round((code_bytes + lib_bytes) * 8 / teacher_params, 8),
}
print(json.dumps(result, indent=1))
json.dump(result, open("research/results/date_program.json", "w"), indent=1)

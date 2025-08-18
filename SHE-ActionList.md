

# SHE-ActionList (Run-Readiness — Final Pass)

**Scope:** only what blocks `pip install -r requirements.txt` + import + running the demo. File quality/style is out of scope.

---

## ✅ What’s already correct
- Package layout: all subpackages under `src/` include `__init__.py`.
- `requirements.txt` has the core stack: `toponetx`, `torch-geometric`, `gudhi`, etc.
- `examples/SHEDemo.py` includes the `sys.path` patch and no longer tries to `exec()` a non-existent file.
- `stanley_reisner.py` is importable (no hyphen).

---

## 🔴 Critical blockers to fix

### A) Wrong absolute import (must be relative)
**File:** `src/core/SHESimplicialConvolution.py`  
**Fix:**
```python
# BEFORE (bad)
from she_core import SHESimplicialComplex, SHEConfig

# AFTER (good)
from .complex.SHESimplicialComplex import SHESimplicialComplex
from .config.SheConfig import SHEConfig
```

### B) Missing imports/definitions inside modules
Add the **exact headers** below so modules import cleanly at runtime.

**1) `src/core/diffusion/SHEHodgeDiffusion.py`** — add at the very top:
```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve
from scipy.linalg import eigh
from ..config.SheConfig import SHEConfig
from ..complex.SHESimplicialComplex import SHESimplicialComplex
from ..SHE import DiffusionResult  # safe unless SHE.py imports this module

logger = logging.getLogger(__name__)
```

**2) `src/core/complex/SHESimplicialComplex.py`** — add at the very top:
```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from scipy.sparse import csr_matrix
from toponetx.classes.simplicial_complex import SimplicialComplex
from ..config.SheConfig import SHEConfig

logger = logging.getLogger(__name__)
try:
    import toponetx  # noqa: F401
    TOPOX_AVAILABLE = True
except Exception:
    TOPOX_AVAILABLE = False
```

> If you later import `SHEHodgeDiffusion` from `SHE.py`, consider guarding the `DiffusionResult` import with `TYPE_CHECKING` or doing a local import inside `analyze_diffusion` to avoid circulars.

---

## 🟡 Alias consistency (prevent silent NameErrors)
Ensure files that use these aliases also import them:

- `numpy as np`
- `torch.nn as nn`
- `seaborn as sns`

**Likely touches:**  
`src/core/visualize/SHEDiffusionVisualizer.py` (`sns`),  
`src/morse/morse.py`, `src/morse/morsescalability.py`, `src/stanleyreisner/stanley_reisner.py` (`np`, possibly `nn`).

---

## 🟠 Optional dependencies actually used in code
These are referenced but **not** in core `requirements.txt`. Decide whether to add, or document as extras in README.

- `pytorch-lightning` → used in `src/core/SHESimplicialConvolution.py` (training)
- `sparse` (pydata/sparse) → used in `core/SHE.py`, `core/diffusion/SHEHodgeDiffusion.py`, `morse/morsescalability.py`, `stanley_reisner.py`
- `numba`, `psutil` → used in `morse/morsescalability.py`, `stanley_reisner.py`

**Recommended README note (keep core lean):**
```bash
# training
pip install pytorch-lightning
# morse scalability
pip install numba psutil sparse
```

---

## 🧪 Smoke test (post-fix)
Create `quick_smoke.py` at repo root and run it:

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from core.config.SheConfig import SHEConfig
from core.complex.SHESimplicialComplex import SHESimplicialComplex
from core.engine.SHEEngine import SHEEngine

cfg = SHEConfig(device="cpu", max_dimension=2)
she = SHEEngine(cfg)
sc = SHESimplicialComplex(name="toy", config=cfg)
print("OK:", isinstance(sc, SHESimplicialComplex))
```

Expected output:
```
OK: True
```

---

## 📋 Quick checklist
- [ ] Change `she_core` import to **relative** imports in `SHESimplicialConvolution.py`.
- [ ] Add the header imports to `SHEHodgeDiffusion.py`.
- [ ] Add the header imports + `TOPOX_AVAILABLE` to `SHESimplicialComplex.py`.
- [ ] Standardize `np`, `nn`, `sns` aliases where used.
- [ ] Decide **extras vs core** for `pytorch-lightning`, `sparse`, `numba`, `psutil` and document in README.
- [ ] Run `quick_smoke.py`.
- [ ] Run the demo in `examples/SHEDemo.py`.

---

### Notes
- If you later introduce a `pyproject.toml`, set `package_dir={"": "src"}` and include the `core`, `morse`, and `stanleyreisner` packages.
- Consider returning plain dicts instead of `DiffusionResult` in `SHEHodgeDiffusion` if you want to completely avoid circular imports.

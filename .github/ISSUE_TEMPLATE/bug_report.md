---
name: Bug report
about: Report incorrect numerics, a crash, or unexpected behaviour
labels: bug
---

**What happened**

**What you expected**

**Minimal reproducer**

```python
import qbm
# ...
```

**Numerical regime** (numerical bugs here are often regime-specific)
- system size (qubits):
- backend (`dense` / `statevector` / `jax`):
- parameter scale / temperature:
- model, loss, optimizer:

**Environment**

```
python -c "import qbm, numpy, scipy; print(qbm.__version__, numpy.__version__, scipy.__version__, qbm.available_backends())"
```

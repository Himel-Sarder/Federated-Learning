<img width="1200" height="628" alt="image" src="https://github.com/user-attachments/assets/a6f08f13-6aac-4484-9e83-a7d5e43f9405" />


# 1. Install Flower

Run this in your terminal/PowerShell:

```bash
pip install flwr
```

If you also need PyTorch and scikit-learn (for your FL project):

```bash
pip install flwr torch torchvision scikit-learn
```

---

## 2. Verify Installation

Open Python and check:

```bash
python
```

Then inside Python:

```python
import flwr
print(flwr.__version__)
```

If no error appears and a version prints, Flower is installed correctly.

---

## 3. Common Installation Issues (Windows Fixes)

### Issue: pip not recognized

```bash
python -m pip install flwr
```

### Issue: Permission error

```bash
pip install flwr --user
```

### Issue: Multiple Python versions

```bash
where python
where pip
```

Make sure both point to the same Python installation.

---

## 4. Recommended: Use Virtual Environment

```bash
python -m venv fl_env
fl_env\Scripts\activate
pip install flwr torch scikit-learn
```

This avoids dependency conflicts.

---

## 5. Quick Test (Hello World)

Create `test_flwr.py`:

```python
import flwr as fl
print("Flower imported successfully")
```

Run:

```bash
python test_flwr.py
```

If it prints without error, setup is complete.

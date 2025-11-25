# 🚨 URGENT: How to Fix NaN in Colab

## The Problem

Your notebook is **loading OLD code from GitHub**, not the new fixes!

When you run the cell with `git clone`, it fetches the code from the repository.
But if you pushed to a feature branch (`claude/ct-scan-reconstruction-017qPRMrPL5j6NqYWFE7uQyG`),
the notebook might be using the **main branch** which doesn't have the fixes yet!

## The Solution - 3 Options

### Option 1: Pull the Correct Branch in Colab (RECOMMENDED)

**Add this cell RIGHT AFTER the git clone cell:**

```python
# CRITICAL: Pull the branch with NaN fixes
import os
os.chdir('/content/PINN_Dadiff')
!git fetch origin
!git checkout claude/ct-scan-reconstruction-017qPRMrPL5j6NqYWFE7uQyG
!git pull origin claude/ct-scan-reconstruction-017qPRMrPL5j6NqYWFE7uQyG
!git log --oneline -1  # Should show "Fix ADRN diffusion stability..."

# Verify the fix is loaded
print("\n🔍 Verifying ADRN beta values...")
with open('ct_reconstruction/src/adrn.py', 'r') as f:
    content = f.read()
    if 'beta_max: float = 0.02' in content:
        print("✅ Correct beta values found!")
    elif 'beta_max: float = 20.0' in content:
        print("❌ OLD CODE STILL LOADED! Beta values are wrong!")
    else:
        print("⚠️  Could not verify beta values")
```

### Option 2: Merge to Main Branch First

```bash
# On your local machine or in Colab terminal:
git checkout main
git merge claude/ct-scan-reconstruction-017qPRMrPL5j6NqYWFE7uQyG
git push origin main
```

Then restart your Colab runtime and re-run.

### Option 3: Direct Code Patch in Notebook

**Add this cell IMMEDIATELY after imports:**

```python
# EMERGENCY PATCH: Force correct ADRN beta values
import ct_reconstruction.src.adrn as adrn_module

# Monkey-patch the CT_ADRN __init__ to use correct values
original_init = adrn_module.CT_ADRN.__init__

def patched_init(self, *args, **kwargs):
    # Force correct beta values
    kwargs['beta_min'] = 0.0001
    kwargs['beta_max'] = 0.02
    original_init(self, *args, **kwargs)
    print("✅ ADRN patched with correct beta values")

adrn_module.CT_ADRN.__init__ = patched_init
```

## How to Verify the Fix

**Add this verification cell before training:**

```python
# Verify ADRN has correct beta values
model_test = create_model(config).to(device)
beta_max = model_test.adrn.betas.max().item()
beta_min = model_test.adrn.betas.min().item()

print(f"ADRN beta_min: {beta_min:.6f}")
print(f"ADRN beta_max: {beta_max:.6f}")

if beta_max > 0.1:
    print("❌ ERROR: beta_max is {beta_max:.3f} - OLD CODE IS BEING USED!")
    print("The notebook is loading code from the wrong branch!")
    print("Follow Option 1 above to fix this.")
else:
    print("✅ Correct beta values loaded!")
```

## Emergency Debug Scripts

If the above doesn't help, run these in Colab:

```python
# Upload emergency_nan_debug.py to Colab, then:
!python emergency_nan_debug.py

# If that passes, run:
!python emergency_loss_debug.py
```

These will pinpoint EXACTLY where NaN appears.

## Ultra-Conservative Training Config

Use this config to start:

```python
config.update({
    'learning_rate': 5e-5,          # Even smaller
    'batch_size': 1,                # Single sample
    'num_diffusion_steps': 1,       # Minimal diffusion
    'gamma': 0.0,                   # NO physics loss
    'lambda_phys_lpce': 0.0,        # NO physics
    'lambda_phys_pace': 0.0,
    'alpha': 1.0,                   # Only pixel loss
    'beta': 0.0,
    'use_amp': False,               # Disable mixed precision
})
```

If THIS works without NaN, gradually increase each parameter one at a time.

## The Root Cause

The NaN comes from `beta_max = 20.0` in ADRN. If you're still getting NaN,
it means the old code with `beta_max = 20.0` is still being loaded.

**Check manually:**
```python
!grep "beta_max" /content/PINN_Dadiff/ct_reconstruction/src/adrn.py
```

Should show: `beta_max: float = 0.02`
If it shows `beta_max: float = 20.0`, the fix didn't load!

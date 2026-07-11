#!/usr/bin/env python3
"""Thin entry point: run the sparsified-CMG validation.  `python3 run.py`.
Requires an editable pycmg on sys.path:  pip install -e /path/to/CMG-python."""
import validate

if __name__ == "__main__":
    validate.test_stall_resume()
    validate.test_spanner_essential()
    validate.test_end_to_end()
    validate.test_correctness()
    validate.test_baswana_sen()
    print("\nAll sparsified-CMG validation checks passed.")

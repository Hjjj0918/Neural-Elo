"""
=============================================================================
 Phase 1: CS:GO EDA -- Entry Point
=============================================================================
 This is a thin wrapper that delegates all work to the modular phase1/ package.

 Usage:
     python phase1_eda.py          # quick entry
     python -m phase1.run_all      # module entry (equivalent)
=============================================================================
"""

from phase1.run_all import main

if __name__ == "__main__":
    main()

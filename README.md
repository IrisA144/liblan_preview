# liblan_preview

PySCF extension for optical/magnetic properties of lanthanide-based materials

## Install

* Simply add `./liblan` to your `PYTHONPATH`. 

* Prerequisites
  * PySCF 1.7 or higher, and all dependencies
  * sympy 1.7 or higher (will be removed in future version)

## Features

* One-shot DMET for molecular systems
* DMET impurity solvers: HF, (SA-)CASSCF
* All electron/DMET-based CASSI-SO
* Regularized DIIS

## QuickStart

* You may find examples for running CASSI-SO calculations in `/liblan_preview/examples`. Specifically, the benchmarks (on 3d-SIMs) could be found in `liblan_preview/benchmarks`. 
* To carry out a DMET+CASSI-SO calculation you have to run the script `run_cassiso.py` twice:
  * Run `run_cassiso.py`
  * Look for the `$title_imp_rohf_orbs.molden` file, from which you should look for suitable CAS orbitals and write them down in `$title_cas_info` 
  * Run `run_cassiso.py` again

## References

* TBD

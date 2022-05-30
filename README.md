# porepy_hazmath
This repository contains runscripts and other functionality needed to develop efficient linear solvers for mixed-dimensional problems. The library combines discretizations obtained by the open-source package [PorePy](https://github.com/pmgbergen/porepy) with solvers implemented in [HAZMATH](https://github.com/HAZmathTeam/hazmath).

# How to use
To experiment with solvers for mixed-dimensional scalar problems, first install PorePy and HAZMATH (see the respective repositories for instructions). Then clone this repository, and start exploring the script [runscript_md_flow](https://github.com/keileg/porepy_hazmath/blob/main/scalar_elliptic/runscript_md_flow.py) or [runscript_md_flow_loop](https://github.com/keileg/porepy_hazmath/blob/main/scalar_elliptic/runscript_md_flow_loop.py) (the latter is used to study solver performance while varying permeability regimes).

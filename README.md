# **<center>PROJECT MASTER IPSA</center>**
## **<center>Modeling the transport of a chemical species in a fluid</center>**


### Description
This project aims to model the transport of a chemical species in a fluid, in 1D and 2D, 
using numerical schemes to approximate the real solution.

For this work, we consider the transport equation:
<center>\[ \frac{\partial u}{\partial t} + c \cdot \nabla \u = f \]</center>

**The Cauchy-Schwarz Inequality**
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

### Contents
This project is divided in six files

- Parameters.py contains the parameters of the simulation
- Conditions.py contains the conditions of the system
- Analytical.py compute the analytical solution of the equation
- Numerical.py compute an approximation of the solution
- Display.py plots graphs and animations of the solutions found
- Main.py calls functions from the previous files



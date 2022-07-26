# Project of Theoretical and Numerical Nuclear Physics: 

In this project, we develop a code to calculate the critical length $L_{crit}$, in order to study an unbounded nuclear diffusion in a fissile material.
The result is a runaway nuclear reaction that can lead to an intense explosion. For simplicity we based our calculations upon simplified Dirichlet 
boundary conditions, i.e. no neutron escape.

The following code is written in JULIA. 


### Project_1D_ODEs.jl
In this code it is possible to see how we develop the study of $L_{crit}$ solving two ODEs insted of the PDE. 

Starting from the PDE:

$\frac{\partial n}{\partial t} =  \mu \frac{\partial^2 n}{\partial x^2} + \eta n$

we can postulate a solution of the form:

$n(t,x)= T(t) X(x)$.

In this way we have to solve two ODEs:

$\frac{dT}{dt} = (\eta-\alpha)T \qquad \frac{d^2 X}{dx^2} = -\frac{\alpha}{\mu} X$

and the solutions are:

$T = A e^{(\eta-\alpha)t}, \qquad X = B\sin(x\sqrt{\alpha/\mu})\quad$ with $\quad\alpha = \mu \big(p \frac{\pi}{L}\big)^2$,

having imposed the boundary conditions at $x=0,L$.

### Project_1D_PDE.jl
In the code it is possible to see how we develop the study of $L_{crit}$ solving directly the PDE. 

In this code we discretize the spatial components and differential operator as:

$X(x) \longrightarrow X_{i} \quad i=1,...,n_{X}, \qquad  \frac{d^2}{dx^2}\longrightarrow \Delta$

$\mu \frac{d^2X}{dx^2} \longrightarrow \mu {\sum_{j=1}} \Delta_{ij} X_{j}$,

allowing us to rewrite the PDE as

$\frac{d n\_i(t)}{dt} = \mu {\sum_{j=1}} \Delta_{ij} n_{j}(t)+\eta\\, n_i(t)$.

### Project_3D.jl
In the code it is possible to see how we develop the study of $L_{crit}$ in three dimension in Cartesian coordinates.

We just adapt the the I method, with the calculation of the ODE, generalizing it to 3D. Starting from the PDE:

$\frac{\partial n}{\partial t} =  \mu \nabla^2 n + \eta n$

we can postulate a solution of the form:

$n(t,x,y,z)= T(t) X(x) Y(y) Z(z)$.

In this way we have to solve four ODEs:

$\frac{dT}{dt} = (\eta-\alpha)T$,

$\frac{d^2 X}{dx^2} = -\frac{\alpha_x}{\mu} X$,

$\frac{d^2 Y}{dy^2} = -\frac{\alpha_y}{\mu} Y$,

$\frac{d^2 Z}{dz^2} = -\frac{\alpha_z}{\mu} Z$,

with $\alpha = \alpha_x+\alpha_y+\alpha_z$.

Moreover, since the $3D$ case is physical, we can compute the critical mass of Nuclear weapons:

$m= \rho V$, where $V=L^3_{crit}$ and $\rho$ is the mass density of the fissile material.

###JULIA libraries
Some libraries are required to run the code correctly:
* Plots
*DifferentialEquations:ODEProblem,solve
*ForwardDiff:derivative
*DiffEqOperators: CentereDifference, Dirichlet0BC
*NumericalIntegration: integrate
*Einsum 
*LinearAlgebra:eigen
*Statistics: mean


## Bibliography

Graham Griffiths. Neutron diffusion. 02 2018. URL: (https://www.researchgate.net/publication/323035158_Neutron_diffusion).

using Plots
using DifferentialEquations: ODEProblem, solve #to define and solve the ODE
using ForwardDiff: derivative #to derive
#to create the matrix and to impose the boundary conditions
using DiffEqOperators: CenteredDifference,Dirichlet0BC, DerivativeOperator,RobinBC
using Statistics: mean #to evaluate the mean

# Now we want to solve the PDE

#constants

#diffusion constant
const μ_U = 2.3446e5 #m^2/s 
const μ_P = 2.6786e5 #m^2/s 

#neutron rate of formation 
const η_U = 1.8958e8 #s^-1 
const η_P = 3.0055e8 #s^-1 

#definition of the PDE 
"""
Returns time derivative of the PDE

`d(uᵢ)/dt =∑ⱼ μ Δᵢⱼ uⱼ+ η uⱼ `, 

where we have discretized the spatial part `n(x,t)->uᵢ(t)` and 

`∂²/∂x²-> Δᵢⱼ`, satisfying zero Dirichlet boundary conditions

# Arguments
- `u::Vector{Float64}`: variable of the ODE 
- `p`: parameters `(μ::Float64, η::Float64, Δ::DerivativeOperator, bc::RobinBC)` of the ODE, where 
`η` is the neutron rate of formation [1/s], and `μ` is the diffusion constant [m²/s],
Δ is the discretized differential operator and bc are the Dirichlet boundary conditions
- `t::Float64`: time at which we compute the derivative [s]

"""
function diffusionPDE(u::Vector{Float64},p, t::Float64)
    
    μ, η, Δ, bc =p  #parameters
    μ::Float64,
    η::Float64, 
    bc::RobinBC, 
    Δ::DerivativeOperator
    
    #Δ is the matrix, bc are the boundary conditions
    
    return μ*Δ*bc*u+η*u 

end


#conditions for the T 
t_span = (0.0, 1e-7);#boundary of the time domain
t_step = 1e-9;#step of t range
t_range = t_span[1]:t_step:t_span[2];#discretized domain
nt=length(t_range);

# For the X 
nx_PDE = 100; #n. points 

""" 
Compute the discretization step

# Argument: 
- `L::Float64`: Length of the domain, Ω:(0,L) [m]

"""
Δx(L::Float64) = L/(nx_PDE+1);

"""
Return the discretized domain Ω:(0,L), ignoring the boundaries

# Argument 
- `L::Float64`: length of the domain, Ω:(0,L) [m]

"""
L_range(L::Float64) = Δx(L):Δx(L):L-Δx(L);  

#initial conditions:
f(x::Float64,L::Float64) = sin(x*π/L);#initial function

const ord_deriv, ord_approx= 2, 2;#order of derivative and approximation
bc = Dirichlet0BC(Float64);#boundary conditions

#differential operator 
""" 
Compute the Differential Operator `d²/dx²` with a given discretization of the domain.
 
Order of the derivative is 2.

Order of approximation is 2.

# Argument: 
- `L::Float64`: Length of the domain, Ω:(0,L) [m]

"""
Δ(L::Float64)= CenteredDifference(ord_deriv,ord_approx,Δx(L),nx_PDE);

#PDE
"""
Create the problem of the PDE: `d(uᵢ)/dt =∑ⱼ μ Δᵢⱼ uⱼ+ η uⱼ `
# Arguments
- `n0::Vector{Float64}`:initial conditions
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `bc::RobinBC`: boundaries conditions

"""
prob_PDE(
    n0::Vector{Float64},
    L::Float64,
    μ::Float64,
    η::Float64,
    bc::RobinBC
)= ODEProblem(diffusionPDE,n0,t_span,[μ, η, Δ(L), bc]);

"""
Returns the solution of the PDE: `d(uᵢ)/dt =∑ⱼ μ Δᵢⱼ uⱼ+ η uⱼ`.

# Arguments 
- `n0::Vector{Float64}`:initial conditions
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `bc::RobinBC`: boundaries conditions

"""
sol_PDE(
    n0::Vector{Float64},
    L::Float64,
    μ::Float64,
    η::Float64,
    bc::RobinBC
) = solve(prob_PDE(n0,L,μ,η,bc));



#function to find the L 
"""
Return the critical lenght of the neutron diffusion for a given set of parameters and accuracy.

Runs a while loop increasing the test value L_crit
and solving the PDE `d(uᵢ)/dt =∑ⱼ μ Δᵢⱼ uⱼ+ η uⱼ `.
Subsequently, computes the time derivative of the solution 
checking an unbounded increment, until the required accuracy.

# Arguments
- μ::Float64: diffusion constant [m²/s]
- η::Float64: neutron rate of formation [1/s]
- ϵ::Float64: target accuracy of the search
- L_in::Float64: starting point of the search [m]
- ΔL::Float64: starting step of the search [m] 
- t_range::StepRangeLen: domain to compute the time ODE [s]
"""
function find_L_crit(
    μ::Float64, 
    η::Float64, 
    ϵ::Float64, 
    L_in::Float64, 
    ΔL::Float64, 
    t_range::StepRangeLen
    ) 
    
    L_loop, L_crit = L_in,0. #initialization
    
    nt=length(t_range)#n. of time points
    weight = Array(range(0.9,1.1, nt)) #weight of the points

    #fixed space points for the mean
    points =[30, 50, 70];
    
    while ΔL ≥ ϵ
        
        L_crit = L_loop+ΔL #all the different L 
        L_range_loop = L_range(L_crit) #L range at L
        
        #PDE
        n0_loop = f.(L_range_loop, L_crit) #n at L 
        
        sol_PDE_loop = sol_PDE(n0_loop,L_crit,μ, η,bc) #solution of the PDE
        sol_PDE_x(t) = mean(sol_PDE_loop(t)[points]) 
    
        partial_sol_x(t) = derivative(sol_PDE_x,t)    
        derivative_check = partial_sol_x.(t_range)
        
        if (derivative_check'*weight)/sum( weight)>0
            ΔL = ΔL/10
        else
            L_loop = L_crit
        end

    end

    return L_crit

end

#######

# parameters for the search
ϵ = 1e-4; #accuracy
L_in = 0.0;
ΔL = 1e-2;#1cm
L_crit_U = find_L_crit(μ_U,η_U,ϵ,L_in,ΔL,t_range);
L_crit_P = find_L_crit(μ_P,η_P,ϵ,L_in,ΔL,t_range);

println("The value of L, for the ²³⁵U, is $(L_crit_U*10^2)cm")
println("The value of L, for the ²³⁹Pu, is $(L_crit_P*10^2)cm")

#initial function
g_U(x)= sin(x*π/L_crit_U);

#time domain
t_range_plot = 0:t_step:1e-7;
n_t=length(t_range_plot);

L_plot_U =L_range(L_crit_U);
n0_plot_U = g_U.(L_plot_U);
sol_plot_U =sol_PDE(n0_plot_U,L_crit_U,μ_U,η_U,bc).(t_range_plot);

nMatrix_U = reduce(hcat,sol_plot_U)';

#we need the boundaries back 
BC1 = zeros(Float64,n_t);
newMatrix_U = hcat(BC1,nMatrix_U,BC1);
newL_range_U = [0.0;L_plot_U;L_crit_U];

#plot for ²³⁵U
Plot_U = plot(
    newL_range_U, t_range*10^6, newMatrix_U, st=:surface, 
    xlabel="x(m)", ylabel="t(μs)", zlabel="n(x,t)",
    title ="Neutron Diffusion in 1D (²³⁵U)\nL=$(L_crit_U*10^2)cm", 
    camera=(25,14),dpi=1000
)
savefig(Plot_U, "NeutronDiffusion1D_IIMethod_U")

####for ²³⁹Pu
g_P(x)= sin(x*π/L_crit_P); #initial function

L_plot_P =L_range(L_crit_P); #L range calculated at a given L
n0_plot = g_P.(L_plot_P); 

#solution of the PDE
sol_plot_P =sol_PDE(n0_plot,L_crit_P,μ_P,η_P,bc).(t_range_plot);
#transform the solution into a matrix
nMatrix_P=reduce(hcat,sol_plot_P)';

#we need the boundaries back 
newMatrix_P = hcat(BC1,nMatrix_P,BC1);
newL_range_P = [0.0;L_plot_P;L_crit_P];

Plot_P = plot(
    newL_range_P, t_range*10^6, newMatrix_P, st=:surface, 
    xlabel="x(m)", ylabel="t(μs)", zlabel="n(x,t)",
    title ="Neutron Diffusion in 1D (²³⁹Pu)\nL=$(round(L_crit_P*10^2,digits =3))cm", 
    camera=(25,14),dpi=1000
)
savefig(Plot_P, "NeutronDiffusion1D_Pu-239_IIMethod.pdf")


##At a different L for ²³⁵U
L_sovra = 0.1270;#m

g_U_sovra(x)= sin(x*π/L_sovra);

L_plot_U_sovra = L_range(L_sovra);
n0_plot_U_sovra = g_U_sovra.(L_plot_U_sovra);
sol_plot_U_sovra =sol_PDE(n0_plot_U_sovra,L_sovra,μ_U,η_U,bc).(t_range_plot);
nMatrix_U_sovra = reduce(hcat,sol_plot_U_sovra)';

#we need the boundaries back
newMatrix_U_sovra= hcat(BC1,nMatrix_U_sovra,BC1);
newL_range_U_sovra = [0.0;L_plot_U_sovra;L_sovra];

#plot for ²³⁵U
Plot_U_sovra = plot(
    newL_range_U_sovra, t_range*10^6, newMatrix_U_sovra, st=:surface, 
    xlabel="x(m)", ylabel="t(μs)", zlabel="n(x,t)",
    title ="Neutron Diffusion in 1D (²³⁵U)\nL=$(L_sovra*10^2)cm", 
    camera=(25,14),dpi=1000
)
savefig(Plot_U_sovra, "NeutronDiffusion1D_U-235-L_Sovra_IIMethod.pdf")

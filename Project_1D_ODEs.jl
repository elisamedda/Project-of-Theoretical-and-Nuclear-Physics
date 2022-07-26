using Plots
using DifferentialEquations: ODEProblem, solve #to define and solve the ODE
using ForwardDiff: derivative #to derive
#to create the matrix and to impose the boundary conditions
using DiffEqOperators: CenteredDifference,Dirichlet0BC 
using LinearAlgebra: eigen #to evaluate the eigenvalues and eigenvectors 
using NumericalIntegration: integrate #to integrate


#diffusion constant
const μ_U = 2.3446e5 #m^2/s 
const μ_P = 2.6786e5  #m^2/s 

#neutron rate of formation 
const η_U = 1.8958e8 #s^-1 
const η_P = 3.0055e8 #s^-1 

#=
Now we will solve the PDE of n re-writting it in terms of two ODEs. 
Thus, in the end we will solve two ODEs instead of a PDE. Then, using the 
superposition principle we can write n as an expansion of the modes found.
=#


#definition of the ODE for T 
"""
Returns time derivative of the ODE
``dT= (η-α)T``.

# Arguments
- `T::Float64`: variable of the ODE 
- `p::Vector{Float64}`: parameters `(η,α)` of the ODE, where 
`η` is the neutron rate of formation [1/s] and `α` [1/s] a parameter depending
on `L` the size of the spatial domain.
- `t::Float64`: time at which we compute the derivative [s]

"""
function diffusion_t(
    T::Float64, 
    p::Vector{Float64},
    t::Float64
    )
     
    η,α = p 
    
    return (η-α)*T

end

""" 
Compute the discretization step.

# Arguments: 
- `L::Float64`: Length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points

"""
Δx(L::Float64,nx::Int64)= L/(nx+1); #discretization step

const ord_deriv = 2; #order of derivative
const ord_approx = 2; #order of approximation

#= 
-μΔX = αX, where Δ is the discretization of the differential operator 
with boundary conditions (BC): 
=#
""" 
Compute the Differential Operator `-μ d²/dx²` with a given discretization of the domain, 
respecting Dirichlet boundary conditions equals to zero.

Order of the derivative is 2.

Order of approximation is 2.

# Arguments: 
- `L::Float64`: Length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]

"""
function Δ(
    L::Float64,
    nx::Int64,
    μ::Float64
    )
    return -μ*CenteredDifference(ord_deriv,ord_approx,Δx(L,nx),nx)*
        Dirichlet0BC(Float64)
end


"""
Return the eigenvalues and eigenvectors of the discretized differential operator
`-μ d²/dx²`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]

"""
function α_eigen(
    L::Float64,
    nx::Int64,
    μ::Float64
    )
    
    #=
    Matrix of Δ, where we focus on the interior points, since 
    the BC sets to 0 the extremal points
    =#
    
    Δ_matrix = reduce(hcat,Array(Δ(L,nx,μ)))[:,1:nx]
    
    return eigen(Δ_matrix)

end

#to extract the eigenvalues and eigenvectors
"""
Return the eigenvalues of the discretized differential operator
`-μ d²/dx²`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]

"""
α_eigenvalues(L::Float64,nx::Int64,μ::Float64)= α_eigen(L, nx, μ).values;

"""
Return the eigenvectors of the discretized differential operator
`-μ d²/dx²`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]

"""
α_eigenvectors(L::Float64,nx::Int64,μ::Float64)= α_eigen(L, nx, μ).vectors;

#q-th eigenvalue
"""
Return the q-th eigenvalues of the discretized differential operator
`-μ d²/dx²`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]
- `q::Int64`: index of the eigenvalue

"""
α(L::Float64,nx::Int64,μ::Float64,q::Int64)= α_eigenvalues(L, nx, μ)[q];

###discretization of the X domain
"""
Return the discretized domain Ω:(0,L), ignoring the boundaries.

# Arguments
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points

"""
L_range(L::Float64,nx::Int64) = Δx(L,nx):Δx(L,nx):L-Δx(L,nx);

###conditions for T

T0=1.0; #initial condition 
"""
Return the parameters of the time ODE: `dT/dt = (η-α)t`.

# Arguments
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `q::Int64`: index of the eigenvalue

"""
p_T(L::Float64,nx::Int64,μ::Float64,η::Float64,q::Int64) = [η, α(L, nx, μ, q)];#parameters
t_step = 1e-9; #step of t range
t_span = (0.0, 1e-7); #boundary of the time domain
#discretized domain:
t_range= t_span[1]:t_step:t_span[2]; 

#function for the ODE of T
"""
Create the problem of the time ODE: `dT/dt = (η-α)t`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `q::Int64`: index of the eigenvalue

"""
prob_T(
    L::Float64,
    nx::Int64,
    μ::Float64,
    η::Float64,
    q::Int64
) = ODEProblem(diffusion_t,T0,t_span,p_T(L,nx,μ,η,q));

"""
Returns the solution of the time ODE: `dT/dt = (η-α)t`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `q::Int64`: index of the eigenvalue

"""
sol_T(
    L::Float64,
    nx::Int64,
    μ::Float64,
    η::Float64,
    q::Int64
) = solve(prob_T(L, nx, μ,η, q));

#q-th eigenvector solution for X
"""
Returns the solution (eigenvector) of the space ODE: `-μ d²X/dx² = α X`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]
- `q::Int64`: index of the eigenvalue

"""
sol_X_eigen(
    L::Float64,
    nx::Int64,
    μ::Float64,
    q::Int64
) = α_eigenvectors(L,nx,μ)[:,q];

#function to find the L 
"""
Return the critical lenght of the neutron diffusion for a given set of parameters and accuracy.

Runs a `while` loop increasing the test value L_crit and solving the time ODE `dT/dt = (η-α)T`. 
Subsequently, computes the time derivative of the solution, checking an unbounded increment,
until the required accuracy is met.

# Arguments
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `ϵ::Float64`: target accuracy of the search
- `nX::Int64`: number of points of the discretization
- `L_in::Float64`: starting point of the search [m]
- `ΔL::Float64`: starting step of the search [m] 
- `q::Int64`: index of the q-th eigenvalue α
- `t_range`::StepRangeLen: domain to compute the time ODE [s]

"""
function find_L_critical(
    μ::Float64, 
    η::Float64, 
    ε::Float64, 
    nx::Int64, 
    L_in::Float64, 
    ΔL::Float64, 
    q::Int64, 
    t_range::StepRangeLen
    )
    
    L_loop,L_crit = L_in, 0. #initialization
    
    nt = length(t_range)#n. of time points
    weight = Array(range(0.9,1.1, nt)) #weight of the points
    
    #ask derivative>0, iterating until the required accuracy
    while ΔL ≥ ε
        
        L_crit = L_loop+ΔL #increasing L 
        
        sol_T_loop = sol_T(L_crit,nx,μ,η,q) #solve ODE
        
        #differentiate the solution
        partial_sol_T(t) = derivative(sol_T_loop, t)
        derivative_check = partial_sol_T.(t_range)
        
        #weighted mean 
        if (derivative_check'*weight)/sum(weight)>0 
            ΔL = ΔL/10  #L (over-)critical: finer step 
        else
            L_loop = L_crit #L sub-critical: new starting L  
        end
    
    end
    
    return L_crit

end

#################

###choice of parameters to find the critical L 
ε = 1e-4; #accuracy
L_in = 0.; #starting L 
ΔL = 10e-2; #starting step
q_choose = 1; #worst case eigenvalue
nx = 100; #n. of points for the discretization

#calculation of the critical L for ²³⁵U and ²³⁹Pu
L_crit_U = find_L_critical(μ_U, η_U, ε, nx, L_in, ΔL, q_choose, t_range); #m
L_crit_P = find_L_critical(μ_P, η_P, ε, nx, L_in, ΔL, q_choose, t_range); #m

println("The value of L, for the ²³⁵U, is $(L_crit_U*10^2)cm")
println("The value of L, for the ²³⁹Pu, is $(L_crit_P*10^2)cm")

#Now we make a plot to visualize the growth in T:
L_plot = [L_crit_U, 0.15, 0.20]; #different L 
sol_T_L(L) = sol_T(L, nx, μ_U,η_U, q_choose); #solution in function of L
partial_sol_T_L(L,t) = derivative(sol_T_L(L), t);#time derivative given L
#evaluate the partial derivative in a given range
partial_sol_T_points(L) = [(t_i*10^6,partial_sol_T_L(L,t_i)) for t_i ∈ t_range];

#function of the plot that depends on L 
partial_sol_plots(L) = scatter(
    partial_sol_T_points(L),
    legend=:topleft, 
    xlabel="t(μs)",
    ylabel="T(t)",
    xtickfont=font(9),
    ytickfont=font(9),
    title ="T ODE derivative ²³⁵U \nL =$(L*10^2)cm",titlefont=font(10),
    labels= nothing
);

#=
Now I plot all the derivatives of T at different L 
in order to show that we need to weight the final points
to better understand if the derivative is increasing or not.
=#
Plot_L1 = partial_sol_plots(L_crit_U);
Plot_L2 = partial_sol_plots(0.10);
Plot_L3 = partial_sol_plots(0.12);
Plot_final = plot(Plot_L1, Plot_L2,Plot_L3, layout = (2,2))
savefig(Plot_final, "Derivative_study.pdf")
#=
now we want the plot of the diffusion. Then, we will need to expand 
a function in terms of the solution eigenvectors
=#

#the function of the eigenvectors at a given time
"""
Returns the q-th eigenvector of the ODE: `-μ d²X/dx² = α_q X`,
with its time evolution, satisfying the time ODE: `d(T_q)/dt = (η-α_q)T_q`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s] 
- `q::Int64`: index of the eigenvalue

"""
function time_eigenvectors(
    L::Float64, 
    nx::Int64, 
    μ::Float64,
    η::Float64, 
    t::Float64
    )
    
    X= α_eigenvectors(L,nx,μ)
    
    for q ∈ 1:nx
        # n_q(x,t) = X_q(x)*T_q(t)
        X[:,q]=X[:,q]*sol_T(L,nx,μ,η,q)(t)
    end
    
    return X

end

#function for the normalization of the eigenvectors
"""
Returns the normalization of the function `F` in a given domain `1/N² = ∫dx |F(x)|²`.

If `x` is of type `StepRangeLen` it casts `Array` on it before computing the rest.

# Arguments 
- `x::Vector{Float64}`: domain of the function
- `F::Vector{Float64}`: discretized function

"""
normalization(x::Vector{Float64}, F::Vector{Float64})= 1/sqrt(integrate(x,F.*F));
normalization(x::StepRangeLen, F::Vector{Float64})= normalization(Array(x),F);

#function to calculate the coefficient of the expansion
"""
Compute the coefficient of the expansion in eigenvectors of the ODE: 

`-μ d²(X_q)/dx² = α_q X_q`, 

of the function `f(x)` in a given domain.

`f(x)` must satisfy the boundary conditions `f(0) = f(L) = 0.`

# Arguments 
- `f::Function`:function we are expanding
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points of the discretization
- `μ::Float64`: diffusion constant [m²/s]

"""
function series_coef(
    f::Function, 
    L::Float64,
    nx::Int64, 
    μ::Float64
    )
    
    X = α_eigenvectors(L,nx,μ)
    
    x = range(0.,L,nx)
    F = f.(x) #discretise the initial function
    
    a_vector = zeros(Float64,nx) #initialization
    
    for q ∈ 1:nx

        #q-th eigenvector
        X_q= X[:,q]

        #coeff. calculation
        a_vector[q]=normalization(x,X_q)^2*integrate(x,F.*X_q)
    
    end
    
    return a_vector 

end

#creation of the function for the series expansion
"""
Compute the expansion in eigenvectors of the ODE: 

`-μ d²(X_q)/dx² = α_q X_q`, 

with its time evolution `T`, satisfying the time ODE: 

`d(T_q)/dt = (η-α_q)T_q`,

of the function `f(x)` in a given domain as
`f(x,t)= ∑_q a_q X_q(x) T_q(t)`.

# Arguments 
- `f::Function`:function we are expanding
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points of the discretization
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s] 
- `t::Float64`: time at which we calculate the expantion [s]

"""
function series_exp(
    f::Function, 
    L::Float64,
    nx::Int64, 
    μ::Float64, 
    η::Float64, 
    t::Float64
    )

    #n = ∑ a_q*n_q(x,t)
    return time_eigenvectors(L,nx,μ,η,t)*series_coef(f,L,nx,μ)

end



###now we make the plot 

#initial function with the critical L that respects the BC 
f_initial(x::Float64) = sin(x*π/L_crit_U);

#series solution calculated at L critical
n(t::Float64)= series_exp(f_initial,L_crit_U,nx,μ_U,η_U,t);

t_range_plot= 0.0:1e-9:1e-7; #time range for the plot
n_t = length(t_range_plot); #number of time points

#create the matrix of n
nMatrix=reduce(hcat, n.(t_range_plot))';

#we need to add back the boundaries
BC1 = zeros(Float64,n_t);

#matrix with the zeros of the BC:
newMatrix = hcat(BC1,nMatrix,BC1);
newL_range = [0.0;L_range(L_crit_U,nx);L_crit_U];

#plot of the diffusion at exactly the critical L 
Plot = plot(
    newL_range, t_range_plot*10^6, newMatrix, st=:surface,
    xlabel="x(m)", ylabel="t(μs)", zlabel="n(x,t)",
    title="Neutron diffusion 1D (²³⁵U)\n at L=$(L_crit_U*10^2)cm",
    camera=(25,14),dpi=1000
)
savefig(Plot,"NeutronDiffusion1D_U-235.pdf")

###same study to make the plot in the case of ²³⁹Pu

#initial function 
f_initial2(x::Float64) = sin(x*π/L_crit_P);

n2(t::Float64)= series_exp(f_initial2,L_crit_P,nx,μ_P,η_P,t);
nMatrix=reduce(hcat, n2.(t_range_plot))';
newMatrix_P= hcat(BC1,nMatrix,BC1);

newL_range_P= [0.0;L_range(L_crit_P,nx);L_crit_P];

Plot2 = plot(
    newL_range_P,t_range_plot*10^6,newMatrix_P, st=:surface,
    xlabel="x(m)", ylabel="t(μs)", zlabel="n(x,t)",
    title="Neutron diffusion 1D (²³⁹Pu)\n at $(round(L_crit_P*10^2,digits=3))cm",
    camera=(25,14),dpi=1000
)
savefig(Plot2, "NeutronDiffusion1D_Pu-239.pdf")

### The same study but with a different L for ²³⁵U
L_sovra = 0.12;

f_initial3(x::Float64) = sin(x*π/L_sovra);

n3(t::Float64)= series_exp(f_initial3,L_sovra,nx,μ_P,η_P,t);

nMatrix3=reduce(hcat, n3.(t_range_plot))';

newMatrix_3= hcat(BC1,nMatrix3,BC1);

newL_range_3= [0.0;L_range(L_sovra,nx);L_sovra];

Plot3 = plot(
    newL_range_3,t_range_plot*10^6,newMatrix_3, st=:surface,
    xlabel="x(m)", ylabel="t(μs)", zlabel="n(x,t)",
    title="Neutron diffusion 1D (²³⁵U)\n at $(L_sovra*10^2)cm",
    camera=(25,14),dpi=1000
)
savefig(Plot3, "NeutronDiffusion1D_U-235-L_Sovra.pdf")

using Plots
using DifferentialEquations: ODEProblem, solve  #to define and solve the ODE
using ForwardDiff: derivative #to derive
#to create the matrix and to impose the boundary conditions
using DiffEqOperators: CenteredDifference, Dirichlet0BC
using NumericalIntegration: integrate #to integrate
using Einsum #for the Einstein sum
using LinearAlgebra: eigen #to evaluate the eigenvalues and eigenvectors 

#constants :

#diffusion constant
const μ_U = 2.3446e5 #m^2/s 
const μ_P = 2.6786e5 #m^2/s 

#neutron rate of formation 
const η_U = 1.8958e8 #s^-1 
const η_P = 3.0055e8 #s^-1 

#density for the calculation of the mass
const ρ_U = 18.71e3 #kg/m^3
const ρ_P = 15.60e3 #kg/m^3

#For the T
"""
Returns time derivative of the ODE
``dT= (η-α)T``

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
Δx(L::Float64,nx::Int64) = L/(nx+1); #discretization step

const ord_deriv= 2; #order of derivative
const ord_approx = 2; #order of approximation

# -μΔX = αX, where Δ is the discretization of the differential operator:
bc = Dirichlet0BC(Float64); #boundary conditions

#differential operators
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
Δ(L::Float64,nx::Int64,μ::Float64) = 
    -μ*CenteredDifference(ord_deriv,ord_approx, Δx(L,nx),nx)*bc;
#Function to calculate eigenvalues & eigenvectors
"""
Return the eigenvalues and eigenvectors of the discretize differential operator
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
    
    #Matrix of Δ, we ignore the 1st and last coloumns from BC 
    
    Δ_matrix = reduce(hcat,Array(Δ(L,nx,μ)))[:,1:nx] 
    
    return eigen(Δ_matrix)
end

#eigenvalues and eigenvectors
"""
Return the eigenvalues of the discretize differential operator
`-μ d²/dx²`.

# Arguments 
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]

"""
α_eigenvalues(L::Float64,nx::Int64,μ::Float64)= α_eigen(L, nx, μ).values;

"""
Return the eigenvectors of the discretize differential operator
`-μ d²/dx²`.

# Arguments
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]

"""
α_eigenvectors(L::Float64,nx::Int64,μ::Float64)= α_eigen(L, nx, μ).vectors;

#q-th eigenvalue
"""
Return the q-th eigenvalues of the discretize differential operator
`-μ d²/dx²`.

# Arguments
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points
- `μ::Float64`: diffusion constant [m²/s]
- `q::Int64`: index of the eigenvalue

"""
α(L::Float64,nx::Int64,μ::Float64,q::Int64)= α_eigenvalues(L, nx, μ)[q];

#the sum of the q-th eigenvalues in x,y,z 
"""
Compute the sum of the q₁-th,q₂-th,q₃-th eigenvalues, respectively
of the discretized differential operators

`-μ d²/dx²`, `-μ d²/dy²` and `-μ d²/dz²`, 

with domains `(0,Lx)`, `(0,Ly)` and `(0,Lz)`.

# Arguments 
- `L::Vector{Float64}`: Vector with the length of the domain, Ωᵢ:[0,Lᵢ] [m]
- `N::Vector{Int64}`: Vector with the numbers of points of the discretization
- `μ::Float64`: diffusion constant [m²/s]
- `Q::Vector{Int64}`:Vector of index of the eigenvalues

"""
function α_sum(
    L::Vector{Float64},
    N::Vector{Int64},
    μ::Float64,
    Q::Vector{Int64}
    )

    Lx, Ly, Lz = L
    nx, ny, nz = N
    qx, qy, qz = Q

   return α(Lx,nx,μ,qx)+α(Ly,ny,μ,qy)+α(Lz,nz,μ,qz)

end

"""
Return the discretized domain Ω:(0,L), ignoring the boundaries.

# Arguments
- `L::Float64`: length of the domain, Ω:(0,L) [m]
- `nx::Int64`: numbers of points

"""
L_range(L::Float64,nx::Int64) = Δx(L,nx):Δx(L,nx):L-Δx(L,nx);


### conditions for T
T0=1.0;#initial condition 

"""
Return the parameters of the time ODE: `dT/dt = (η-α)t`.

# Arguments
- `L::Vector{Float64}`: Vector with the length of the domain, Ωᵢ:(0,Lᵢ) [m]
- `N::Vector{Int64}`: Vector with the numbers of points of the discretization
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `Q::Vector{Int64}`:Vector of index of the eigenvalues

"""
p_T(
    L::Vector{Float64},
    N::Vector{Int64},
    μ::Float64,
    η::Float64,
    Q::Vector{Int64}
) = [η, α_sum(L, N, μ, Q)]; #parameters of the time ODE

t_step = 1e-9; #step of t range
t_span = (0.0, 1e-7); #boundary of the time domain
#discretized domain:
t_range = t_span[1]:t_step:t_span[2];

#function for the ODE of T
"""
Create the problem of the time ODE: `dT/dt = (η-α)t`.

# Arguments
- `L::Vector{Float64}`: Vector with the length of the domain, Ωᵢ:(0,Lᵢ) [m]
- `N::Vector{Int64}`: Vector with the numbers of points of the discretization
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `Q::Vector{Int64}`:Vector of index of the eigenvalues

"""
prob_T(
    L::Vector{Float64},
    N::Vector{Int64},
    μ::Float64,
    η::Float64,
    Q::Vector{Int64}
) = ODEProblem(diffusion_t, T0, t_span, p_T(L, N, μ, η, Q)); 

"""
Returns the solution of the time ODE: `dT/dt = (η-α)t`.

# Arguments
- `L::Vector{Float64}`: Vector with the length of the domain, Ωᵢ:(0,Lᵢ) [m]
- `N::Vector{Int64}`: Vector with the numbers of points of the discretization
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `Q::Vector{Int64}`:Vector of index of the eigenvalues

"""
sol_T(
    L::Vector{Float64},
    N::Vector{Int64},
    μ::Float64,
    η::Float64,
    Q::Vector{Int64}
) = solve(prob_T(L, N, μ, η, Q));

#q-th eigenvector solution for X
"""
Returns the q-th solution (eigenvector) of the spatial ODE: 
`-μ d²X/dx² = α X`.

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
) = α_eigenvectors(L, nx, μ)[:,q];


#function to find the critical L 
"""
Return the critical lenght of the neutron diffusion for a given set of parameters and accuracy.

Runs a `while` loop increasing the test value `L_crit` and solving the time ODE `dT/dt = (η-α)T`.
Subsequently, compute the time derivative of the solution checking an unbounded increment,
until the required accuracy.

# Arguments
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s]
- `ϵ::Float64`: target accuracy of the search
- `N::Vector{Int64}`: Vector with the numbers of points of the discretizations
- `L_in::Float64`: starting point of the search [m]
- `ΔL::Float64`: starting step of the search [m] 
- `Q::Vector{Int64}`:Vector of indeces of the q-th eigenvalues α
- `t_range`::StepRangeLen: domain to compute the time ODE [s]

"""
function find_L_critical(
    μ::Float64, 
    η::Float64, 
    ϵ::Float64,
    N::Vector{Int64}, 
    L_in::Float64, 
    ΔL::Float64, 
    Q::Vector{Int64}, 
    t_range::StepRangeLen
    )
    
    L_loop, L_crit = L_in,0 #initialization
    
    nt=length(t_range)#n. of time points
    weight = Array(range(0.9,1.1, nt)) #weight of the points
    
    #ask derivative>0, iterating until the required accuracy
    while ΔL ≥ ϵ

        L_crit=L_loop+ΔL #increasing L 
        L_vec = [L_crit,L_crit,L_crit]
        
        sol_T_loop=sol_T(L_vec,N,μ,η,Q) #solve ODE
        
        #differentiate the solution
        partial_sol_T(t)=derivative(sol_T_loop, t)
        derivative_check=partial_sol_T.(t_range)
        
        #weighted mean 
        if (derivative_check'*weight)/sum(weight)>0 
            ΔL = ΔL/10  #L (over-)critical: finer step 
        else
            L_loop= L_crit #L sub-critical: new starting L  
        end
    
    end
    
    return L_crit

end

#choice of parameters to find L critical
ε = 1e-4;#accuracy;
L_in = 0.0; #initialization of L; 
ΔL = 10e-2; #10cm ;
q_choose = 1; #worst case 
Q_choose = [q_choose, q_choose,q_choose];
nx = 100; #n. of points for the discretization
N = [nx,nx,nx];


L_crit_U = find_L_critical(μ_U, η_U, ε, N, L_in, ΔL, Q_choose, t_range);#m
L_crit_P = find_L_critical(μ_P, η_P, ε, N, L_in, ΔL, Q_choose, t_range); #m

println("The value of L, for the ²³⁵U, is $(L_crit_U*10^2)cm")
println("The value of L, for the ²³⁹Pu, is $(L_crit_P*10^2)cm")

#critical mass m = V*ρ
m_crit_U = L_crit_U^3 *ρ_U; #kg
m_crit_P = L_crit_P^3*ρ_P;#kg

println("The value of the critical mass, for the ²³⁵U, is $(m_crit_U) kg")
println("The value of the critical mass, for the ²³⁹Pu, is $(m_crit_P) kg")

##### series expansion #####

"""
Returns the normalization of the function `F` in a given domain `1/N² = ∫dx|F(x)|²`.

If `x` is of type `StepRangeLen` it casts `Array` on it before computing the rest.

# Arguments
- `x::Vector{Float64}`: domain of the function
- `F::Vector{Float64}`: discretized function

"""
normalization(
    x::Vector{Float64}, 
    F::Vector{Float64}
) = 1/sqrt(integrate(x, F.*F));

normalization(
    x::StepRangeLen, 
    F::Vector{Float64}
) = normalization(Array(x), F);

#function to calculate the coef of the expansion
"""
Compute the expansion in eigenvectors of the ODEs: 

`-μ d²(X_qx)/dx² = α_qx X_qx`,

`-μ d²(Y_qy)/dy² = α_qy Y_qy`,

`-μ d²(Z_qz)/dz² = α_qz Z_qz`, 

with their time evolution `T`, satisfying the time ODE: 

`d(T_q)/dt = (η-α_q)T_q` where `α_q= α_qx+α_qy+α_qz`

of the function `f(x)` in a given domain as

`f(x,t)= ∑_(qx,qy,qz) a_(qx,qy,qz) X_qx(x) Y_qy(y) Z_qz(z) T_q(t)`,
with `qx+qy+qz ≤ Q_max`.

where `f(x,y,z)` must satisfy the boundary conditions
`f(0,y,z) = f(L,y,z) = f(x,0,z) = f(x,L,z) = f(x,y,0) = f(x,y,L)`, `∀x,y,z`.

# Arguments
- `f::Function`:function we are expanding
- `L::Vector{Float64}`:Vector with the length of the domain, Ωᵢ:(0,Lᵢ) [m]
- `N::Vector{Int64}`: Vector with the numbers of points of the discretizations
- `μ::Float64`: diffusion constant [m²/s]
- `η::Float64`: neutron rate of formation [1/s] 
- `t::Float64`: time at which we calculate the expantion [s]
- `Q_max::Int64`: index cut off of the series expansion

"""
function series(
    f::Function,
    L::Vector{Float64},
    N::Vector{Int64},
    μ::Float64,
    η::Float64,
    t::Float64,
    Q_max::Int64
    )
    
    L1, L2, L3 = L #L for X,Y,Z
    nx,ny,nz = N 
    
    #eigenvectors
    X = α_eigenvectors(L1,nx,μ)
    Y = α_eigenvectors(L2,ny,μ)
    Z = α_eigenvectors(L3,nz,μ)
    
    x_range = range(0.,L1,nx) #the range
    y_range = range(0.,L2,ny)
    z_range = range(0.,L3,nz)
    
    #discretize the initial function
    @einsum F[i,j,k] := f(x_range[i], y_range[j], z_range[k]) 
    f_series = zeros(Float64,(nx,ny,nz)) #initialization
    
    for q1 ∈ 1:nx, q2 ∈ 1:ny, q3 ∈ 1:nz 

        if q1+q2+q3 == Q_max #cut off of the expansion
            return f_series
        end    
        
        #q_i-th eigenvectors
        X_q= X[:,q1]
        Y_q= Y[:,q2]
        Z_q= Z[:,q3]

        #total eigenfunction
        @einsum R[i,j,k] := X_q[i]*Y_q[j]*Z_q[k]

        #normalization of the basis
        norm = normalization(x_range,X_q)*
            normalization(y_range,Y_q)*
            normalization(z_range,Z_q)
        #coeff. calculation
        a= norm^2*integrate((x_range,y_range,z_range),F.*R)

        #expansion term updating
        f_series .+= a.*R.*sol_T(L,N,μ,η,[q1,q2,q3])(t)

    end

    return f_series

end


########

#initial function
#initial function
f_initial(
    x::Float64,
    y::Float64,
    z::Float64
) = sin(π*x/L_crit_U)*sin(π*y/L_crit_U)*sin(π*z/L_crit_U); #initial function 

Q_max =50;#cutoff of the series

nx = 100;#n. of points for the discretization
N = [nx,nx,nx]; #n. of points for X,Y,Z

L = [L_crit_U, L_crit_U, L_crit_U]; #L Vector

#calculation of n
n(t::Float64)= series(f_initial,L,N,μ_U, η_U,t,Q_max);

#take n(t=0) for the plot
t_plot = 0.;
n_t = n(t_plot);

L_range_n = range(0.,L_crit_U,nx);
z_fix = round(L_range_n[50],digits=4)*10^2#cm

#fix the z-axis to z=50 for the plot
n_plot = n_t[:,:,50];  

#we need to add back the boundaries on x and y
BC1=zeros(Float64,nx);
newMatrix = hcat(BC1,n_plot,BC1);
BC2=zeros(Float64,nx+2);
newmatrix2 = vcat(BC2',newMatrix, BC2');

newL_range = [0.0;L_range_n;L_crit_U];

#plot
Plot_zero = plot(
    newL_range,newL_range,newmatrix2, st=:surface,
    xlabel="x(m)", ylabel="y(m)", zlabel="n(x,y)",
    title="Neutron diffusion 3D (²³⁵U)\n at L=$(L_crit_U*10^2)cm and Q_max =$Q_max.\nz=$(z_fix)cm t=$(t_plot)s",
     camera=(24,14),dpi=1000
)
savefig(Plot_zero, "N3D_tzero.pdf");#save the fig

### now with L sovra-critical
L_sovra = 21.e-2 #m

#initial function
f_initial_sovra(
    x::Float64,
    y::Float64,
    z::Float64
) = sin(π*x/L_sovra)*sin(π*y/L_sovra)*sin(π*z/L_sovra); #initial function 

L_Sovra = [L_sovra, L_sovra, L_sovra]; #L Vector


#calculation of n
n_sovra(t::Float64)= series(f_initial_sovra,L_Sovra,N,μ_U, η_U,t,Q_max);

#take n(t=0) for the plot
t_plot = 0.;
n_t = n_sovra(t_plot);

L_range_n_Sovra = range(0.,L_sovra,nx);
z_fix_sovra=round(L_range_n_Sovra[50],digits=4)*10^2#cm
#fix the z-axis to z=50 for the plot
n_plot = n_t[:,:,50];  

#we need to add back the boundaries on x and y
BC1=zeros(Float64,nx);
newMatrix = hcat(BC1,n_plot,BC1);
BC2=zeros(Float64,nx+2);
newmatrix2 = vcat(BC2',newMatrix, BC2');

newL_range_sovra = [0.0;L_range_n_Sovra;L_sovra];

#plot
Plot_sovra_zero = plot(
    newL_range_sovra,newL_range_sovra,newmatrix2, st=:surface,
    xlabel="x(m)", ylabel="y(m)", zlabel="n(x,y)",
    title="Neutron diffusion 3D (²³⁵U)\n at L=$(L_sovra*10^2)cm and Q_max =$Q_max.\nz=$(z_fix_sovra)cm t=$(t_plot)s",
    camera=(24,14),dpi=1000
)
savefig(Plot_sovra_zero, "N3D_sovra_tzero.pdf");#save the fig

#### now sovra-critical at a later time

#take n(t>0) for the plot
t_plot = 1e-7;
n_t_sovra = n(t_plot);

L_range_n_sovra_t= range(0.,L_sovra,nx);
z_fix_sovra_t = round(L_range_n_sovra_t[50], digits=4)*10^2 #cm
#fix the z-axis to z=50 for the plot
n_plot_sovra = n_t_sovra[:,:,50];  

#we need to add back the boundaries on x and y
BC1=zeros(Float64,nx);
newMatrix_sovra = hcat(BC1,n_plot_sovra,BC1);
BC2=zeros(Float64,nx+2);
newmatrix_sovra_t = vcat(BC2',newMatrix_sovra, BC2');

newL_range_sovra_t = [0.0;L_range_n_sovra_t;L_sovra];

#plot
Plot_sovra_late = plot(
    newL_range_sovra_t,newL_range_sovra_t,newmatrix_sovra_t, st=:surface,
    xlabel="x(m)", ylabel="y(m)", zlabel="n(x,y)",
    title="Neutron diffusion 3D (²³⁵U)\n at L=$(L_sovra*10^2)cm and Q_max =$Q_max.\nz=$(z_fix_sovra_t)cm, t=1e-7s",
     camera=(24,14),dpi=1000
)
savefig(Plot_sovra_late, "N3D_sovra_tlate.pdf");#save the fig

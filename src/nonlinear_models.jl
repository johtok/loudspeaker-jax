using ModelingToolkit
using ModelingToolkit: D_nounits as D, expand_derivatives, Differential
using Polynomials

"""
Nonlinear speaker models with different mechanical formulations.
"""

function NonlinearSpeakerModel_1stOrder(; name=:speaker)
    @independent_variables t
    @parameters Re=4.6733e0 R2_0=2.2612e0 L2_0=1.2014e-3
    @parameters l0=9.6813e-4 l1=-4.4770e-3 l2=-4.5958e-1 l3=1.0987e2 l4=-9.3638e3
    @parameters f1=-1.7094e-3 f2=8.5381e-5
    @parameters b0=1.1257e1 b1=4.5493e1 b2=-1.0975e4 b3=2.5793e6 b4=-3.1371e8
    @parameters k0=2.8536e3 k1=-5.2850e5 k2=5.6092e7 k3=-1.8850e9 k4=6.8811e10
    @parameters Mm=1.1484e-1 Rm=5.3898e0
    
    @variables i(t)=0.0 i2(t)=0.0 d(t)=0.0 v(t)=0.0 u(t)
    
    # Define the nonlinear parameter expressions
    Le_expr = Polynomial([l0, l1, l2, l3, l4])(d) * Polynomial([1, f1, f2])(i)
    Lest_expr = expand_derivatives(Le_expr + Differential(i)(Le_expr))
    Bl_expr = Polynomial([b0, b1, b2, b3, b4])(d)
    Km_expr = Polynomial([k0, k1, k2, k3, k4])(d)
    L2_expr = L2_0/l0*Le_expr
    R2_expr = R2_0/l0*Le_expr
    
    eqs = [
        # state‐space with magnetic-force and eddy loops
        expand_derivatives(D(i) ~ (-i*(Re + R2_expr)/Lest_expr) + i2*R2_expr/Lest_expr + d*0 + (-v/Lest_expr*(Differential(d)(Le_expr) * i + Bl_expr)) + u/Lest_expr),
        expand_derivatives(D(i2) ~ i*R2_expr/L2_expr+Re*i2+Differential(i)(L2_expr)/(L2_expr*Lest_expr) + (-i2*(R2_expr+Differential(t)(L2_expr))/L2_expr) + d*0 + v*i2*Differential(i)(L2_expr)*(Bl_expr+Differential(d)(Le_expr)*i)/(L2_expr*Lest_expr)),
        expand_derivatives(D(d) ~ i*0 + i2*0 + d*0 + v),
        expand_derivatives(D(v) ~ i*(Bl_expr+1/2*Differential(d)(Le_expr)*i)/Mm + i2*1/2*Differential(d)(L2_expr)*i2/Mm + (-d*Km_expr/Mm) + (-v*Rm/Mm))
    ]
    @named speaker = System(eqs, t)
    return speaker
end

function NonlinearSpeakerModel_2ndOrder(; name=:speaker)
    @independent_variables t
    @parameters Re=4.6733e0 R2_0=2.2612e0 L2_0=1.2014e-3
    @parameters l0=9.6813e-4 l1=-4.4770e-3 l2=-4.5958e-1 l3=1.0987e2 l4=-9.3638e3
    @parameters f1=-1.7094e-3 f2=8.5381e-5
    @parameters b0=1.1257e1 b1=4.5493e1 b2=-1.0975e4 b3=2.5793e6 b4=-3.1371e8
    @parameters k0=2.8536e3 k1=-5.2850e5 k2=5.6092e7 k3=-1.8850e9 k4=6.8811e10
    @parameters Mm=1.1484e-1 Rm=5.3898e0
    
    @variables i(t)=0.0 i2(t)=0.0 d(t)=0.0 u(t)
    
    # Define the nonlinear parameter expressions
    Le_expr = Polynomial([l0, l1, l2, l3, l4])(d) * Polynomial([1, f1, f2])(i)
    Lest_expr = expand_derivatives(Le_expr + Differential(i)(Le_expr))
    Bl_expr = Polynomial([b0, b1, b2, b3, b4])(d)
    Km_expr = Polynomial([k0, k1, k2, k3, k4])(d)
    L2_expr = L2_0/l0*Le_expr
    R2_expr = R2_0/l0*Le_expr
    
    eqs = [
        # state‐space with magnetic-force and eddy loops (1st order electrical)
        expand_derivatives(D(i) ~ (-i*(Re + R2_expr)/Lest_expr) + i2*R2_expr/Lest_expr + (-D(d)/Lest_expr*(Differential(d)(Le_expr) * i + Bl_expr)) + u/Lest_expr),
        expand_derivatives(D(i2) ~ i*R2_expr/L2_expr+Re*i2+Differential(i)(L2_expr)/(L2_expr*Lest_expr) + (-i2*(R2_expr+Differential(t)(L2_expr))/L2_expr) + D(d)*i2*Differential(i)(L2_expr)*(Bl_expr+Differential(d)(Le_expr)*i)/(L2_expr*Lest_expr)),
        # mechanics (2nd order)
        expand_derivatives(D(D(d)) ~ i*(Bl_expr+1/2*Differential(d)(Le_expr)*i)/Mm + i2*1/2*Differential(d)(L2_expr)*i2/Mm + (-d*Km_expr/Mm) + (-D(d)*Rm/Mm))
    ]
    @named speaker = System(eqs, t)
    return speaker
end

"""
Nonlinear speaker model (1st order) as a split ODE system.
Returns a tuple of (electrical_system!, mechanical_system!, parameters) for use with SplitODEProblem.
"""
function NonlinearSpeakerModel_1stOrder_SplitODE(; name=:speaker)
    # Model parameters
    Re = 4.6733e0
    R2_0 = 2.2612e0
    L2_0 = 1.2014e-3
    l0 = 9.6813e-4
    l1 = -4.4770e-3
    l2 = -4.5958e-1
    l3 = 1.0987e2
    l4 = -9.3638e3
    f1 = -1.7094e-3
    f2 = 8.5381e-5
    b0 = 1.1257e1
    b1 = 4.5493e1
    b2 = -1.0975e4
    b3 = 2.5793e6
    b4 = -3.1371e8
    k0 = 2.8536e3
    k1 = -5.2850e5
    k2 = 5.6092e7
    k3 = -1.8850e9
    k4 = 6.8811e10
    Mm = 1.1484e-1
    Rm = 5.3898e0
    
    # Create polynomial objects
    P_d_poly = Polynomial([l0, l1, l2, l3, l4])  # P(d) polynomial
    Q_i_poly = Polynomial([1, f1, f2])           # Q(i) polynomial
    BL_poly = Polynomial([b0, b1, b2, b3, b4])   # BL(d) polynomial
    Km_poly = Polynomial([k0, k1, k2, k3, k4])   # Km(d) polynomial
    
    # Parameter unpacking function
    function unpack_params!(i, i2, d_mm, v, p)
        P_d_poly, Q_i_poly, BL_poly, Km_poly,
        L2_0, R2_0, Re, Mm, Rm, u_fun = p
        
        # Convert d from mm to m for internal calculations
        d_m = d_mm / 100.0
        
        # Compute using polynomials
        P_d = P_d_poly(d_m)
        Q_i = Q_i_poly(i)
        Le = P_d * Q_i
        
        # Derivatives using polynomial differentiation
        dP_dd = derivative(P_d_poly)(d_m)
        dQ_di = derivative(Q_i_poly)(i)
        
        dLe_dd = dP_dd * Q_i
        dLe_di = P_d * dQ_di
        Lest = Le + i * dLe_di
        
        BL = BL_poly(d_m)
        Km = Km_poly(d_m)
        
        # Eddy current parameters
        L2 = L2_0/l0 * Le
        R2 = R2_0/l0 * Le
        dL2_dd = L2_0/l0 * dLe_dd
        dL2_di = L2_0/l0 * dLe_di
        dL2_dt = dL2_dd * v
        
        return Le, dLe_dd, dLe_di, Lest, BL, Km, L2, R2, dL2_dd, dL2_di, dL2_dt, Re, Mm, Rm, u_fun
    end
    
    # Electrical system (handles i and i2 dynamics)
    function electrical_system!(du, u, p, t)
        i, i2, d_mm, v = u
        Le, dLe_dd, dLe_di, Lest, BL, _, L2, R2, dL2_dd, dL2_di, dL2_dt, Re, _, _, u_fun = unpack_params!(i, i2, d_mm, v, p)
        
        ui = u_fun(t)  # user-supplied input u(t)
        
        # Electrical dynamics
        du[1] = -i*(Re + R2)/Lest + i2*R2/Lest - (v/Lest)*(dLe_dd*i + BL) + ui/Lest
        du[2] = i*R2/L2 + Re*i2 + (dL2_di * du[1])/(L2*Lest) - i2*(R2 + dL2_dt)/L2 + v*i2*(dL2_di*(BL + dLe_dd*i))/(L2*Lest)
        
        # Mechanical states remain unchanged in electrical step
        du[3] = 0.0  # d_mm
        du[4] = 0.0  # v
    end
    
    # Mechanical system (handles d and v dynamics)
    function mechanical_system!(du, u, p, t)
        i, i2, d_mm, v = u
        _, dLe_dd, _, _, BL, Km, _, _, dL2_dd, _, _, _, Mm, Rm, _ = unpack_params!(i, i2, d_mm, v, p)
        
        # Electrical states remain unchanged in mechanical step
        du[1] = 0.0  # i
        du[2] = 0.0  # i2
        
        # Mechanical dynamics
        # Note: v is in m/s, so d_mm/dt = v * 100 (convert m/s to mm/s)
        du[3] = v * 100.0  # ḋ_mm = v * 100 (convert m/s to mm/s)
        du[4] = i*(BL + 0.5*dLe_dd*i)/Mm + 0.5*i2^2 * dL2_dd/Mm - (d_mm/100.0)*Km/Mm - v*Rm/Mm  # v̇ (d_mm converted to m)
    end
    
    # Return the systems and parameters
    return electrical_system!, mechanical_system!, (P_d_poly, Q_i_poly, BL_poly, Km_poly, L2_0, R2_0, Re, Mm, Rm)
end

"""
Nonlinear speaker model (1st order) as a split ODE system with settable nonlinearities.
Returns a tuple of (electrical_system!, mechanical_system!, parameters) for use with SplitODEProblem.

# Arguments
- `P_d_coeffs`: Coefficients for P(d) polynomial [l0, l1, l2, l3, l4]
- `Q_i_coeffs`: Coefficients for Q(i) polynomial [1, f1, f2]
- `BL_coeffs`: Coefficients for BL(d) polynomial [b0, b1, b2, b3, b4]
- `Km_coeffs`: Coefficients for Km(d) polynomial [k0, k1, k2, k3, k4]
- `Re`: Electrical resistance (default: 4.6733e0)
- `R2_0`: Eddy current resistance (default: 2.2612e0)
- `L2_0`: Eddy current inductance (default: 1.2014e-3)
- `Mm`: Mechanical mass (default: 1.1484e-1)
- `Rm`: Mechanical resistance (default: 5.3898e0)

# Example
```julia
# Linear model (no nonlinearities)
P_d_coeffs = [9.6813e-4, 0.0, 0.0, 0.0, 0.0]  # constant inductance
Q_i_coeffs = [1.0, 0.0, 0.0]  # no current dependence
BL_coeffs = [1.1257e1, 0.0, 0.0, 0.0, 0.0]  # constant BL
Km_coeffs = [2.8536e3, 0.0, 0.0, 0.0, 0.0]  # constant stiffness

electrical_system!, mechanical_system!, params = NonlinearSpeakerModel_1stOrder_SplitODE_Settable(
    P_d_coeffs, Q_i_coeffs, BL_coeffs, Km_coeffs
)
```
"""
function NonlinearSpeakerModel_1stOrder_SplitODE_Settable(;
    P_d_coeffs=[9.6813e-4, -4.4770e-3, -4.5958e-1, 1.0987e2, -9.3638e3],
    Q_i_coeffs=[1.0, -1.7094e-3, 8.5381e-5],
    BL_coeffs=[1.1257e1, 4.5493e1, -1.0975e4, 2.5793e6, -3.1371e8],
    Km_coeffs=[2.8536e3, -5.2850e5, 5.6092e7, -1.8850e9, 6.8811e10],
    Re=4.6733e0, R2_0=2.2612e0, L2_0=1.2014e-3,
    Mm=1.1484e-1, Rm=5.3898e0,
    name=:speaker)
    
    # Create polynomial objects from coefficients
    P_d_poly = Polynomial(P_d_coeffs)  # P(d) polynomial
    Q_i_poly = Polynomial(Q_i_coeffs)  # Q(i) polynomial
    BL_poly = Polynomial(BL_coeffs)    # BL(d) polynomial
    Km_poly = Polynomial(Km_coeffs)    # Km(d) polynomial
    
    # Parameter unpacking function
    function unpack_params!(i, i2, d_mm, v, p)
        P_d_poly, Q_i_poly, BL_poly, Km_poly,
        L2_0, R2_0, Re, Mm, Rm, u_fun = p
        
        # Convert d from mm to m for internal calculations
        d_m = d_mm / 100.0
        
        # Compute using polynomials
        P_d = P_d_poly(d_m)
        Q_i = Q_i_poly(i)
        Le = P_d * Q_i
        
        # Derivatives using polynomial differentiation
        dP_dd = derivative(P_d_poly)(d_m)
        dQ_di = derivative(Q_i_poly)(i)
        
        dLe_dd = dP_dd * Q_i
        dLe_di = P_d * dQ_di
        Lest = Le + i * dLe_di
        
        BL = BL_poly(d_m)
        Km = Km_poly(d_m)
        
        # Eddy current parameters
        L2 = L2_0/P_d_coeffs[1] * Le  # Use first coefficient as reference
        R2 = R2_0/P_d_coeffs[1] * Le
        dL2_dd = L2_0/P_d_coeffs[1] * dLe_dd
        dL2_di = L2_0/P_d_coeffs[1] * dLe_di
        dL2_dt = dL2_dd * v
        
        return Le, dLe_dd, dLe_di, Lest, BL, Km, L2, R2, dL2_dd, dL2_di, dL2_dt, Re, Mm, Rm, u_fun
    end
    
    # Electrical system (handles i and i2 dynamics)
    function electrical_system!(du, u, p, t)
        i, i2, d_mm, v = u
        Le, dLe_dd, dLe_di, Lest, BL, _, L2, R2, dL2_dd, dL2_di, dL2_dt, Re, _, _, u_fun = unpack_params!(i, i2, d_mm, v, p)
        
        ui = u_fun(t)  # user-supplied input u(t)
        
        # Electrical dynamics
        du[1] = -i*(Re + R2)/Lest + i2*R2/Lest - (v/Lest)*(dLe_dd*i + BL) + ui/Lest
        du[2] = i*R2/L2 + Re*i2 + (dL2_di * du[1])/(L2*Lest) - i2*(R2 + dL2_dt)/L2 + v*i2*(dL2_di*(BL + dLe_dd*i))/(L2*Lest)
        
        # Mechanical states remain unchanged in electrical step
        du[3] = 0.0  # d_mm
        du[4] = 0.0  # v
    end
    
    # Mechanical system (handles d and v dynamics)
    function mechanical_system!(du, u, p, t)
        i, i2, d_mm, v = u
        _, dLe_dd, _, _, BL, Km, _, _, dL2_dd, _, _, _, Mm, Rm, _ = unpack_params!(i, i2, d_mm, v, p)
        
        # Electrical states remain unchanged in mechanical step
        du[1] = 0.0  # i
        du[2] = 0.0  # i2
        
        # Mechanical dynamics
        # Note: v is in m/s, so d_mm/dt = v * 100 (convert m/s to mm/s)
        du[3] = v * 100.0  # ḋ_mm = v * 100 (convert m/s to mm/s)
        du[4] = i*(BL + 0.5*dLe_dd*i)/Mm + 0.5*i2^2 * dL2_dd/Mm - (d_mm/100.0)*Km/Mm - v*Rm/Mm  # v̇ (d_mm converted to m)
    end
    
    # Return the systems and parameters
    return electrical_system!, mechanical_system!, (P_d_poly, Q_i_poly, BL_poly, Km_poly, L2_0, R2_0, Re, Mm, Rm)
end 
# Final Project: The Artemis Re-entry Challenge

## Context

Your team at a private aerospace firm is tasked with simulating the final 100 seconds of a capsule's re-entry into the Earth's atmosphere. You are given a set of discrete, noisy data points representing the Drag Coefficient ($C_d$) as a function of the Angle of Attack ($\alpha$).

To successfully ensure the safety of the payload, you must reconstruct the drag profile, solve the equations of motion using a custom ODE solver, and calculate the total thermal energy absorbed by the heat shield.


## The Equations of Motion

We model the capsule's descent in 1D (pure vertical drop for simplicity). Let $y(t)$ be the altitude in meters and $v(t) = \frac{dy}{dt}$ be the vertical velocity. Because the capsule is falling, $v(t) < 0$, and atmospheric drag acts upwards (positive $y$ direction).

The system of Ordinary Differential Equations (ODEs) is:

$$\frac{dy}{dt} = v$$
$$\frac{dv}{dt} = -g + \frac{1}{2m} \rho(y) A C_d(\alpha(t)) v^2$$

### Parameters Description:
* Gravity ($g$): $9.81 \text{ m/s}^2$
* Capsule Mass ($m$): $3000 \text{ kg}$
* Cross-sectional Area ($A$): $12.0 \text{ m}^2$
* Atmospheric Density ($\rho$): Uses the barometric formula $\rho(y) = \rho_0 e^{-y/H}$
    * $\rho_0 = 1.225 \text{ kg/m}^3$ (Sea level density)
    * $H = 8500 \text{ m}$ (Scale height)
* Angle of Attack Profile ($\alpha(t)$ ): The capsule's angle of attack oscillates during descent:
  
    $$\alpha(t) = 20 + 15 \sin\left(\frac{2\pi t}{100}\right) \quad \text{(in degrees)}$$

### Initial Conditions:
* $y(0) = 50,000 \text{ m}$
* $v(0) = -2,000 \text{ m/s}$
* Time interval: $t \in [0, 100] \text{ seconds}$

### Drag Data:
You are provided the following wind-tunnel data for the drag coefficient:
* $\alpha$ (degrees): `[0, 10, 20, 30, 40]`
* $C_d$ (dimensionless): `[0.25, 0.55, 0.95, 1.35, 1.60]`


## Project Requirements

### Data Reconstruction
You must interpolate the $C_d(\alpha)$ data using a basis from the library. Explain your choice.

### ODE Solver
You must use a numerical ODE solver you developed from scratch (see lab in class for example or implement another one). Do not use `scipy.integrate.solve_ivp`. Solve the coupled system for $y(t)$ and $v(t)$.

### Kinematic Analysis
The human body can only withstand a certain amount of **Jerk** (the rate of change of acceleration, $j(t) = a'(t)$). 
Use the numerical differentiation module to compute the jerk profile $j(t)$ from your computed acceleration data. 

### Thermal Energy
The total work done by the drag force is converted into heat. 

$$W_{drag} = \int_{0}^{T} F_{drag}(t) |v(t)| dt = \int_{0}^{T} \left( \frac{1}{2} \rho(y(t)) v(t)^2 A C_d(\alpha(t)) \right) |v(t)| dt$$

Use the integration module to calculate the total heat energy absorbed in Joules.


## Submission Guidelines
* Submit a single Python Notebook through Canvas.
* Provide clear justification and descriptions.
* Use the plotting cababilities of the library when possible. Devise your own plots that you deem informative.
* The final plot should be a 2x2 panel plot showing:
    1.  The interpolated $C_d(\alpha)$ curve vs. data points.
    2.  Altitude $y(t)$ and velocity $v(t)$ over time.
    3.  Acceleration $a(t)$ over time.
    4.  Jerk $j(t)$ over time.
* At the end of the project, answer the following question (no implementation needed):
   * Assume the capsule loses a part of its heat shield at a certain time ($t=50$) resulting in an instant drop of the drag coefficient. What would you change to solve the same problem?


A. Rewrite the above as an unconstraint problem (use the linear constraint to substitute out  x2
 )
 
 \begin{equation}
\begin{array}{rrclcl}
\displaystyle \max  & x_1^{0.5}.x_2^{0.5}\\
\textrm{s.t.} & {0.5}x_1 + x_2=1 \\
& x_1,x_2\geq0\\
\end{array}
\end{equation}

i. Plots the objective function for the feasible range for  x1

ii. Write down the first order condition and find the solution (assume an interior solution).

iii. Derive the second derivative of the objective function at the optimal.

iv. Evaluate the second derivative at the optimal value of  x1 and for the entire feasible range. Plots your results.

v. Find the solution using a Optim package, and the GoldenSection() solver option.

## i

rewriting optimization:

$x_2=1-0.5x_1$


Now we are solving the below
 \begin{equation}
\begin{array}{rrclcl}
\displaystyle \max  & (x_1-0.5x_1^2)^{0.5}\\
\textrm{s.t.} 
& 2 \geq x_1\geq0\\
\end{array}
\end{equation}


```julia
using PyPlot

x = collect(0:0.01:2)
ff = (x- 0.5* x.^2).^0.5

fig, ax = subplots()
ax[:plot](x,ff,color="blue",linewidth=2,label=L"(x_1-0.5x_1^2)^{0.5}",alpha=1)
ax[:legend](loc="top center")
```


![png](output_3_0.png)





    PyObject <matplotlib.legend.Legend object at 0x7fde8cb52f10>



## ii

$ f = (x- 0.5x^2)^{0.5}$

$ Df=\dfrac{Df(x)}{x} = \dfrac{0.5-0.5x}{\sqrt{(x- 0.5 x^2)}}=0$

$\rightarrow  0.5-0.5x=0 $

$\rightarrow x=1$


Below is the plot of derivative of f in the range


```julia
using PyPlot

x = collect(0:0.01:2)
Df = (0.5-0.5*x)/(x- 0.5* x.^2).^0.5

fig, ax = subplots()
ax[:plot](x,Df,color="blue",linewidth=2,alpha=1)
```


![png](output_6_0.png)





    201-element Array{Any,1}:
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca13b10>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca13dd0>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40050>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40210>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca403d0>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40590>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40750>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40910>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40ad0>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40c90>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca40e50>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca39050>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8ca39210>
     ⋮                                                          
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd4bc90>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd4be50>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56050>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56210>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd563d0>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56590>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56750>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56910>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56ad0>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56c90>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd56e50>
     PyObject <matplotlib.lines.Line2D object at 0x7fde8cd5d050>



##  iii & iv

$ Df^{''}=\dfrac{D^{2}f(x)}{dx^2} = \dfrac{-0.5}{(2x-x^2) \sqrt{(x- 0.5 x^2)}}$

$\rightarrow  Max { Df^{''}}= \dfrac{-1}{\sqrt 2} $ $at$ $x=1$


```julia
using PyPlot

x = collect(0:0.01:2)
#Df2= (-0.5)./ (2*x-x.^2)
Df2 = -0.5 ./ ((2 * x - x.^2).*((x- 0.5*x .^2).^0.5))

plot(x,Df2)
```


![png](output_9_0.png)





    1-element Array{Any,1}:
     PyObject <matplotlib.lines.Line2D object at 0x7fde8a6bd490>



As shown all values are negative
$Df''$ doesn seem to be positive semi-definite.

## V


```julia
using Optim
f_univariate(x) = -sqrt(x - 0.5 *x^2)
optimize(f_univariate, 0.0,2.0,GoldenSection())
```

    WARNING: Base.ASCIIString is deprecated, use String instead.
      likely near In[1]:3
    WARNING: Base.ASCIIString is deprecated, use String instead.
      likely near In[1]:3
    WARNING: Base.ASCIIString is deprecated, use String instead.
      likely near In[1]:3
    WARNING: Base.ASCIIString is deprecated, use String instead.
      likely near In[1]:3





    Results of Optimization Algorithm
     * Algorithm: Golden Section Search
     * Search Interval: [0.000000, 2.000000]
     * Minimizer: 1.000000
     * Minimum: -0.707107
     * Iterations: 37
     * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16): true
     * Objective Function Calls: 38



### for some reason this optimization sometimes work and sometimes doesn't print the result.

here is a printout of output from shell

optimize(f_univariate, 0.0,2.0,GoldenSection())
Results of Optimization Algorithm
  * Algorithm: Golden Section Search
  * Search Interval: [0.000000, 2.000000]
 * Minimizer: 1.000000e+00
 * Minimum: -7.071068e-01
 * Iterations: 37
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16): true
 * Objective Function Calls: 38


Confirming that 1 is the minimizer and maximum is is -(-0.707107)

We can also get $x_2$ which means $x_2=.5$

** B. Write out the lagrange of the original optimization problem **

** i. Derive the first order conditions **

** ii. Solve the first order conditions and find the lagranger multiplier value **


## i

$f = \sqrt x_1 . \sqrt{x_2} $

$h(x)=0.5x_1+x_2-1$

$ f'=[\dfrac{Df(x_1,x_2)}{x_1} ; \dfrac{Df(x_1,x_2)}{x_2}]  = [\dfrac{\sqrt{x_2}}{2\sqrt{x_1}} ;\dfrac{\sqrt{x_1}}{2\sqrt{x_2}}]$

$L(x,\lambda)= f(x)+\lambda^{T}h(x)$
$L(x,\lambda)= \sqrt x_1 . \sqrt{x_2} +\lambda(0.5x_1+x_2-1)$

# ii

$L^{'}(x,\lambda)$

1)$\dfrac{\sqrt x_2}{2\sqrt x_1}+0.5\lambda=0$

2)$\dfrac{\sqrt x_1}{2\sqrt x_2}+\lambda=0$

3)$ 0.5x_1+x_2=1$

solving above $x_1=1, x_2=0.5$


```julia
-sqrt(1)/2*sqrt(0.5)
# here is lambda
```




    -0.3535533905932738



Could also be calculated as below, but  with no constraint on x sign we will have are searching for a solution in the complex space. Need to incorporate as filter to solve for this


```julia
function df(x)
    #x[1]>=0
    #x[2]>=0
    return [sqrt(x[2])/2*sqrt(x[1]);sqrt(x[1])/2*sqrt(x[2])]
end

function d2f(x)
    return [-sqrt(x[2])/4*x[1]^1.5 1/4*sqrt(x[1])*sqrt(x[2]);1/4*sqrt(x[2])*sqrt(x[1]) -sqrt(x[1])/4*x[2]^1.5]
end

function h(x)
    return 0.5*x[1]+x[2]-1
end

function dh(x)
    return [0.5;1]
end

function d2h(x)
    return [0 0;0 0]
end

tol = 1.0e-5
x0 = [1;0.5]
#x0[1]>=0
#x0[2]>=0
lb0 = 1.0
err=1.0e10
k=1
while err>tol 
    dL0 = [df(x0)+dh(x0).*lb0; h(x0)]
    d2L0 = [d2f(x0)+d2h(x0).*lb0 dh(x0);
            dh(x0)' 0]
    dd = -inv(d2L0)*dL0
    x=x0+dd[1:length(x0)]
    lb=lb0+dd[(length(x0)+1):(length(x0)+length(lb0))]
    err=maximum(abs([x0;lb0]-[x;lb]))
    x0,lb0 = x,lb
    println(k,x0,lb0)
    k = k+1
end
```

    1[1.47059,0.264706][-0.45754]
    2[1.86479,0.0676044][-0.381584]
    3[2.21181,-0.105905][-0.209375]



    DomainError:
    sqrt will only return a complex result if called with a complex argument. Try sqrt(complex(x)).

    

     in sqrt(::Float64) at ./math.jl:209

     in macro expansion; at ./In[2]:31 [inlined]

     in anonymous at ./<missing>:?




C. Find the solution to the original problem using Julia's NLopt and the LD_SLSQP solver.


```julia
using JuMP, NLopt
m = Model(solver=NLoptSolver(algorithm=:LD_SLSQP))

@variable(m, x1, start = 0.5)
@variable(m, x2, start = 0.1)
@NLconstraint(m, 0.5*x1+x2  ==  1)
@NLobjective(m, Max, sqrt(x1) * sqrt(x2))

solve(m)
println("x1 = ", getvalue(x1), "; x2 = ", getvalue(x2), " z = ", getobjectivevalue(m))
```

    x1 = 0.9999999999099749; x2 = 0.5000000000450124 z = 0.7071067811865475


2 - You were given a sample of size  $N$  containing a response variable  $y$  and predictors  $x_j$ ,  $j=1,2,..,k$ . Let  $y$  be a  $N×1$  vector the values for response variable, and  $X$  be a  N×k  matrix containing the values for each predictor. Assume that the first predictor takes the value of  $1$  for every observation. Lastly, assume that the response variable  $y$  is linearly related to the predictors.

A. Write out the least-squares minimization problem in matrix form:

B. Write out the first order conditions, and derive the closed-form solution,  β^  in matrix notation.

C. Given the dataset provided:

i. Create the matrix  y  and  X  and compute the value of  β^ 

ii. Compute the optimal  β^  using a Julia optimization solver.


###  A & B.

$$Y= X\beta + \epsilon$$

If $b$ is a $k * 1$ vector of estimates of $\beta$, then the estimated model may be
written as

$$y= Xb +e$$

Here e denotes the n * 1 vector of residuals, which can be computed from the
data and the vector of estimates $b$ by means of

$$e =y - Xb$$

least square estimator is by minimizinf $S(b)$

$$S(b)=\sum_{i=1}^n e_i^{2}=ee^{'} $$ 

objective is to minimize $S(b) $ so we set the derivative eqaul to 0

$$\frac{\partial S}{\partial b} = -2X^{'}y+2X^{'}Xb$$

$$X^{'}y=X^{'}Xb$$





### C

Dataset missing...


```julia
#y_hat = X'*beta + v
#rms_error = norm(y_hat-y)/sqrt(length(y))
```

3 - Write out the Taylor expansion of the function  $f(x)=e^x$ , include the first six components. Plot the function  $f(x)=e^x$  and its first, second, third, forth, and fifth order approximation in the range  [0,2].

The Taylor series of  $f(x)$
  about  $x=x_0+ϵ$
  

$f(x)≈ f(x_0)+\frac{1}{1!} f^1(x_0)ϵ +\frac{1}{2!}f^2(x_0)ϵ^2+\frac{1}{3!}f^3(x_0)ϵ^3+\frac{1}{4!}f^4(x_0)ϵ^4+\frac{1}{5!}f^5(x_0)ϵ^5+\frac{1}{6!}f^6(x_0)ϵ^6$
 
 


$f(x)≈ e^{x_0}+\frac{1}{1!} e^{x_0}ϵ +\frac{1}{2!}e^{x_0}ϵ^2+\frac{1}{3!}e^{x_0}ϵ^3+\frac{1}{4!}e^{x_0}ϵ^4+\frac{1}{5!}e^{x_0}ϵ^5+\frac{1}{6!}e^{x_0}ϵ^6$


```julia
using Plots
pyplot()

x = collect(0:0.1:2)

eps=1.2
x0=x-eps


fx = e.^x

f1x = e.^x0+ (eps.*e.^x0)/2

f2x = e.^x0+ (eps.*e.^x0)/2 + (eps^2.*e.^x0)/6

f3x = e.^x0+ (eps.*e.^x0)/2 + (eps^2.*e.^x0)/6 + (eps^3.*e.^x0)/9

f4x = e.^x0+ (eps.*e.^x0)/2 + (eps^2.*e.^x0)/6 + (eps^3.*e.^x0)/9+ (eps^4.*e.^x)/16

f5x = e.^x0+ (eps.*e.^x0)/2 + (eps^2.*e.^x0)/6 + (eps^3.*e.^x0)/9+ (eps^4.*e.^x)/16+ (eps^5.*e.^x)/25

f6x = e.^x0+ (eps.*e.^x0)/2 + (eps^2.*e.^x0)/6 + (eps^3.*e.^x0)/9+ (eps^4.*e.^x)/16+ (eps^5.*e.^x)/25+ (eps^6.*e.^x)/36


default(size=(700,700))
Plots.plot(x,fx,color=:red, linewidth=1, label= "f(x) = e^{2}")
Plots.plot!(x,f1x,color=:blue, linewidth=1, label="first")
Plots.plot!(x,f2x,color=:green, linewidth=1, label="Second")
Plots.plot!(x,f3x,color=:black, linewidth=1, label="Third")
Plots.plot!(x,f4x,color=:orange, linewidth=1, label="Fourth")
Plots.plot!(x,f5x,color=:pink, linewidth=1, label="Fifth")
Plots.plot!(x,f6x,color=:grey, linewidth=1, label="Sixth")
```




<img src="data:image/png;base64,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" />



## Output didn't Print. Please see  other attachment for the graph

## 4. Find implied volatility

Since the Black Sholes equation is already given, we essentially have to solve for volatility (aka implied volatility)


```julia
using Distributions
using DataFrames

stdnorm = Normal(0,1)

function option(S0, K, T, r, s, C)
    d1 = (log(S0/K)+(r+0.5*s^2)*T)/(s*sqrt(T))
    d2 = d1 - s * sqrt(T)
    (S0 * cdf(stdnorm,d1) - K * e^(r * T) * cdf(stdnorm,d2)) - C
end

function bisec(f::Function,a::Float64,b::Float64,tol::Float64)
    if a>=b
        error("a must be less than b") 
    end
    if f(a)*f(b) > 0
        error("signs of f(a) and f(b) are not opposites") 
    end
    i = 1
    err=b-a
    c=(a+b)/2
    resual = DataFrame(a=a,b=b,c=c,err=err)
    while err>tol
        if f(c)==0.0
            break
        end
        if f(a)*f(c)<0
            a=a
            b=c
        else
            a=c
            b=b
        end
        c=(a+b)/2
        err=b-a
        i = i + 1
        push!(resual, @data([a,b,c,err]))
    end
    return c, resual
end
```




    bisec (generic function with 1 method)



### First stock




```julia
function ff1(s)
    option(50,50,250, 0.00008,s,4.4580) 
end

c, resual = bisec(ff1,0.,1.,1e-6);
println("Bisection result")
println(c)
println("Iterations")
println(resual)
```

    Bisection result
    0.015770435333251953
    Iterations
    21×4 DataFrames.DataFrame
    │ Row │ a         │ b         │ c         │ err         │
    ├─────┼───────────┼───────────┼───────────┼─────────────┤
    │ 1   │ 0.0       │ 1.0       │ 0.5       │ 1.0         │
    │ 2   │ 0.0       │ 0.5       │ 0.25      │ 0.5         │
    │ 3   │ 0.0       │ 0.25      │ 0.125     │ 0.25        │
    │ 4   │ 0.0       │ 0.125     │ 0.0625    │ 0.125       │
    │ 5   │ 0.0       │ 0.0625    │ 0.03125   │ 0.0625      │
    │ 6   │ 0.0       │ 0.03125   │ 0.015625  │ 0.03125     │
    │ 7   │ 0.015625  │ 0.03125   │ 0.0234375 │ 0.015625    │
    │ 8   │ 0.015625  │ 0.0234375 │ 0.0195313 │ 0.0078125   │
    │ 9   │ 0.015625  │ 0.0195313 │ 0.0175781 │ 0.00390625  │
    │ 10  │ 0.015625  │ 0.0175781 │ 0.0166016 │ 0.00195313  │
    │ 11  │ 0.015625  │ 0.0166016 │ 0.0161133 │ 0.000976563 │
    │ 12  │ 0.015625  │ 0.0161133 │ 0.0158691 │ 0.000488281 │
    │ 13  │ 0.015625  │ 0.0158691 │ 0.0157471 │ 0.000244141 │
    │ 14  │ 0.0157471 │ 0.0158691 │ 0.0158081 │ 0.00012207  │
    │ 15  │ 0.0157471 │ 0.0158081 │ 0.0157776 │ 6.10352e-5  │
    │ 16  │ 0.0157471 │ 0.0157776 │ 0.0157623 │ 3.05176e-5  │
    │ 17  │ 0.0157623 │ 0.0157776 │ 0.01577   │ 1.52588e-5  │
    │ 18  │ 0.01577   │ 0.0157776 │ 0.0157738 │ 7.62939e-6  │
    │ 19  │ 0.01577   │ 0.0157738 │ 0.0157719 │ 3.8147e-6   │
    │ 20  │ 0.01577   │ 0.0157719 │ 0.0157709 │ 1.90735e-6  │
    │ 21  │ 0.01577   │ 0.0157709 │ 0.0157704 │ 9.53674e-7  │



```julia
println("Let's make sure the calculated volatility indeed results in a root")
println(ff1(0.0157709))
```

    Let's make sure the calculated volatility indeed results in a root
    0.00018449534258202505


### Second stock





```julia
function ff2(s)
    option(50,50,250,0.00016,s,4.0143) 
end

c, resual = bisec(ff2,0.,1.,1e-6);
println("Bisection result")
println(c)
println("Iterations")
println(resual)
```

    Bisection result
    0.016271114349365234
    Iterations
    21×4 DataFrames.DataFrame
    │ Row │ a         │ b         │ c         │ err         │
    ├─────┼───────────┼───────────┼───────────┼─────────────┤
    │ 1   │ 0.0       │ 1.0       │ 0.5       │ 1.0         │
    │ 2   │ 0.0       │ 0.5       │ 0.25      │ 0.5         │
    │ 3   │ 0.0       │ 0.25      │ 0.125     │ 0.25        │
    │ 4   │ 0.0       │ 0.125     │ 0.0625    │ 0.125       │
    │ 5   │ 0.0       │ 0.0625    │ 0.03125   │ 0.0625      │
    │ 6   │ 0.0       │ 0.03125   │ 0.015625  │ 0.03125     │
    │ 7   │ 0.015625  │ 0.03125   │ 0.0234375 │ 0.015625    │
    │ 8   │ 0.015625  │ 0.0234375 │ 0.0195313 │ 0.0078125   │
    │ 9   │ 0.015625  │ 0.0195313 │ 0.0175781 │ 0.00390625  │
    │ 10  │ 0.015625  │ 0.0175781 │ 0.0166016 │ 0.00195313  │
    │ 11  │ 0.015625  │ 0.0166016 │ 0.0161133 │ 0.000976563 │
    │ 12  │ 0.0161133 │ 0.0166016 │ 0.0163574 │ 0.000488281 │
    │ 13  │ 0.0161133 │ 0.0163574 │ 0.0162354 │ 0.000244141 │
    │ 14  │ 0.0162354 │ 0.0163574 │ 0.0162964 │ 0.00012207  │
    │ 15  │ 0.0162354 │ 0.0162964 │ 0.0162659 │ 6.10352e-5  │
    │ 16  │ 0.0162659 │ 0.0162964 │ 0.0162811 │ 3.05176e-5  │
    │ 17  │ 0.0162659 │ 0.0162811 │ 0.0162735 │ 1.52588e-5  │
    │ 18  │ 0.0162659 │ 0.0162735 │ 0.0162697 │ 7.62939e-6  │
    │ 19  │ 0.0162697 │ 0.0162735 │ 0.0162716 │ 3.8147e-6   │
    │ 20  │ 0.0162697 │ 0.0162716 │ 0.0162706 │ 1.90735e-6  │
    │ 21  │ 0.0162706 │ 0.0162716 │ 0.0162711 │ 9.53674e-7  │



```julia
println("Let's make sure the calculated volatility indeed results in a root")
println(ff2(0.0162706))
```

    Let's make sure the calculated volatility indeed results in a root
    -8.325771471895393e-5


### Third stock


```julia
function ff3(s)
    option(50,60,250,0.00016,s,3.2961) 
end

c, resual = bisec(ff3,0.,1.,1e-6);
println("Bisection result")
println(c)
println("Iterations")
println(resual)
```

    Bisection result
    0.023456096649169922
    Iterations
    21×4 DataFrames.DataFrame
    │ Row │ a         │ b         │ c         │ err         │
    ├─────┼───────────┼───────────┼───────────┼─────────────┤
    │ 1   │ 0.0       │ 1.0       │ 0.5       │ 1.0         │
    │ 2   │ 0.0       │ 0.5       │ 0.25      │ 0.5         │
    │ 3   │ 0.0       │ 0.25      │ 0.125     │ 0.25        │
    │ 4   │ 0.0       │ 0.125     │ 0.0625    │ 0.125       │
    │ 5   │ 0.0       │ 0.0625    │ 0.03125   │ 0.0625      │
    │ 6   │ 0.0       │ 0.03125   │ 0.015625  │ 0.03125     │
    │ 7   │ 0.015625  │ 0.03125   │ 0.0234375 │ 0.015625    │
    │ 8   │ 0.0234375 │ 0.03125   │ 0.0273438 │ 0.0078125   │
    │ 9   │ 0.0234375 │ 0.0273438 │ 0.0253906 │ 0.00390625  │
    │ 10  │ 0.0234375 │ 0.0253906 │ 0.0244141 │ 0.00195313  │
    │ 11  │ 0.0234375 │ 0.0244141 │ 0.0239258 │ 0.000976563 │
    │ 12  │ 0.0234375 │ 0.0239258 │ 0.0236816 │ 0.000488281 │
    │ 13  │ 0.0234375 │ 0.0236816 │ 0.0235596 │ 0.000244141 │
    │ 14  │ 0.0234375 │ 0.0235596 │ 0.0234985 │ 0.00012207  │
    │ 15  │ 0.0234375 │ 0.0234985 │ 0.023468  │ 6.10352e-5  │
    │ 16  │ 0.0234375 │ 0.023468  │ 0.0234528 │ 3.05176e-5  │
    │ 17  │ 0.0234528 │ 0.023468  │ 0.0234604 │ 1.52588e-5  │
    │ 18  │ 0.0234528 │ 0.0234604 │ 0.0234566 │ 7.62939e-6  │
    │ 19  │ 0.0234528 │ 0.0234566 │ 0.0234547 │ 3.8147e-6   │
    │ 20  │ 0.0234547 │ 0.0234566 │ 0.0234556 │ 1.90735e-6  │
    │ 21  │ 0.0234556 │ 0.0234566 │ 0.0234561 │ 9.53674e-7  │



```julia
println("Let's make sure the calculated volatility indeed results in a root")
println(ff3(0.0234556))
```

    Let's make sure the calculated volatility indeed results in a root
    -5.618692313991147e-5



```julia

```

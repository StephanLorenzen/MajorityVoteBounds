from math import ceil, log, sqrt, exp

# Compute Igel bound:
def lamb(emp_risk, n, r, KL, delta=0.5):

    lamb = 2.0 / (sqrt((2.0*float(n-r)*emp_risk)/(KL+log(2.0*sqrt(float(n-r))/delta)) + 1.0) + 1.0)
    
    bound = emp_risk / (1.0 - lamb/2.0) + (KL + log((2.0*sqrt(float(n-r)))/delta))/(lamb*(1.0-lamb/2.0)*float(n-r))

    return min(1.0,2.0*bound)


# Optimize Lambda bound:
def optimizeLamb(emp_risks, n, r, delta=0.5, eps=0.001):
    m = len(emp_risks)
    
    pi  = [1/float(m)] * m
    rho = [1/float(m)] * m
    KL = 0.0
    
    lamb = 1.0
    emp_risk = sum(emp_risks) / float(m)

    bound = 1.0
    upd = emp_risk / (1.0 - lamb/2.0) + (KL + log((2.0*sqrt(float(n-r)))/delta))/(lamb*(1.0-lamb/2.0)*float(n-r))

    while bound-upd > eps:
        bound = upd
        lamb = 2.0 / (sqrt((2.0*float(n-r)*emp_risk)/(KL+log(2.0*sqrt(float(n-r))/delta)) + 1.0) + 1.0)
        for h in range(m):
            rho[h] = pi[h]*exp(-lamb*float(n-r)*emp_risks[h])
        norm = float(sum(rho))
        rho = [r/norm for r in rho]


        emp_risk = sum([rho[h]*emp_risks[h] for h in range(m)])
        KL = computeKL(rho,pi)

        upd = emp_risk / (1.0 - lamb/2.0) + (KL + log((2.0*sqrt(float(n-r)))/delta))/(lamb*(1.0-lamb/2.0)*float(n-r)) 

    return rho

# State space based Monte Carlo simulation for Option pricing

This project shall help estimate the probability of an option to be in the money at maturity. This repository is for research purposes only; no investment advice. 



## What it does

The ss_mc class will take returns as input. Then it can perform the following tasks:
1. estimate the distribution of the returns based on maximum likelihood
2. apply a Kalman filter and fit distributions to the returns for every state idetified by the filter
3. apply a monte carlo simulation to estimate the return distribution at maturity of the option


## Origin of idea

One of the crucial assumptions for option pricing is the assumption of normally distributed returns. Empirically, this seems to be a good but wrong estimate. This project tries to simulate return paths without this assumption. 


## Future Todos: 

In future commits, I will try to add functions to price options after the simulation. Since entire paths are simulated, path-dependent options will be the main target. Also, I will add variance reduction methods to the monte carlo simulation. 
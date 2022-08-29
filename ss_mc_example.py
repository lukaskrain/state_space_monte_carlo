# only for test cases
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web

from ss_mc import *


start = datetime.datetime(2005, 1, 1)
end = datetime.datetime(2017, 1, 1)
f = web.DataReader("F", "yahoo", start, end)
ret = np.array(f.Close.pct_change()[1:]).reshape(-1, 1)
    
n = 500
T = 30
    
ssmc = SS_MC()
ssmc.fit_states(pd.DataFrame(ret), n=2)
ssmc.estimate_scauchy_states()
    
ssmc.fitted_ss_mc(n, T)

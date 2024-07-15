## This file contains a class for analytical and numerical solution for 
## designated problem defined in the ipynb file. Briefly, the equation is
## dT/dt = R (Tenv - T0)

## Importing necessary libraries
import functools
import numpy as np
from scipy.integrate import solve_ivp
import inspect

class Soln():
    
    def __init__(self, Tenv: float = None, T0: float = None, 
                 t_start: float = None, t_end: float = None, 
                 increment: float = None, cooling_rate: float = None, 
                 time_data: list = None):
        
        self.Tenv = Tenv
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.time_data = time_data
        self.increment = increment
        self.t_start = t_start
        self.t_end = t_end
        self.adjust_time_data()


    def adjust_time_data(self):
        self.time_data = self.time_data if self.time_data is not None else np.array([])

        if self.t_start > self.t_end or self.t_start == self.t_end:
            raise ValueError("Start time and end time can't be equal or start time can't be higher than end time")

        if self.time_data.size == 0:
            if self.increment == 0 or self.increment < 0 or self.increment == None:
                raise ValueError("The increment parameter can't be equal or lower than zero -> {}".format(self.increment))
            else:
                self.time_data = np.linspace(self.t_start, self.t_end, int(abs(self.t_end - self.t_start) / self.increment), dtype='int')
        else:
            self.t_start = self.time_data.min()
            self.t_end = self.time_data.max()
        
    def set(self, name, value):

        print("{} changed from {} to {}".format(name, self.t_end, value))
        if name == "t_start" or name == "t_end" or name == "increment":
            print("Re-adjusting time data according to the newly set variables")
            self.adjust_time_data()
        if name not in self.get_init_param_names():
            raise AttributeError(f"Cannot set unknown attribute '{name}'")
        setattr(self, name, value)

    def get(self, name):
        if name not in self.get_init_param_names():
            raise AttributeError(f"Unknown attribute '{name}'")
        return getattr(self, name)

    ## Used for getting the parameters at the initialization stage
    def get_init_param_names(self):
        # Use inspect to get the parameter names from the __init__ method
        return inspect.signature(self.__init__).parameters.keys()

    ## Check input variables during initialization. If there is a None value, 
    ## it will raise an error
    def check_input_variables(self):
        missing_params = [name for name, value in vars(self).items() if value is None]
        if missing_params:
            raise ValueError(f"The following parameters must be provided and cannot be None: {', '.join(missing_params)}")

    ## Returns calculated temperature over a given time defined in self.time_data
    ## the returned value is a np.array    
    def analytical_solution(self):
        self.check_input_variables()
        if self.time_data.size == 0:
            raise ValueError("time_data must be provided and cannot be empty")
        return self.Tenv + (self.T0 - self.Tenv) * np.exp(-self.cooling_rate*self.time_data)
    
    ## Returns calculated temperature over a given time defined in self.time_data
    ## the returned value is a np.array
    def numerical_solution(self):
        self.check_input_variables()
        
        def newton_cooling(t_, T_, R_, T_env):
            return -R_ * (T_ - T_env)
        
        t_span = (self.t_start, self.t_end)  # Time span for the solution
        sol = solve_ivp(newton_cooling, t_span, [self.T0], args=(self.cooling_rate, self.Tenv), t_eval=self.time_data)
        return sol.y[0]
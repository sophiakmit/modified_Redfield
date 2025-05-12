#=====================================================================
#
#                     Modified Redfield Script
#
#=====================================================================

# by Sophia Valentina Kmit Mikkelsen 2024 - 2025

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import codata2014 as const
from scipy.integrate import quad, quad_vec, nsum, solve_ivp
from scipy.optimize import curve_fit


""" using astropy for handling units """


class Bath:
    def __init__(self, temperature, beta, omega, reorg, relax_rate, t_fixed, time_array, n_max):
        """
        initilize bath 

        parameters
        temperature: temp. of environment (float)
        omega: frequencies of bath (ndarray)
        """
        self.temperature = temperature    
        self.omega = omega
        self.reorg = reorg
        self.relax_rate = relax_rate
        self.t_fixed = t_fixed 
        self.beta = beta
        self.time_array = time_array
        self.n_max = n_max
    
    def plot_spectral_density(self, omega_values):

        spectral_density = (2 * self.reorg * self.relax_rate * omega_values)/((omega_values)**2 + self.relax_rate**2)

        x = omega_values/self.relax_rate

        plt.figure(figsize=(10,6))
        plt.plot(x.value, (spectral_density/self.relax_rate).value, label='spectral density', color='green')
        plt.xlabel('Frequency $\omega$ (eV)')
        plt.ylabel('J($\omega$) (eV) ')
        plt.legend()
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.show()
        return
    
    def plot_spectral_density2(self):
        # want x = omega / gamma to be  dimensionless
        x = np.linspace(0,6,500) 
        omega = x * self.relax_rate     # which has the unit eV

        spectral_density = (2 * self.reorg * self.relax_rate * omega)/(omega**2 + self.relax_rate**2)

        plt.figure(figsize=(8,5))
        plt.plot(x, (spectral_density/self.relax_rate).value, label='OBO', color='green')
        plt.ylabel("$C$''$(\omega) / \gamma$ ", fontsize=15)
        plt.xlabel('$\omega / \gamma$ ', fontsize=15)
        plt.legend(fontsize=13)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)        
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.show()
        return

    def g_dot_dot_sum(self, t, plot, int):
        """ 
        calculate the correlation function (2nd time deri. of g) using Matsubara frequencies
        output in [ev^2]
        input: t (for integration), time_array (for plotting), plot = True/False, and int = True/False depending on what is wanted. (if plot=True, then input a time array)
        output: g_dot_dot(t) or plot of g_dot_dot(t) with unit ps^-2
        """

        factor = 2 * np.pi / (self.beta * const.hbar.to(u.eV * u.ps))
        # print("factor relax rate TEST",(factor**2 - eV_to_ps1(self.relax_rate)**2)**-1 )

        if int:            
            # only real part                                                                                
            C = ( self.reorg * self.relax_rate * np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) * cot((self.beta * self.relax_rate * 2**-1).value) ) # has unit [eV**2]

            # calculate the discrete sum up to n_max
            summation = nsum(lambda n: (np.exp(- n * factor * t * u.ps) * factor * n) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2), 1, self.n_max)  # has the unit [ps] 

            # remaining tail of summation is approximated by an integral
            def sum_tail(n):
                return ((np.exp(- n * factor * t * u.ps)  * factor * n) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2)).value # has unit [ps]

            tail, err = quad(sum_tail, self.n_max, np.inf) # get integration warning; integral probably divergent or slowly convergent

            total_sum = summation.sum + tail 
            # print(total_sum, "total summation") # is practically zero

            return const.hbar.to(u.eV * u.ps)**-2 * (C + (self.reorg * self.relax_rate * 4)/(self.beta * const.hbar.to(u.eV * u.ps)) * float(total_sum)*u.ps)
        C_values = []
        total_sum_list = []

        if plot:
            for t in self.time_array:
                C = - 1j * self.reorg * self.relax_rate * np.exp(- eV_to_ps1(self.relax_rate) * t) + self.reorg * self.relax_rate * np.exp(- eV_to_ps1(self.relax_rate) * t) * cot((self.beta * self.relax_rate / 2).value)
                
                # calculate the discrete sum up to n_max
                summation = nsum(lambda n: (np.exp(- n * factor * t )  * factor * n) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2), 1, self.n_max)
                
                def sum_tail(n):
                    return ((np.exp(- n * factor * t)  * factor * n) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2)).value # has unit * ps
                
                tail, err = quad(sum_tail, self.n_max, np.inf)
                
                total_sum = summation.sum + tail 
                # print(total_sum, "total summation")
            
                C_values.append( (const.hbar.to(u.eV * u.ps)**-2 * (C + (self.reorg * self.relax_rate * 4)/(self.beta * const.hbar.to(u.eV * u.ps)) * float(total_sum)*u.ps)).value)        # nsum removes the units !
                total_sum_list.append(total_sum)
            print(np.sum(total_sum_list), "total sum")
            print(total_sum_list[0], "0 element in total sum")
            print(total_sum_list.index(max(total_sum_list)),"index of max of list")
            print(max(total_sum_list),"max of list")

            print(max(C_values),"max of list C_val")


            plt.figure(figsize=(12, 10))
            plt.plot(self.time_array, np.real(C_values), label=r"Re[C(t)]", color='mediumblue')
            # plt.plot(self.time_array, np.imag(C_values), label=r"Im[C(t)]", linestyle='dashed', color='royalblue')
            plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
            plt.ylabel(r"Correlation Function, $[$ps$^{-2}]$", fontsize=28)
            plt.legend(fontsize=26)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.title("Correlation Function, Full Expression", fontsize=32)
            plt.grid()
            plt.subplots_adjust(left=0.149, right=0.98)
            plt.show() 

        return

    def g_dot_sum(self, t, plot, int):
        """ 
        calculate the correlation function (2nd time deri. of g) in the using Matsubara frequencies
        input: t (for integration), time_array (for plotting), plot = True/False, and int = True/False depending on what is wanted. (if plot=True, then input a time array)
        output: g_dot(t) or plot of g_dot(t) with unit ps^-1
        """        
        factor = 2 * np.pi / (self.beta * const.hbar.to(u.eV * u.ps))

        if int:
           # calculate the discrete sum up to n_max
            summation = nsum(lambda n: (1 - np.exp(- n * factor * t * u.ps)) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2), 1, self.n_max)    # summation has unit [ps**2]

            # remaining tail of summation is approximated by an integral
            def sum_tail(n):
                return (1 - (np.exp(- n * factor * t * u.ps)) ) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2).value # has unit [ps**2]

            tail, err = quad(sum_tail, self.n_max, np.inf)

            total_sum = summation.sum + tail #* u.ps**2

            # part of g_dot(t) w/o Matsubara summation
            g_dot = 1j * eV_to_ps1(self.reorg) * (np.exp(-eV_to_ps1(self.relax_rate) * t * u.ps) - 1) + eV_to_ps1(self.reorg) * cot((self.beta * self.relax_rate * 2**-1).value) * (1 - np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps)) # unit [ps**-1]

            return g_dot + (4 * eV_to_ps1(self.reorg) * eV_to_ps1(self.relax_rate))/(self.beta * const.hbar.to(u.eV * u.ps)) * total_sum * u.ps**2 # nsum removes the unit of summation
        
        g_dot_values = []
                                                                                                           
        if plot:
            for t in self.time_array:
                summation = nsum(lambda n: (1 - np.exp(- n * factor * t)) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2), 1, self.n_max)


                # remaining tail of summation is approximated by an integral
                def sum_tail(n):
                    return (np.exp(- n * factor * t) * ((n * factor) / ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2))).value # has unit eV**2 * ps

                tail, err = quad(sum_tail, self.n_max, np.inf,limit=300)

                total_sum = summation.sum + tail 
                # print(total_sum, "total sum")

                # part of g_dot(t) w/o Matsubara summation
                g_dot = 1j * eV_to_ps1(self.reorg) * (np.exp(-eV_to_ps1(self.relax_rate) * t) - 1) + eV_to_ps1(self.reorg) * cot((self.beta * self.relax_rate * 2**-1).value) * (1 - np.exp(- eV_to_ps1(self.relax_rate) * t))

                g_dot_values.append(g_dot + (eV_to_ps1(self.reorg) * eV_to_ps1(self.relax_rate) * 4)/(self.beta * const.hbar.to(u.eV * u.ps)) * total_sum * u.ps**2) # nsum removes the unit
            
            plt.figure(figsize=(12, 10))
            plt.plot(self.time_array, np.real(g_dot_values), label=r"Re$[\dot{g}(t)]$", color='firebrick')
            plt.plot(self.time_array, np.imag(g_dot_values), label=r"Im$[\dot{g}(t)]$", linestyle='dashed', color='maroon')
            plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
            plt.ylabel(r"Time Derivative of $g(t)$, $[$ps$^{-1}]$", fontsize=28)
            plt.legend(fontsize=26)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.title(r"Time Derivative of $g(t)$, Full Expression", fontsize=32)
            plt.grid()
            plt.subplots_adjust(left=0.149, right=0.98)
            plt.show()
                                                                                                              # unit for summation is ps^2. nsum removes it
    
    def g_sum(self, t, plot, exp_plt, int):
        """
        calculate the line broadening function using Eq. 8.48 - with Matsubara frequencies
        input: t (for integration), time_array (for plotting), plot = True/False, exp_plt = True/False and int = True/False depending on what is wanted. (if plot=True, then input a time array)
        output: g(t) or plots of g(t) og exp(-g(t)) (unitless)
        """
        factor = 2 * np.pi / (self.beta * const.hbar.to(u.eV * u.ps))
        
        if int:    
            # calculate the discrete sum up to n_max
            summation = nsum(lambda n: (np.exp(- n * factor * t * u.ps) + factor * n * t * u.ps - 1) / (factor * n * ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2)), 1, self.n_max) # has unit [ps**3], nsum removes it

            # remaining tail of summation is approximated by an integral
            def sum_tail(n):
                return ((np.exp(- n * factor * t * u.ps) + factor * n * t * u.ps - 1) / (factor * n * ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2))).value # has unit [ps**3]

            tail, err = quad(sum_tail, self.n_max, np.inf,limit=300)

            total_sum = summation.sum + tail 
            g = -1j * self.reorg * self.relax_rate**-1 * (np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) + eV_to_ps1(self.relax_rate) * t * u.ps - 1) +  self.reorg * self.relax_rate**-1 *\
                  cot((self.beta * self.relax_rate * 2**-1).value) * (np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) + eV_to_ps1(self.relax_rate) * t * u.ps - 1)
            
            return g + (4 * eV_to_ps1(self.reorg) * eV_to_ps1(self.relax_rate))/(self.beta * const.hbar.to(u.eV * u.ps)) * total_sum * u.ps**3

        g_values = []
        exp_g_values = []

        if plot:
            for t in self.time_array:
                summation = nsum(lambda n: (np.exp(- n * factor * t) + factor * n * t - 1) / (factor * n * ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2)), 1, self.n_max) # has unit [ps**3]

                # remaining tail of summation is approximated by an integral
                def sum_tail(n):
                    return ((np.exp(- n * factor * t) + factor * n * t - 1) / (factor * n * ((n * factor)**2 - eV_to_ps1(self.relax_rate)**2))).value # has unit [ps**3]

                tail, err = quad(sum_tail, self.n_max, np.inf,limit=300)
                total_sum = summation.sum + tail 
                # print(total_sum, "total summation")

                g = -1j * self.reorg * self.relax_rate**-1 * (np.exp(- eV_to_ps1(self.relax_rate) * t) + eV_to_ps1(self.relax_rate) * t - 1) +  self.reorg * self.relax_rate**-1 *\
                  cot((self.beta * self.relax_rate * 2**-1).value) * (np.exp(- eV_to_ps1(self.relax_rate) * t) + eV_to_ps1(self.relax_rate) * t - 1)
                
                g_t = g + (4 * eV_to_ps1(self.reorg) * eV_to_ps1(self.relax_rate))/(self.beta * const.hbar.to(u.eV * u.ps)) * total_sum * u.ps**3
                g_values.append(g_t)
                exp_g_values.append(np.exp(-g_t))
                                                                                # multiply with unit for summation 
            plt.figure(figsize=(12, 10))
            plt.plot(self.time_array, np.real(g_values), label=r"Re$[g(t)]$", color = 'darkolivegreen')
            plt.plot(self.time_array, np.imag(g_values), label=r"Im$[g(t)]$", linestyle='dashed', color = 'olivedrab')
            plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
            plt.ylabel(r"g(t)", fontsize=28)
            plt.legend(fontsize=26)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)            
            plt.title(r"Line Broadening function, Full expression", fontsize=32)
            plt.subplots_adjust(left=0.149, right=0.98)
            plt.grid()
            plt.show() 

            if exp_plt:
                plt.figure(figsize=(12,10))
                plt.plot(self.time_array, np.real(exp_g_values), label=r"Re$[\exp(-g(t))]$", color='peru')
                plt.plot(self.time_array, np.imag(exp_g_values), label= r"Im$[\exp(-g(t))]$", linestyle='dashed', color = 'saddlebrown')
                plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
                plt.ylabel(r"exp(-g(t))", fontsize=28)
                plt.legend(fontsize=26)
                plt.xticks(fontsize=26)
                plt.yticks(fontsize=26)
                plt.title(r"$\exp(-g(t))$, Full Expression for $g(t)$", fontsize=32)
                plt.subplots_adjust(left=0.149, right=0.98)
                plt.grid()
                plt.show()
        return None

    def g_dot_dot_HT(self, t, plot, int):
        """
        calculate the correlation function (2nd time deri. of g) in the high temperature limit
        input: t (for integration), time_array (for plotting), plot = True/False, and int = True/False depending on what is wanted. (if plot=True, then input a time array)
        output: g_dot_dot(t) or plot of g_dot_dot(t) with unit ps^-2
        """
        factor = 2 * np.pi / (self.beta * const.hbar.to(u.eV * u.ps))

        if int:                                                                                         
            #keeping relax rate and reorg as energies and multi. w hbar^-2         
            # C = (const.hbar.to(u.eV * u.ps))**-2 * (2 * self.beta**-1 *  self.reorg * np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) - 1j * self.reorg * self.relax_rate * np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) )
            # only Re[C(t)]
            C = (const.hbar.to(u.eV * u.ps))**-2 * 2 * self.beta**-1 *  self.reorg * np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) 

        if plot:
            C = (const.hbar.to(u.eV * u.ps))**-2 * (2 * self.beta**-1 *  self.reorg * np.exp(- eV_to_ps1(self.relax_rate) * self.time_array) - 1j * self.reorg * self.relax_rate * np.exp(- eV_to_ps1(self.relax_rate) * self.time_array)) 

            plt.figure(figsize=(12, 10))
            plt.plot(self.time_array, np.real(C), label=r"Re[C(t)]", color='cornflowerblue')
            plt.plot(self.time_array, np.imag(C), label=r"Im[C(t)]", linestyle='dashed')
            plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
            plt.ylabel(r"Correlation Function, $[$ps$^{-2}]$", fontsize=28)
            plt.legend(fontsize=26, loc='upper right')
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.title("Correlation Function, HT Limit Expression", fontsize=32)
            plt.grid()
            plt.subplots_adjust(left=0.149, right=0.98)
            plt.show() 

        return C

    def g_dot_HT(self, t, plot, int):
        """
        calculate the time derivative of the line broadening function in the high temperature limit
        input: t (for integration), time_array (for plotting), plot = True/False, and int = True/False depending on what is wanted. (if plot=True, then input a time array)
        output: g_dot(t) or plot of g_dot(t) with unit ps^-1
        """
        if int:
            g_dot = 2 * self.beta**-1 * self.reorg * self.relax_rate**-1 * (const.hbar.to(u.eV * u.ps))**-1 * (1 - np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps ))  + 1j * eV_to_ps1(self.reorg) * (np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps ) - 1)

        if plot:
            g_dot = 2 * self.beta**-1 * self.reorg * self.relax_rate**-1 * (const.hbar.to(u.eV * u.ps))**-1 * (1 - np.exp(- eV_to_ps1(self.relax_rate) * self.time_array)) + 1j * eV_to_ps1(self.reorg) * (np.exp(- eV_to_ps1(self.relax_rate) * self.time_array) - 1)
            plt.figure(figsize=(12, 10))
            plt.plot(self.time_array, np.real(g_dot), label=r"Re$[\dot{g}(t)]$", color='lightcoral')
            plt.plot(self.time_array, np.imag(g_dot), label=r"Im$[\dot{g}(t)]$", linestyle='dashed', color='indianred')
            plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
            plt.ylabel(r"Time Derivative of $g(t)$, $[$ps$^{-1}]$", fontsize=28)
            plt.legend(fontsize=26)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.title(r"Time Derivative of $g(t)$, HT Limit Expression", fontsize=32)
            plt.grid()
            plt.subplots_adjust(left=0.149, right=0.98)
            plt.show()
        return g_dot

    def g_HT(self, t, plot, exp_plt, int):
        """
        calculate the line broadening function in the high temperature limit
        input: t (for integration), time_array (for plotting), plot = True/False, exp_plt = True/False and int = True/False depending on what is wanted. (if plot=True, then input a time array)
        output: g(t) or plots of g(t) og exp(-g(t)) (unitless)
        """
        if int:
            g = 2 * self.beta**-1 * self.reorg * self.relax_rate**-2 * (np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) + eV_to_ps1(self.relax_rate) * t * u.ps - 1) - 1j * self.reorg * self.relax_rate**-1 * \
                  (np.exp(- eV_to_ps1(self.relax_rate) * t * u.ps) + eV_to_ps1(self.relax_rate) * t * u.ps - 1)
            return g

        if plot:
            g = 2 * self.beta**-1 * self.reorg * self.relax_rate**-2 * (np.exp(- eV_to_ps1(self.relax_rate) * self.time_array) + eV_to_ps1(self.relax_rate) * self.time_array - 1) - 1j * self.reorg * self.relax_rate**-1 * \
                  (np.exp(- eV_to_ps1(self.relax_rate) * self.time_array) + eV_to_ps1(self.relax_rate) * self.time_array - 1)
            
            plt.figure(figsize=(12, 10))
            plt.plot(self.time_array, np.real(g), label=r"Re$[g(t)]$", color = 'olive')
            plt.plot(self.time_array, np.imag(g), label=r"Im$[g(t)]$", linestyle='dashed', color = 'yellowgreen')
            plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
            plt.ylabel(r"g(t)", fontsize=28)
            plt.legend(fontsize=26)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)            
            plt.title(r"Line Broadening function, HT Limit Expression", fontsize=32)
            plt.grid()
            plt.subplots_adjust(left=0.149, right=0.98)
            plt.show()

            if exp_plt:
                plt.figure(figsize=(12,10))
                plt.plot(self.time_array, np.real(np.exp(-g)), label=r"Re$[\exp(-g(t))]$", color='orange')
                plt.plot(self.time_array, np.imag(np.exp(-g)), label= r"Im$[\exp(-g(t))]$", linestyle='dashed', color = 'goldenrod')
                plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
                plt.ylabel(r"exp(-g(t))", fontsize=28)
                plt.legend(fontsize=26)
                plt.xticks(fontsize=26)
                plt.yticks(fontsize=26)
                plt.title(r"$\exp(-g(t))$, HT Limit Expression for $g(t)$", fontsize=32)
                plt.grid()
                plt.subplots_adjust(left=0.149, right=0.98)
                plt.show()

    @staticmethod
    def initial_population(dim, population_type):
        """
        get initial population of system
        input: dimensions of Hamiltonian, the population type (ground state p[0,0], first state p[1,0] and second [2,0])
        output: population vector
        """

        p = np.zeros((dim,1))

        if population_type == "ground":
            p[0,0] = 1
        
        elif population_type == "first":
            p[1,0] = 1

        elif population_type == "second":
            p[2,0] =1

        return p

    def memory_kernel(self, Hamiltonian, coef_tensor, t_min, t_max, HT):
        """
        calculate memory kernel for population simulation
        input:
        output:
        """
        dim = Hamiltonian.shape[0]
        kernel = np.zeros((Hamiltonian.shape))
        error_matrix = np.zeros((Hamiltonian.shape))

        def integrand(t, initial, final):
            """
            Return the complex integrand for the transition from 'initial' to 'final'.
            """
            if HT == True:
                g_t = self.g_HT(t, plot=False, exp_plt=False, int=True)
                g_dot_t = self.g_dot_HT(t,plot=False,int=True) 
                g_dot_dot_t = self.g_dot_dot_HT(t,plot=False,int=True)
            if HT == False:
                g_t = self.g_sum(t, plot=False, exp_plt=False, int=True)
                g_dot_t = self.g_dot_sum(t,plot=False,int=True) 
                g_dot_dot_t = self.g_dot_dot_sum(t,plot=False,int=True)

            E_i = (Hamiltonian[initial, initial] * u.eV - self.reorg * coef_tensor[initial, initial, initial, initial])
            E_f = (Hamiltonian[final, final]   * u.eV - self.reorg * coef_tensor[final, final, final, final])

            energy_diff = (E_i - E_f) * (const.hbar.to(u.eV * u.ps))**-1

            exp_1 = np.exp(1j * energy_diff * t * u.ps)  

            reorg_i = self.reorg * coef_tensor[initial, initial, initial, initial]
            reorg_f = self.reorg * coef_tensor[final, final, initial, initial]

            exp_2 = np.exp(- g_t * coef_tensor[initial, initial, initial, initial] - g_t * coef_tensor[final, final, final, final] + 2 * g_t * coef_tensor[initial, initial, final, final] - 2j * eV_to_ps1(reorg_i - reorg_f) * t * u.ps)
            curly_1 = g_dot_t * coef_tensor[initial, initial, initial, final] + g_dot_t * coef_tensor[initial, final, final, final] + 2j * eV_to_ps1(self.reorg) * coef_tensor[initial, final, initial, initial]
            curly_2 = np.conj(g_dot_t) * coef_tensor[initial, initial, final, initial] - g_dot_t * coef_tensor[final, final, final, initial] - 2j * eV_to_ps1(self.reorg) * coef_tensor[initial, initial, final, initial]

            integrand_value = (exp_1 * exp_2 * ( g_dot_dot_t * coef_tensor[initial, final, final, initial] + curly_1 * curly_2 ).value)
            
            return integrand_value, g_dot_dot_t * coef_tensor[initial, final, final, initial] + curly_1 * curly_2
        

        for from_state in range(dim):        # column index
            for to_state in range(dim):      # row index
                if to_state != from_state:
                    # computing the integral that yields K[to_state, from_state]
                    integral_val_r, error_val_r = quad(lambda t: np.real(integrand(t, from_state, to_state)[0]), t_min, t_max, limit=1000, epsabs=1e-12, epsrel=1e-10)
                    integral_val_i, error_val_i = quad(lambda t: np.imag(integrand(t, from_state, to_state)[0]), t_min, t_max, limit=1000, epsabs=1e-12, epsrel=1e-10)
                    integral_val = integral_val_r + 1j * integral_val_i
                    kernel[to_state, from_state] = 2 * np.real(integral_val)
                    error_matrix[to_state, from_state] = error_val_r + error_val_i
                    if np.real(integral_val) < 0:
                        print(from_state, to_state, "from_state -> to_state that cause negative ME's for kernel")
                        # kernel[to_state, from_state] = np.abs(kernel[to_state, from_state])       # trying to assign negative off-diag. elements to their absolute value

            # after all off-diagonals in this column are set:
            kernel[from_state, from_state] = -np.sum(kernel[:, from_state])
       
        return kernel  #* u.ps**-1
    
    def memory_kernel_osc(self, Hamiltonian, coef_tensor, t_min, t_max, HT):
        """
        calculate memory kernel for population simulation using a specialized oscillatory integration to handle the highly oscillatory exp_1 factor.
        input: hamiltonian, coef_tensor, limits for integration and True/False boolean for HT
        output: memory kernel (matrix with same dimensions as the Hamiltonian)
        """
        
        dim = Hamiltonian.shape[0]
        kernel = np.zeros((dim, dim))
        error_matrix = np.zeros((dim, dim))

        def oscillatory_integral(f, a, b, omega, **kwargs):
            """
            compute I = int_a^b dt f(t) e^(i ω t) using SciPy weighted integration
            where f is a (possibly complex) relatively smooth function
            
            the actual integral is
            int = int_a^b dt f(t) cos(ωt) dt + i int_a^b dt f(t) sin(ωt) 
            but using weight option of the routine quad to evaluate the two integrals
            """
            # because f is complex, write f(t) = f_real(t) + i f_imag(t).
            f_real = lambda t: np.real(f(t))
            f_imag = lambda t: np.imag(f(t))
            
            # unsing SciPy's weighted integration
            # int dt f(t) cos(ωt) = int dt A(t) cos(ωt) - int dt B(t) sin(ωt)
            I_cos_f_real, err1 = quad(f_real, a, b, weight='cos', wvar=omega, **kwargs)
            I_sin_f_imag, err2 = quad(f_imag, a, b, weight='sin', wvar=omega, **kwargs)
            I_real = I_cos_f_real - I_sin_f_imag
            
            # and equivalently int dt f(t) sin(ωt) = int dt f_real(t) sin(ωt) + int dt f_imag(t) cos(ωt)
            I_sin_f_real, err3 = quad(f_real, a, b, weight='sin', wvar=omega, **kwargs)     # ω is the variable using for the weighted integration. energy/freq. differences are the input (later in code)
            I_cos_f_imag, err4 = quad(f_imag, a, b, weight='cos', wvar=omega, **kwargs)
            I_imag = I_sin_f_real + I_cos_f_imag
            
            # collecting the integration terms
            I = I_real + 1j * I_imag
            err_total = err1 + err2 + err3 + err4
            return I, err_total

        for from_state in range(dim):  # column index
            for to_state in range(dim):  # row index
                if to_state != from_state:
                    # befine the integrand in two parts: factor out the problematic exp_1.
                    if HT:
                        g_t = self.g_HT
                        g_dot_t = self.g_dot_HT
                        g_dot_dot_t = self.g_dot_dot_HT
                    else:
                        g_t = self.g_sum
                        g_dot_t = self.g_dot_sum
                        g_dot_dot_t = self.g_dot_dot_sum

                    E_i = Hamiltonian[from_state, from_state] * u.eV - self.reorg * coef_tensor[from_state, from_state, from_state, from_state]
                    E_f = Hamiltonian[to_state, to_state]     * u.eV - self.reorg * coef_tensor[to_state, to_state, to_state, to_state]
                    energy_diff = (E_i - E_f) * (const.hbar.to(u.eV * u.ps))**-1
                    # frequency is used for the oscillatory part
                    omega = float(energy_diff * u.ps)  # ensure omega is a plain float

                    reorg_i = self.reorg * coef_tensor[from_state, from_state, from_state, from_state]
                    reorg_f = self.reorg * coef_tensor[to_state, to_state, from_state, from_state]
                    
                    # now the smooth function f(t) is defined (the easy part of the integrand)
                    def f(t):
                        g_val = g_t(t, plot=False, exp_plt=False, int=True)
                        g_dot_val = g_dot_t(t, plot=False, int=True)
                        g_dot_dot_val = g_dot_dot_t(t, plot=False, int=True)
                        
                        exp_2 = np.exp(- g_val * coef_tensor[from_state, from_state, from_state, from_state] - g_val * coef_tensor[to_state, to_state, to_state, to_state] + 2 * g_val * coef_tensor[from_state, from_state, to_state, to_state] - 2j * eV_to_ps1(reorg_i - reorg_f) * t * u.ps)
                
                        curly_1 = (g_dot_val * coef_tensor[from_state, from_state, from_state, to_state] + g_dot_val * coef_tensor[from_state, to_state, to_state, to_state] + 2j * eV_to_ps1(self.reorg) * coef_tensor[from_state, to_state, from_state, from_state])
                        
                        curly_2 = (np.conj(g_dot_val) * coef_tensor[from_state, from_state, to_state, from_state] - g_dot_val * coef_tensor[to_state, to_state, to_state, from_state] - 2j * eV_to_ps1(self.reorg) * coef_tensor[from_state, from_state, to_state, from_state])
                        
                        F = exp_2 * ( (g_dot_dot_val * coef_tensor[from_state, to_state, to_state, from_state] + curly_1 * curly_2).value )
                        return F  # the non-oscillatory part

                    # full integrand is then f(t)*exp(1j*omega*t), use our oscillatory_integral routine with the defined omega and input the non-oscillatory function, f.
                    I, err = oscillatory_integral(lambda t: f(t), t_min, t_max, omega, limit=1000, epsabs=1e-12, epsrel=1e-10)
                    # multiply the real part of the integral by 2, cf. the definition 
                    integral_val = 2 * np.real(I)       
                    if integral_val < 0:
                        print(from_state, to_state, "from_state -> to_state that cause negative ME's for kernel")
                        # integral_val = np.conj(integral_val)
                        # integral_val = np.abs(integral_val)
                    kernel[to_state, from_state] = integral_val
                    error_matrix[to_state, from_state] = err

            # after computing off-diagonals for this state, determine diagonal element (for population conservation)
            kernel[from_state, from_state] = -np.sum(kernel[:, from_state])
        
        return kernel#, error_matrix

    def memory_kernel_YF02(self, Hamiltonian, coef_tensor, t_min, t_max, HT):
        """
        calculate memory kernel for population simulation
        input:
        output:
        """
        dim = Hamiltonian.shape[0]
        kernel = np.zeros((Hamiltonian.shape))
        error_matrix = np.zeros((Hamiltonian.shape))

        def integrand(t, initial, final):

            if HT == True:
                g_t = self.g_HT(t, plot=False, exp_plt=False, int=True)
                g_dot_t = self.g_dot_HT(t,plot=False,int=True) 
                g_dot_dot_t = self.g_dot_dot_HT(t,plot=False,int=True)
            if HT == False:
                g_t = self.g_sum(t, plot=False, exp_plt=False, int=True)
                g_dot_t = self.g_dot_sum(t,plot=False,int=True) 
                g_dot_dot_t = self.g_dot_dot_sum(t,plot=False,int=True)
           
            energy_diff = eV_to_ps1((Hamiltonian[initial, initial] * u.eV - self.reorg * coef_tensor[initial, initial, initial, initial]) - (Hamiltonian[final, final] * u.eV - self.reorg * coef_tensor[final, final, final, final])) # defined in ps^-1
            # first exponential function in the memory kernel
            exp_1 = np.exp(1j * energy_diff * t * u.ps)        
            reorg_diff = 2 * self.reorg * coef_tensor[final, final, initial, initial] - (self.reorg * coef_tensor[initial, initial, initial, initial] - self.reorg * coef_tensor[final, final, final, final])
            # second exponential function in the memory kernel
            exp_2 = np.exp(- g_t * coef_tensor[initial, initial, initial, initial] - g_t * coef_tensor[final, final, final, final] + 2 * g_t * coef_tensor[initial, initial, final, final] + 1j * eV_to_ps1(reorg_diff) * t * u.ps )
            # first curly bracket                                                                                                                   
            curly_1 = g_dot_t * coef_tensor[initial, final, final, final] - g_dot_t * coef_tensor[initial, final, initial, initial] + 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[initial, final, initial, initial] 
            # second curly bracket 
            curly_2 = g_dot_t * coef_tensor[final, initial, final, final] - g_dot_t * coef_tensor[final, initial, initial, initial] - 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[final, initial, initial, initial] 

            integrand = exp_1 * exp_2 * ( g_dot_dot_t * coef_tensor[initial, final, final, initial] + curly_1 * curly_2).value 
            # integrand has the unit ps^-2                                                           # product of caruly_1 and curly_2 has the unit ps^-2

            return integrand

        for initial in range(dim):          # column
            for final in range(dim):      # row
                if initial != final: 
                    integral_real, error_real = quad(lambda t: np.real(integrand(t, initial, final)), t_min, t_max, limit=200, epsabs=1e-8, epsrel=1e-6)
                    # integral_imag, error_imag = quad(lambda t: np.imag(integrand(t, e_n, e_m)), t_min, t_max, limit=200, epsabs=1e-8, epsrel=1e-6)
                    kernel[final, initial] = 2 * integral_real
                    error_matrix[final, initial] = error_real #+ error_imag
                    if integral_real < 0:
                        print(initial, final, "from_state -> to_state that cause negative ME's for kernel")
                        # kernel[final, initial] = np.abs(kernel[final, initial])         # trying to assign negative off-diagonal elements to positive values ..
            kernel[initial, initial] = -np.sum(kernel[:, initial])
    
        print("error for time integration: \n", error_matrix)
        return kernel  #* u.ps**-1

    def memory_kernel_osc_YF02(self, Hamiltonian, coef_tensor, t_min, t_max, HT):
        """
        calculate memory kernel using expression from Yang&Flemming 02 for population simulation using a specialized oscillatory integration to handle the highly oscillatory exp_1 factor.
        input: hamiltonian, coef_tensor, limits for integration and True/False boolean for HT
        output: memory kernel (matrix with same dimensions as the Hamiltonian)
        """
        
        dim = Hamiltonian.shape[0]
        kernel = np.zeros((dim, dim))
        error_matrix = np.zeros((dim, dim))

        def oscillatory_integral(f, a, b, omega, **kwargs):
            """
            compute I = int_a^b dt f(t) e^(i ω t) using SciPy weighted integration
            where f is a (possibly complex) relatively smooth function
            
            the actual integral is
            int = int_a^b dt f(t) cos(ωt) dt + i int_a^b dt f(t) sin(ωt) 
            but using weight option of the routine quad to evaluate the two integrals
            """
            # because f is complex, write f(t) = f_real(t) + i f_imag(t).
            f_real = lambda t: np.real(f(t))
            f_imag = lambda t: np.imag(f(t))
            
            # unsing SciPy's weighted integration
            # int dt f(t) cos(ωt) = int dt A(t) cos(ωt) - int dt B(t) sin(ωt)
            I_cos_f_real, err1 = quad(f_real, a, b, weight='cos', wvar=omega, **kwargs)
            I_sin_imag, err2 = quad(f_imag, a, b, weight='sin', wvar=omega, **kwargs)
            I_real = I_cos_f_real - I_sin_imag
            
            # and equivalently int dt f(t) sin(ωt) = int dt f_real(t) sin(ωt) + int dt f_imag(t) cos(ωt)
            I_sin_f_real, err3 = quad(f_real, a, b, weight='sin', wvar=omega, **kwargs)     # ω is the variable using for the weighted integration. energy/freq. differences are the input (later in code)
            I_cos_f_imag, err4 = quad(f_imag, a, b, weight='cos', wvar=omega, **kwargs)
            I_imag = I_sin_f_real + I_cos_f_imag
            
            # collecting the integration terms
            I = I_real + 1j * I_imag
            err_total = err1 + err2 + err3 + err4
            return I, err_total

        for from_state in range(dim):  # column index
            for to_state in range(dim):  # row index
                if to_state != from_state:
                    # befine the integrand in two parts: factor out the problematic exp_1.
                    if HT:
                        g_t = self.g_HT
                        g_dot_t = self.g_dot_HT
                        g_dot_dot_t = self.g_dot_dot_HT
                    else:
                        g_t = self.g_sum
                        g_dot_t = self.g_dot_sum
                        g_dot_dot_t = self.g_dot_dot_sum

                    E_i = Hamiltonian[from_state, from_state] * u.eV - self.reorg * coef_tensor[from_state, from_state, from_state, from_state]
                    E_f = Hamiltonian[to_state, to_state]     * u.eV - self.reorg * coef_tensor[to_state, to_state, to_state, to_state]
                    energy_diff = (E_i - E_f) * (const.hbar.to(u.eV * u.ps))**-1
                    # frequency is used for the oscillatory part
                    omega = float(energy_diff * u.ps)  # ensure omega is a plain float

                    # now the smooth function f(t) is defined (the easy part of the integrand)
                    def f(t):
                        g_val = g_t(t, plot=False, exp_plt=False, int=True)
                        g_dot_val = g_dot_t(t, plot=False, int=True)
                        g_dot_dot_val = g_dot_dot_t(t, plot=False, int=True)
                        
                        reorg_diff = self.reorg * coef_tensor[to_state, to_state, to_state, to_state] + self.reorg * coef_tensor[from_state, from_state, from_state, from_state] - 2 * self.reorg * coef_tensor[to_state, to_state, from_state, from_state]
                        # second exponential function in the memory kernel
                        exp_2 = np.exp(- g_val * coef_tensor[from_state, from_state, from_state, from_state] - g_val * coef_tensor[to_state, to_state, to_state, to_state] + 2 * g_val * coef_tensor[from_state, from_state, to_state, to_state] - 1j * eV_to_ps1(reorg_diff) * t * u.ps )
                        # first curly bracket                                                                                                                   
                        curly_1 = g_dot_val * coef_tensor[from_state, to_state, to_state, to_state] - g_dot_val * coef_tensor[from_state, to_state, from_state, from_state] + 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[from_state, to_state, from_state, from_state] 
                        # second curly bracket 
                        curly_2 = g_dot_val * coef_tensor[to_state, from_state, to_state, to_state] - g_dot_val * coef_tensor[to_state, from_state, from_state, from_state] - 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[to_state, from_state, from_state, from_state] 
                        
                        F = exp_2 * ( (g_dot_dot_val * coef_tensor[from_state, to_state, to_state, from_state] + curly_1 * curly_2).value )
                        return F  # the non-oscillatory part

                    # full integrand is then f(t)*exp(1j*omega*t), use our oscillatory_integral routine with the defined omega and input the non-oscillatory function, f.
                    I, err = oscillatory_integral(lambda t: f(t), t_min, t_max, omega, limit=1000, epsabs=1e-12, epsrel=1e-10)
                    # multiply the real part of the integral by 2, cf. the definition 
                    integral_val = 2 * np.real(I)       
                    if integral_val < 0:
                        print(from_state, to_state, "from_state -> to_state that cause negative ME's for kernel")
                        # integral_val = np.conj(integral_val)
                        # integral_val = np.abs(integral_val)
                    kernel[to_state, from_state] = integral_val
                    error_matrix[to_state, from_state] = err

            # after computing off-diagonals for this state, determine diagonal element (for population conservation)
            kernel[from_state, from_state] = -np.sum(kernel[:, from_state])
        
        return kernel#, error_matrix

    def memory_kernel_CM97(self, Hamiltonian, coef_tensor, t_min, t_max, HT):
        """
        calculate memory kernel for population simulation using the expression from Zhang et al (CM97) 
        input:
        output:
        """
        dim = Hamiltonian.shape[0]
        kernel = np.zeros((Hamiltonian.shape))
        error_matrix = np.zeros((Hamiltonian.shape))

        def integrand(t, initial, final):

            if HT == True:
                g_t = self.g_HT(t, plot=False, exp_plt=False, int=True)
                g_dot_t = self.g_dot_HT(t,plot=False,int=True) 
                g_dot_dot_t = self.g_dot_dot_HT(t,plot=False,int=True)
            if HT == False:
                g_t = self.g_sum(t, plot=False, exp_plt=False, int=True)
                g_dot_t = self.g_dot_sum(t,plot=False,int=True) 
                g_dot_dot_t = self.g_dot_dot_sum(t,plot=False,int=True)

            energy_diff = eV_to_ps1((Hamiltonian[final, final] * u.eV - self.reorg * coef_tensor[final, final, final, final]) - (Hamiltonian[initial, initial] * u.eV - self.reorg * coef_tensor[initial, initial, initial, initial])) # defined in ps^-1

            # first exponential function in the memory kernel
            exp_1 = np.exp(-1j * energy_diff * t * u.ps)        
            # reorg_diff = self.reorg * coef_tensor[e_n, e_n, e_n, e_n] - self.reorg * coef_tensor[e_m, e_m, e_n, e_n]
            reorg_diff = self.reorg * (coef_tensor[initial, initial, initial, initial] - coef_tensor[final, final, initial, initial])
            # second exponential function in the memory kernel
            exp_2 = np.exp(- g_t * coef_tensor[final, final, final, final] - g_t * coef_tensor[initial, initial, initial, initial] + 2 * g_t * coef_tensor[initial, initial, final, final] - 2 * 1j * eV_to_ps1(reorg_diff) * t * u.ps )
            # first curly bracket                                                                                                                   
            curly_1 = g_dot_t * coef_tensor[initial, final, initial, initial] - g_dot_t * coef_tensor[initial, final, final, final] + 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[initial, final, initial, initial] 
            # second curly bracket 
            curly_2 = g_dot_t * coef_tensor[initial, initial, final, initial] - g_dot_t * coef_tensor[final, final, final, initial] + 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[final, initial, initial, initial] 

            integrand = exp_1 * exp_2 * ( g_dot_dot_t * coef_tensor[final, initial, initial, final] - curly_1 * curly_2).value
            # integrand has the unit ps^-2                                                           # product of caruly_1 and curly_2 has the unit ps^-2
            return integrand

        for initial in range(dim):          # column
            for final in range(dim):      # row
                if initial != final: 
                    integral_real, error_real = quad(lambda t: np.real(integrand(t, initial, final)), t_min, t_max, limit=200, epsabs=1e-8, epsrel=1e-6)
                    # integral_imag, error_imag = quad(lambda t: np.imag(integrand(t, e_n, e_m)), t_min, t_max, limit=200, epsabs=1e-8, epsrel=1e-6)
                    kernel[final, initial] = 2 * integral_real
                    error_matrix[final, initial] = error_real #+ error_imag
                    if integral_real < 0:
                        print(initial, final, "from_state -> to_state that cause negative ME's for kernel")
                        # kernel[final, initial] = np.abs(kernel[final, initial])         # trying to assign negative off-diagonal elements to positive values ..
    
            kernel[initial, initial] = - np.sum(kernel[:, initial])
                # kernel has the unit ps^-1 (bc. integrand [ps^-2], integrate ; times ps)

        print("error for time integration: \n", error_matrix)
        return kernel  #* u.ps**-1

    def memory_kernel_osc_CM97(self, Hamiltonian, coef_tensor, t_min, t_max, HT):
        """
        calculate memory kernel using expression from Zhang&Mukamel97(98) for population simulation using a specialized oscillatory integration to handle the highly oscillatory exp_1 factor.
        input: hamiltonian, coef_tensor, limits for integration and True/False boolean for HT
        output: memory kernel (matrix with same dimensions as the Hamiltonian)
        """
        
        dim = Hamiltonian.shape[0]
        kernel = np.zeros((dim, dim))
        error_matrix = np.zeros((dim, dim))

        def oscillatory_integral(f, a, b, omega, **kwargs):
            """
            compute I = int_a^b dt f(t) e^(i ω t) using SciPy weighted integration
            where f is a (possibly complex) relatively smooth function
            
            the actual integral is
            int = int_a^b dt f(t) cos(ωt) dt + i int_a^b dt f(t) sin(ωt) 
            but using weight option of the routine quad to evaluate the two integrals
            """
            # because f is complex, write f(t) = f_real(t) + i f_imag(t).
            f_real = lambda t: np.real(f(t))
            f_imag = lambda t: np.imag(f(t))
            
            # unsing SciPy's weighted integration
            # int dt f(t) cos(ωt) = int dt A(t) cos(ωt) - int dt B(t) sin(ωt)
            I_cos_f_real, err1 = quad(f_real, a, b, weight='cos', wvar=omega, **kwargs)
            I_sin_f_imag, err2 = quad(f_imag, a, b, weight='sin', wvar=omega, **kwargs)
            I_real = I_cos_f_real - I_sin_f_imag
            
            # and equivalently int dt f(t) sin(ωt) = int dt f_real(t) sin(ωt) + int dt f_imag(t) cos(ωt)
            I_sin_f_real, err3 = quad(f_real, a, b, weight='sin', wvar=omega, **kwargs)     # ω is the variable using for the weighted integration. energy/freq. differences are the input (later in code)
            I_cos_f_imag, err4 = quad(f_imag, a, b, weight='cos', wvar=omega, **kwargs)
            I_imag = I_sin_f_real + I_cos_f_imag
            
            # collecting the integration terms
            I = I_real + 1j * I_imag
            err_total = err1 + err2 + err3 + err4
            return I, err_total

        for from_state in range(dim):  # column index
            for to_state in range(dim):  # row index
                if to_state != from_state:
                    # befine the integrand in two parts: factor out the problematic exp_1.
                    if HT:
                        g_t = self.g_HT
                        g_dot_t = self.g_dot_HT
                        g_dot_dot_t = self.g_dot_dot_HT
                    else:
                        g_t = self.g_sum
                        g_dot_t = self.g_dot_sum
                        g_dot_dot_t = self.g_dot_dot_sum

                    E_i = Hamiltonian[from_state, from_state] * u.eV - self.reorg * coef_tensor[from_state, from_state, from_state, from_state]
                    E_f = Hamiltonian[to_state, to_state]     * u.eV - self.reorg * coef_tensor[to_state, to_state, to_state, to_state]
                    energy_diff = (E_f - E_i) * (const.hbar.to(u.eV * u.ps))**-1
                    # frequency is used for the oscillatory part
                    omega = float(- energy_diff * u.ps)  # ensure omega is a plain float

                    # now the smooth function f(t) is defined (the easy part of the integrand)
                    def f(t):
                        g_val = g_t(t, plot=False, exp_plt=False, int=True)
                        g_dot_val = g_dot_t(t, plot=False, int=True)
                        g_dot_dot_val = g_dot_dot_t(t, plot=False, int=True)
                        
                        reorg_diff = self.reorg * (coef_tensor[from_state, from_state, from_state, from_state] - coef_tensor[to_state, to_state, from_state, from_state])
                        # second exponential function in the memory kernel
                        exp_2 = np.exp(- g_val * coef_tensor[to_state, to_state, to_state, to_state] - g_val * coef_tensor[from_state, from_state, from_state, from_state] + 2 * g_val * coef_tensor[from_state, from_state, to_state, to_state] - 2 * 1j * eV_to_ps1(reorg_diff) * t * u.ps )
                        # first curly bracket                                                                                                                   
                        curly_1 = g_dot_val * coef_tensor[from_state, to_state, from_state, from_state] - g_dot_val * coef_tensor[from_state, to_state, to_state, to_state] + 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[from_state, to_state, from_state, from_state] 
                        # second curly bracket 
                        curly_2 = g_dot_val * coef_tensor[from_state, from_state, to_state, from_state] - g_dot_val * coef_tensor[to_state, to_state, to_state, from_state] + 2 * 1j * eV_to_ps1(self.reorg) * coef_tensor[to_state, from_state, from_state, from_state] 

                        F = exp_2 * ( (g_dot_dot_val * coef_tensor[from_state, to_state, to_state, from_state] - curly_1 * curly_2).value )
                        return F  # the non-oscillatory part

                    # full integrand is then f(t)*exp(1j*omega*t), use our oscillatory_integral routine with the defined omega and input the non-oscillatory function, f.
                    I, err = oscillatory_integral(lambda t: f(t), t_min, t_max, omega, limit=1000, epsabs=1e-12, epsrel=1e-10)
                    # multiply the real part of the integral by 2, cf. the definition 
                    integral_val = 2 * np.real(I)       
                    if integral_val < 0:
                        print(from_state, to_state, "from_state -> to_state that cause negative ME's for kernel")
                    kernel[to_state, from_state] = integral_val
                    error_matrix[to_state, from_state] = err

            # after computing off-diagonals for this state, determine diagonal element (for population conservation)
            kernel[from_state, from_state] = -np.sum(kernel[:, from_state])
        
        return kernel#, error_matrix

    def pop_derivative(self, t, population, kernel):

        dim = kernel.shape[0]
        pop_deri = np.zeros(dim)       # define spacer for time derivative of the population
       
        population = population.flatten()
        for e_n in range(dim):
            gain = np.fromiter((kernel[e_n, e_m] * population[e_m] for e_m in range(dim) if e_m != e_n), dtype=float).sum() # fromiter creates an array with allocated date (here product of kernel MEs and population element) for iterable object
            loss = np.fromiter((kernel[e_m, e_n] * population[e_n] for e_m in range(dim) if e_m != e_n), dtype=float).sum()
            pop_deri[e_n] = gain - loss 
        
        return pop_deri

    def Hamiltonian(self, dim, diagonalize):
        """
        function that generates the system Hamiltonian depending on the dimension
        input: the dimension (row/col) of the Hamiltonian and diagonalize=True/False depending on what is wanted
        output: either the Hamiltonian in electronic basis or excitonic basis (where it has been diagonalized)
        """

        V_FW1 = 50.77 * 10**-3      #* 10       # [eV] electronic coupling for FW1
        V_FW2 = 15.16 * 10**-3      #* 10       # [eV] electronic coupling for FW1
        V_CR1 = 26.18 * 10**-3      #* 10       # [eV] electronic coupling for CR1
        V_CR2 = 0.04384 * 10**-3    #* 10       # [eV] electronic coupling for CR2
        eps_exc = 3.2432                   # [eV] energy of D-C*-A
        eps_part_CSS = 3.45672             # [eV] energy of D^+ - C^- - A
        eps_fully_CSS =  2.074666          # [eV] energy of D^+ - C - A^-

        H = np.zeros((dim,dim))

        if dim == 2:
            H[0,0] = eps_exc     
            H[1,1] = eps_part_CSS   
            H[0,1] = V_FW1      
            H[1,0] = V_FW1   


        if dim == 3:
            H[0,0] = eps_exc     
            H[1,1] = eps_part_CSS      
            H[2,2] = eps_fully_CSS     

            H[0,1] = V_FW1      
            H[1,0] = V_FW1     
            H[1,2] = V_FW2      
            H[2,1] = V_FW2     

        if dim == 4:        # ground state energy is zero
            # H[0,0] = 2
            H[1,1] = eps_exc     
            H[2,2] = eps_part_CSS      
            H[3,3] = eps_fully_CSS     

            H[1,2] = V_FW1      
            H[2,1] = V_FW1     
            H[2,3] = V_FW2      
            H[3,2] = V_FW2    
            H[0,2] = V_CR1
            H[2,0] = V_CR1
            H[0,3] = V_CR2
            H[3,0] = V_CR2

        if diagonalize:
            eig_val, eig_vec = np.linalg.eig(H)
            diag_matrix = np.linalg.inv(eig_vec) @ H @ eig_vec

            return diag_matrix
        
        return H
    
    def calc_coef(self, Hamiltonian, dim):
        """
        function for calculating the s coefficients needed for the Modified Redfield Eq.
        input: Hamiltonian (in electronic basis) and the dimension (row/col) of the Hamiltonian
        output: rank 4 tensor containing all the products of transformation coefficients
        """

        eig_val, eig_vec = np.linalg.eig(Hamiltonian)
        diag_matrix = np.linalg.inv(eig_vec) @ Hamiltonian @ eig_vec
        eig_vec_T = eig_vec.T                                       # transpose in order to match indexing for coef in articles
        s_en_em = np.zeros((dim, dim, dim))                         # placeholder for coefficients for ModRed (here also defined in terms of the electronic site basis)

        print(np.allclose(eig_vec @ eig_vec.T.conj(), np.eye(dim)))


        for i in range(dim):
            for e_n in range(dim):
                for e_m in range(dim):
                    s_en_em[i, e_n, e_m] = np.conj(eig_vec_T[e_n, i]) * eig_vec_T[e_m, i] 

        s_rank4 = np.zeros((dim, dim, dim, dim), dtype=complex) # placeholder for rank 4 tensor

        # taking product of s's and summing over i for each dimension of s_rank4. want 4 indices so we function can be called and indices thereafter can be inserted
        for e_n in range(dim):
            for e_m in range(dim):
                for e_g in range(dim):
                    for e_d in range(dim):
                        s_rank4[e_n, e_m, e_g, e_d] = np.sum(s_en_em[:, e_n, e_m] * s_en_em[:, e_g, e_d])

        threshold = 1e-15   # define a threshold for small values
        s_rank4[np.abs(s_rank4) < threshold] = 0  

        return s_rank4 
    
    def detailed_balance(self, kernel, H_exci):
        """
        checks the detailed balance condition for a rate kernel
        K[i,j] / (K[j,i]*exp(-beta*(E_i - E_j))) for each off-diagonal element
        
        parameters:
            kernel  : 2D numpy array (the rate kernel)
            H_exci  : 2D numpy array (the exciton Hamiltonian. diagonalized)
            beta    : inverse temperature factor
        """
        dim = kernel.shape[0]
        transitions = []  # e.g. "i -> f"
        print("detailed balance check (for each off-diagonal element):")
        for i in range(dim):
            for f in range(dim):
                if i != f:
                    # detailed balance: K_if = K_fi * exp[-beta*(E_i - E_f)]
                    lhs = kernel[i, f]
                    # rhs = kernel[f, i] * np.exp( - (H_exci[i,i] - H_exci[f,f]) * u.eV * self.beta)       # w/o hbar and energy difference in eV
                    rhs = kernel[f, i] * np.exp( - eV_to_ps1((H_exci[i,i] - H_exci[f,f]) * u.eV) * self.beta * const.hbar.to(u.eV * u.ps))        # with hbar and energy difference as frequency


                    # exp = np.exp( - (H_exci[i,i] - H_exci[f,f]) * u.eV * self.beta)       # w/o hbar and energy difference in eV
                    exp = np.exp( - eV_to_ps1((H_exci[i,i] - H_exci[f,f]) * u.eV) * self.beta * const.hbar.to(u.eV * u.ps) )        # with hbar and energy difference as frequency (get same results as above)
                    arg = - eV_to_ps1((H_exci[i,i] - H_exci[f,f]) * u.eV) * self.beta * const.hbar.to(u.eV * u.ps) 
                    transitions.append(f"{i}→{f}")
                    print(f"transition {i}→{f}: K[{i},{f}] = {lhs:.3e}, K[{f},{i}]*exp(-ß w_if hbar)= {rhs:.3e}, exp {exp:.3e}, arg {arg:.3e}") 

def cot(x):
    return np.cos(x)/np.sin(x)

def coth(x):
        return 1 / np.tanh(x)

def cm1_to_eV(cm1):
    """
    unit change from reciprocal centimeters (cm^-1) to electron volts (eV).
        input: value [cm^-1]
        output: value [eV]    
        c [cm/s]
        h [J s]
    """
    return (cm1 * const.c.cgs * const.h).to(u.eV)

def eV_to_s1(eV):
    """
    unit change from eV to s^-1.
        input: value [eV]
        output: value [s^-1]    
        h [eV s]
    """
    return (eV / const.hbar.to(u.eV * u.s))

def s1_to_eV(s1):
    """
        unit change from s^-1 to eV.
        input: value [s^-1]    
        output: value [eV]
        h [eV s]
    """
    return (s1 * const.hbar.to(u.eV * u.s))

def s_to_eV1(s):
    """
    test of unit change from s to eV^-1
    input: value [s]
    output: value [eV^-1]
    h [eV s]
    """
    return s/(const.hbar.to(u.eV * u.s))

def ps_to_eV1(ps):
    """
    test of unit change from ps to eV^-1
    input: value [ps]
    output: value [eV^-1]
    h [eV ps]
    """
    # return ps/(const.hbar.to(u.eV * u.ps))
    return ps/(const.hbar.to(u.eV * u.ps))

def eV_to_ps1(eV):
    """
    unit change from eV to ps^-1.
        input: value [eV]
        output: value [ps^-1]    
        h [eV ps]
    """
    # return (eV / const.hbar.to(u.eV * u.ps))
    return (eV / const.hbar.to(u.eV * u.ps))

def ps1_to_eV(ps1):
    """
        unit change from ps^-1 to eV.
        input: value [ps^-1]    
        output: value [eV]
        h [eV ps]
    """
    return (ps1 * const.hbar.to(u.eV * u.ps))


def single_exponential(t, A, tau):
    return A * (1 - np.exp(-t / tau))


def main(plot=True):
    temperature = 77 * u.K
    temp_HT = 300 * u.K
    beta = 1 / (const.k_B * temperature).to(u.eV)
    beta_HT = 1 / (const.k_B * temp_HT).to(u.eV)
    reorg = cm1_to_eV(100 * u.cm**-1) # 100 cm^-1; value from AV15
    relax_rate = s1_to_eV(((100 * u.fs)**(-1)).to(u.s ** (-1))) #* 10**-1 # (50 fs)**-1 value from AV15 
    relax_rate_HT = s1_to_eV(((100 * u.fs)**(-1)).to(u.s ** (-1)))  # (100 fs)**-1 value from IF09_parameters, corresponds to 53 cm^-1

    time_array = np.linspace(0, 0.6, 10**4) * u.ps  
    
    freq_array_HT = np.arange(0,5,0.01) * u.eV
    freq_array = np.arange(-600,600,0.01) * u.eV
    t_fixed = 3 * u.fs  

    n_max = 10**6

    bath_HT = Bath(temp_HT, beta_HT, freq_array_HT, reorg, relax_rate_HT, t_fixed.to(u.ps), time_array, n_max)
    bath = Bath(temperature, beta, freq_array, reorg, relax_rate, t_fixed.to(u.ps), time_array, n_max)
    # plot_spec_dens = bath_HT.plot_spectral_density2()

    ### plot g(t), g_dot(t) and C(t) - both summation and HTL
    # bath.g_dot_dot_sum(t_fixed.to(u.ps), plot=True, int=False)
    # bath_HT.g_dot_dot_sum(t_fixed.to(u.ps), plot=True, int=False)
    # bath.g_dot_dot_HT(t_fixed.to(u.ps), plot=True, int=False)
    # bath.g_dot_sum(t_fixed.to(u.ps), plot=True, int=False)
    # bath.g_dot_HT(t_fixed.to(u.ps), plot=True, int=False)
    # bath.g_sum(t_fixed.to(u.ps), plot=True, exp_plt = True, int=False)
    # bath.g_HT(t_fixed.to(u.ps), plot=True, exp_plt = True, int=False)
    
    t_min = 0
    t_max = np.inf

    ### integrate over t
    corr_HT = bath_HT.g_dot_dot_HT(t_fixed.value, plot=False, int=True)
    g_dot_HT = bath_HT.g_dot_HT(t_fixed.value, plot=False, int=True)
    g_HT = bath_HT.g_HT(t_fixed.value, plot=False, exp_plt=False, int=True)

    corr_sum = bath_HT.g_dot_dot_sum(t_fixed.value, plot=False, int=True)
    g_dot_sum = bath_HT.g_dot_sum(t_fixed.value, plot=False, int=True)

    # time_integral_C_HT, error_C_HT = quad(lambda t: bath_HT.g_dot_dot_HT(t, plot=False, int=True).real * u.ps**2 + bath_HT.g_dot_dot_HT(t, plot=False, int=True).imag * u.ps**2, t_min, t_max)
    # time_integral_g_dot_HT, error_g_dot_HT = quad(lambda t: bath_HT.g_dot_HT(t, plot=False, int=True).real * u.ps**1 + bath_HT.g_dot_HT(t, plot=False, int=True).imag * u.ps**1, t_min, t_max)
    # time_integral_g_HT, error_g_HT = quad(lambda t: np.exp(- (bath_HT.g_HT(t, plot=False, exp_plt = False, int=True).real + bath_HT.g_HT(t, plot=False, exp_plt = False, int=True).imag)), t_min, t_max)
    # time_integral_C_sum, error_C_sum = quad(lambda t: bath_HT.g_dot_dot_sum(t, plot=False, int=True).real * u.ps**2 + bath_HT.g_dot_dot_sum(t, plot=False, int=True).imag * u.ps**2, t_min, t_max)
    # time_integral_g_dot_sum, error_g_dot_sum = quad(lambda t: bath_HT.g_dot_sum(t, plot=False, int=True).real * u.ps**1 + bath_HT.g_dot_sum(t, plot=False, int=True).imag * u.ps**1, t_min, t_max)
    # time_integral_g_sum, error_g_sum = quad(lambda t: np.exp(- (bath_HT.g_sum(t, plot=False, exp_plt = False, int=True) + bath_HT.g_sum(t, plot=False, exp_plt = False, int=True))), t_min, t_max)
  
    ### coefficients and population simulations
    
    dim = 4
    H_elec = bath.Hamiltonian(dim, diagonalize=False)
    H_exci = bath.Hamiltonian(dim, diagonalize=True)
    coef = bath.calc_coef(H_elec, dim)
    
    initial_pop = bath_HT.initial_population(dim, population_type="first")
    print("Initial population:", initial_pop)

    ### KERNEL

    kernel_HT = bath_HT.memory_kernel_osc(H_exci, coef, t_min, t_max, HT=True)
    print("kernel:\n", kernel_HT)
    eig_val_kernel, _ = np.linalg.eig(kernel_HT)
    print("eig val kernel:", eig_val_kernel)

    ### plot eigen values of memory kernel
    plt.figure(figsize=(8, 5))
    plt.scatter(eig_val_kernel.real, eig_val_kernel.imag, color='blue', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # x-axis
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # y-axis
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Eigenvalues of Memory Kernel on the Complex Plane")
    plt.show()

    detailed_balance_check = bath_HT.detailed_balance(kernel_HT, H_exci)

    # plot_kernel = bath_HT.plot_memory_kernel_terms(H_exci, coef, t_min=0, t_max=1, e_n=1, e_m=0, num_points=200)
    print("for each column, sum of row elements of kernel:", np.sum(kernel_HT, axis=0))

    deri_pop = bath_HT.pop_derivative(t_fixed, initial_pop, kernel_HT)
    print("derivative of population: \n", deri_pop)

    t_span = [0, 100]        # time span for simulation
    t_eval = np.linspace(t_span[0], t_span[1], 501)
                                                                                                                        # LSODA
    sol = solve_ivp(bath_HT.pop_derivative, t_span, initial_pop.flatten(), t_eval=t_eval, args=(kernel_HT,), method='DOP853')
    colors = ['darkmagenta','olive', 'hotpink', 'orange'] 

    plt.figure(figsize=(12, 10))
    for i in range(dim):
        print(i)
        plt.plot(sol.t, sol.y[i], color=colors[i])#, label=labels[i])
    plt.plot(sol.t, np.sum(sol.y, axis=0), 'k--', label="Sum of P(t)")  # Check total population
    plt.xlabel(r"Time, $[$ps$]$", fontsize=28)
    plt.ylabel("Population", fontsize=28)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend((r'$\mathregular{D-C-A}$', r'$\mathregular{D-C^{*}-A}$', r'$\mathregular{D^{+}-C^{-}-A}$', r'$\mathregular{D^+-C-A^-}$', 'sum of pop.'), fontsize=26, loc='upper right')
    plt.title("Population Dynamics", fontsize=32)
    plt.grid()
    plt.subplots_adjust(left=0.149, right=0.98)
    plt.show()

    
if __name__ == "__main__":
    main(plot = True)

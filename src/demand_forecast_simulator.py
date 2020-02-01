# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:45:49 2020

@author: Raymon van Dinter
Assignment - Forecasting weather-dependent demand simulation script
"""
import statistics

import numpy as np
from scipy.stats import poisson


class DemandForecastSimulator:
    """
    The Demand Forecast Simulator, a simulator that takes perishable items into account.
    Simply set demand and call the simulate() function.
    """

    def __init__(self, model, shelf_life=7, I0=0, review_p=1, lead_time=1, z=1, TW=30, TK=7, ss=1.5, mu=2):
        """
        Initialize the DemandForecastSimulator
        :param model: string name of S estimation model. Use 'constant', 'MA' (mean average),
        or 'ML' for manual input demand
        :param shelf_life: Products have a fixed self life (expiration date) of m days counted from the moment the
        products are added to stock. Expired products are removed from stock
        :param I0: Initial stock
        :param review_p: review period
        :param lead_time: lead time
        :param z: buffer day
        :param TW: model warmup time
        :param TK: model cooldown time
        :param ss: safety stock
        :param mu: arbitrairly set mu for the constant model
        """
        self.model = model
        self.n_age_classes = shelf_life
        self.I0 = I0
        self.mu = mu
        self.z = z
        self.lead_time = lead_time
        self.review_p = review_p
        self.TW = TW
        self.TK = TK
        self.ss = ss

    def adjust_I_(self, n, t, I, D, short):
        """
        Adjust the stock using the demand, state a shortage when occurring
        :param I: stock
        :param D: demand
        :param short: shortage
        :param n: number of iterations
        :param t: day iteration
        :return: None
        """
        # Demand is higher than total stock
        if D[n, t] > np.sum(I[n, t, :]):
            short[n, t] = abs(D[n, t] - np.sum(I[n, t, :]))
            I[n, t, :].fill(0)
        # Demand is lower than total stock
        else:
            demand = D[n, t]
            i = 0
            while demand > 0:
                # Demand is higher than first element in stock (apply demand fifo-wise)
                if demand > I[n, t, i]:
                    demand -= I[n, t, i]
                    I[n, t, i] -= I[n, t, i]
                # Demand is lower than an element in stock
                else:
                    I[n, t, i] -= demand
                    demand = 0
                i += 1

    def get_S_(self, n, t, D):
        """
        Get a value for S
        :param n: number of iterations
        :param t: day iteration
        :param D: demand numpy array with shape (n, t)
        :return: single value for S
        """
        S = 0

        # Model selector
        if self.model == 'constant':
            S = (self.lead_time + self.review_p + self.z) * self.mu
        elif self.model == 'MA':
            # First month cannot be predicted with MA,
            # because preceding data is not available
            if t < 28:
                S = (2 + self.z) * self.mu
            else:
                # Get the mean demand of the four preceding weekdays of today and tomorrow and add add a safety stock
                today_4 = D[n, :t][::-1][7::7][0:3]
                tomorrow_4 = D[n, :t][::-1][8::7][0:3]
                S = (statistics.mean(today_4) + statistics.mean(tomorrow_4)) * self.ss
        elif self.model == 'ML':
            # Get the predicted demand of today and tomorrow and add a safety stock
            S = (D[0][t] + D[0][t + 1]) * self.ss
        return S

    def get_sim_D_(self, t):
        """
        Get a single value for demand using the poisson distribution with a daily shifting mu
        :param t: day iteration
        :return: single demand value
        """
        week_day = t % 6
        mu = [5, 3, 4, 6, 8, 9][week_day]
        return poisson.rvs(mu, size=1)[0]

    def set_D(self, D, n_runs, TO):
        """
        Set demand with data defined by another class or method
        :param D: demand array
        :param n_runs: number of iterations
        :param TO: observation time
        :return: None
        """
        days = self.TW + TO + self.TK
        self.D = np.full(shape=(n_runs, days), fill_value=D)

    def simulate(self, n_runs, TO):
        """
        Simulate the demand forecasting
        :param n_runs: number of iterations
        :param TO: observation time
        :return: D, S, Q, I, short, waste, ShortRun, WasteRun all from observation time
        """
        days = self.TW + TO + self.TK

        # Initialize arrays
        ShortRun = np.zeros(shape=n_runs)
        WasteRun = np.zeros(shape=n_runs)
        S = np.zeros(shape=(n_runs, days))
        Q = np.zeros(shape=(n_runs, days))
        I = np.zeros(shape=(n_runs, days, self.n_age_classes + 1))
        short = np.zeros(shape=(n_runs, days))
        waste = np.zeros(shape=(n_runs, days))

        for n in range(n_runs):
            I[n, 0, -1] = self.I0
            for t in range(days - 1):
                # Set order up to level to estimated demand
                # Uncomment the line underneath if you would like to have a poisson distribution simulated demand
                # D[n, t] = self.get_D_(t)
                S[n, t] = self.get_S_(n, t, self.D)
                # Set order quantity
                Q[n, t] = max(0, S[n, t] - np.sum(I[n, t, :]))
                # Set and meet demand
                self.adjust_I_(n, t, I, self.D, short)
                # Set next days starting inventory
                I[n, t + 1, -1] = Q[n, t]
                # Compute waste for this day
                waste[n, t] = I[n, t, 0]
                # Fifo shift
                I[n, t] = np.roll(I[n, t], -1)
                I[n, t + 1] = np.add(I[n, t + 1], I[n, t])

            # Calculate stats
            ShortRun[n] = np.sum(short[n, self.TW:-self.TK]) / np.sum(self.D[n, self.TW:-self.TK])
            WasteRun[n] = np.sum(waste[n, self.TW:-self.TK]) / np.sum(Q[n, self.TW:-self.TK])

        return self.D[:, self.TW:-self.TK], S[:, self.TW:-self.TK], Q[:, self.TW:-self.TK], I[:, self.TW:-self.TK], \
               short[:, self.TW:-self.TK], waste[:, self.TW:-self.TK], ShortRun, WasteRun

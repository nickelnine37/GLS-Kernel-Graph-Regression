import numpy as np

filter_functions = {'inverse': lambda lamL, beta: (1 + beta * lamL) ** -1,
                    'exponential': lambda lamL, beta: np.exp(-beta * lamL),
                    'ReLu': lambda lamL, beta: np.maximum(1 - beta * lamL, 0),
                    'sigmoid': lambda lamL, beta: 2 * np.exp(-beta * lamL) * (1 + np.exp(-beta * lamL)) ** -1,
                    'cosine': lambda lamL, beta: np.cos(lamL * np.pi / (2 * lamL.max())) ** beta,
                    'cut-off': lambda lamL, beta: (lamL <= 1 / beta).astype(int) if beta != 0 else np.ones_like(lamL)}
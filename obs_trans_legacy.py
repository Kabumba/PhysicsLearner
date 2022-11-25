import torch

import mirror_data
from configuration import Configuration


class ObservationTransformerLegacy:
    def __init__(self, config: Configuration):
        self.config = config
        self.n_factor = self.n_factor()

    def n_factor(self):
        factor = 1
        if self.config.mirror:
            factor += 1
        if self.config.invert:
            factor += 1
        if self.config.mirror_and_invert:
            factor += 1
        if self.config.num_car_in > 0 or self.config.num_car_out > 0:
            factor *= 2
        return factor

    def __call__(self, sample, mode):
        x, y = sample
        x_new = torch.zeros(self.config.in_size, device=x.device)
        y_new = torch.zeros(self.config.out_size, device=y.device)
        fci = 0  # first car index input
        fco = 0  # first car index output
        dbi = self.config.num_car_in * 46 # index where delta ball obs start

        # car swap
        swap_cars = False
        if self.config.num_car_in > 0 or self.config.num_car_out > 0:
            swap_cars = mode >= self.n_factor / 2
            if swap_cars:
                mode -= self.n_factor / 2

        # mirror operations
        if mode > 0:
            if self.config.mirror and mode == 1:
                x = mirror_data.mirror_state(x)
                y = mirror_data.mirror_state(y)
            elif self.config.mirror_and_invert and mode == (self.n_factor / 2 - 1):
                x = mirror_data.invert_and_mirror_state(x)
                y = mirror_data.invert_and_mirror_state(y)
            elif self.config.invert:
                x = mirror_data.invert_state(x)
                y = mirror_data.invert_state(y)

        # inputs

        if self.config.ball_in:
            x_new[:9] = x[:9]
            fci = 9
            dbi += 9
        else:
            if self.config.delta_inputs:
                fci = 9
        if self.config.num_car_in == 1:
            if swap_cars:
                x_new[fci:fci + 38] = x[47:85]
                x_new[fci + 38: fci + 46] = x[93:]
            else:
                x_new[fci:fci + 38] = x[9:47]
                x_new[fci + 38: fci + 46] = x[85:93]
        if self.config.num_car_in == 2:
            if swap_cars:
                x_new[fci:fci + 38] = x[47:85]
                x_new[fci + 38:fci + 76] = x[9:47]
                x_new[fci + 76:fci + 84] = x[93:]
                x_new[fci + 84:fci + 92] = x[85:93]
            else:
                x_new[fci:fci + 92] = x[9:]
            x_new[fci + 38:fci + 53] -= x_new[9:24]
        if self.config.ball_in:
            if self.config.delta_inputs:
                x_new[dbi:dbi + 3] = x[:3] - x_new[9:12]
                x_new[dbi + 3:dbi + 9] = x[3:9] - x_new[18:24]
        else:
            if self.config.delta_inputs:
                fci = 9
                x_new[:3] = x[:3]-x_new[9:12]
                x_new[3:9] = x[3:9] - x_new[18:24]

        # targets
        if self.config.ball_out:
            y_new[:9] = y[:9]
            fco = 9
            if self.config.delta_targets:
                y_new[:9] -= x_new[:9]
        if self.config.num_car_out >= 1:
            if swap_cars:
                y_new[fco:fco + 15] = y[47:62]
                # skip over boost-amount
                y_new[fco + 15:fco + 20] = y[63:68]
                # leave out time related data
            else:
                y_new[fco:fco + 15] = y[9:24]
                # skip over boost-amount
                y_new[fco + 15:fco + 20] = y[25:30]
                # leave out time related data
            if self.config.delta_targets:
                y_new[fco:fco + 3] -= x_new[fci:fci+3]
                y_new[fco+3:fco + 9] -= x_new[fci+3:fci + 9]
                y_new[fco + 9:fco + 15] -= x_new[fci + 9:fci + 15]
        if self.config.num_car_out == 2:
            if swap_cars:
                y_new[fco + 20:fco + 35] = y[9:24]
                # skip over boost-amount
                y_new[fco + 35:fco + 40] = y[25:30]
                # leave out time related data
            else:
                y_new[fco + 20:fco + 35] = y[47:62]
                # skip over boost-amount
                y_new[fco + 35:fco + 40] = y[63:68]
                # leave out time related data
            if self.config.delta_targets:
                y_new[fco + 20:fco + 23] -= x_new[fci + 47:fci + 50]
                y_new[fco + 23:fco + 29] -= x_new[fci + 50:fci + 56]
                y_new[fco + 29:fco + 35] -= x_new[fci + 56:fci + 62]
        return x_new, y_new

import torch

import mirror_data
from configuration import Configuration


class ObservationTransformer:
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

        x_new[:9] = x[:9]
        if swap_cars:
            x_new[9:47] = x[47:85]
            x_new[47:85] = x[9:47]
            x_new[85:93] = x[93:101]
            x_new[93:101] = x[85:93]
        else:
            x_new[9:101] = x[9:101]

        y_new[:9] = y[:9]
        if swap_cars:
            y_new[9:24] = y[47:62]
            # skip over boost-amount
            y_new[24:29] = y[63:68]
            # leave out time related data
        else:
            y_new[9:24] = y[9:24]
            # skip over boost-amount
            y_new[24:29] = y[25:30]
            # leave out time related data
        return x_new, y_new

import numpy as np


def rmse(deviation):  # root mean square error
    rmse = np.sqrt(((deviation) ** 2).mean())
    return rmse


class MetricRegression():
    def __init__(self, real, predict):
        self.real = np.array(real)
        self.predict = np.array(predict)
        self.samples_error = self.predict - self.real

    def MAE(self):
        mean_abs_error = np.mean(np.abs(self.predict - self.real))
        return mean_abs_error

    def STD(self):  # standard deviation
        std = np.std(self.samples_error, ddof=1)
        # print("standard deviation: %.4f" % std)
        return std

    def RMSE(self):
        RMSE_value = rmse(np.array(self.samples_error))
        return RMSE_value

    def MAP_error(self):  # Mean Absolute Percentage Error, MAPE
        abs_errors = np.abs(self.samples_error)
        ape_errors = abs_errors / self.real
        mean_ape = np.mean(ape_errors)
        return mean_ape

    def PPMC(self):  # Pearson Product Moment Correlation, PPMC
        predict_mean = np.mean(self.real)
        real_mean = np.mean(self.predict)

        numerator = np.sum((self.predict - predict_mean) * (self.real - real_mean))
        denominator = np.sqrt(np.sum(np.square(self.predict - predict_mean))) * np.sqrt(
            np.sum(np.square(self.real - real_mean)))
        PPMC = numerator / denominator
        return PPMC

    def Mape(self):  # mean APE
        abs_errors = np.abs(self.samples_error)
        ape_errors = abs_errors / self.real
        mean_ape = np.mean(ape_errors)
        return mean_ape


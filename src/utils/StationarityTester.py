from statsmodels.tsa.stattools import adfuller, kpss

SIGNIFICANCE_LEVEL = 0.05

class StationarityTester:
    @staticmethod
    def run_test(signal, stationarity_test='kdss'):
        map_test_type_to_test = {
            'kdss': StationarityTester.run_kdss,
            'adf': StationarityTester.run_adf
        }

        return map_test_type_to_test[stationarity_test](signal)

    @staticmethod
    def run_kdss(signal):
        result = kpss(signal, regression='c')
        print('\nKPSS Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        for key, value in result[3].items():
            print('Critial Values:')
            print(f'   {key}, {value}')
        return result[1] <= SIGNIFICANCE_LEVEL

    @staticmethod
    def run_adf(signal):
        # ADF Test
        result = adfuller(signal, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print('Critial Values:')
            print(f'   {key}, {value}')
        return result[1] <= SIGNIFICANCE_LEVEL


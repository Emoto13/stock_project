MAP_PERIODICITY_TO_SEASONALITY_PARAMETERS = {
    'monthly': { 
        'period': 30.5,
        'fourier_order': 5
        },
    'weekly': {
        'period': 7,
        'fourier_order': 3
        },
    'daily': {
        'period': 1,
        'fourier_order': 1
        },
}

MAP_PERIODICITY_TO_FREQUENCY = {
    'monthly': { 'freq': 'M', 'periods': 12 },
    'weekly': { 'freq': 'W', 'periods': 52 },
    'daily': { 'freq': 'D', 'periods': 365 }
}

MAP_PERIODICITY_TO_SEASONALITY = {
    "daily": { "daily_seasonality": True},
    "weekly": { "weekly_seasonality": True},
    "monthly": { "monthly_seasonality": True}
}
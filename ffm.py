catalog =\
'/Users/leonardovanderlaat/Desktop/tornillos/characteristics/catalog.csv'
time_column = 't1'
variable = 'count'
rule = '12H'
explosion = '2016-04-30 00:00:00+0000'

import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams["scatter.marker"] = '.'
from cycler import cycler
rcParams['axes.prop_cycle'] = cycler('color', ['k', 'm', 'y', 'k'])


# Frequency label
time_unit = {'D':'días', 'H':'horas', 'M':'minutos', 'S':'segundos'}
if int(rule[:-1]) == 1:
    time_unit[rule[-1]] = time_unit[rule[-1]][:-1]

# Get explosion time
explosion = datetime.datetime.strptime(explosion, '%Y-%m-%d %H:%M:%S%z')

def get_rate(catalog, time_column, rule):
    """
    Gets the rate of events per unit of time (defined by rule)
    Parameters
    ----------
    catalog: str
        Path to csv catalog
    time_column: str
        Name of the column containing the time of each event
    rule: str
        Frequency for resampling the data (eg. '1D' or '12H')
    Returns
    -------
    time: np array
        time array
    rate: np array
        Rate array
    inverse_rate: np array
        1/rate
    """
    # Load catalog
    df = pd.read_csv(catalog)
    df.index = pd.to_datetime(df[time_column])

    # Resample
    df         = df.drop(df.columns[:-1], axis=1)
    df.columns = ['count']
    df         = df.resample(rule).count()
    time = np.asarray([md.date2num(datetime) for datetime in df.index])

    # Get inverse rate
    rate         = np.asarray([df['count']])[0]
    inverse_rate = np.asarray([1/df['count']])[0]

    # Remove inf
    finite_values = np.isfinite(inverse_rate)
    inverse_rate  = inverse_rate[finite_values]
    time          = time[finite_values]
    rate          = rate[finite_values]

    return time, rate, inverse_rate

time, rate, inverse_rate = get_rate(catalog, time_column, rule)

def get_linregress(time, inverse_rate):
    """
    Compute de lineal regression
    Parameters
    ----------
    time: np array
        time array
    inverse_rate: np array
        1/rate
    Returns
    -------
    """
    slope, intercept, r_value, p_value, std_err = linregress(time,
                                                             inverse_rate)

    num_time_forecast = -intercept/slope
    forecast_dt = md.num2date(num_time_forecast, tz=None)

    new_x = np.arange(time[0], num_time_forecast + 1, 1)
    new_y = slope * new_x + intercept

    # Real intercept
    intercept = new_x[0]*slope + intercept


    return new_x, new_y, intercept, slope, r_value, forecast_dt

def get_forecast_error(forecast_dt, explosion, time_unit):
    """
    Calculates forecast-actual time difference
    Parameters
    ----------
    forecast_dt: datetime
        Datetime of forecast
    explosio: datetime
        Actual date and time of explosion
    time_unit: dict
    Returns
    -------
    diff: float
        difference in seconds
    diff_str: str
        Difference with easily grasping units
    """
    diff = (forecast_dt - explosion).total_seconds()
    if abs(diff) <= 60:
        diff_str = '{} {}'.format(int(diff), time_unit['S'])
    if abs(diff) > 60 and abs(diff) <= 3600:
        diff_str = '{} {}'.format(int(diff/60), time_unit['M'])
    if abs(diff) > 3600 and abs(diff) <= 86400:
        diff_str = '{} {}'.format(int(diff/3600), time_unit['H'])
    elif abs(diff) > 86400:
        diff_str = '{} {}'.format(int(diff/86400), time_unit['D'])

    if diff > 0:
        diff_str = '+' + diff_str

    return diff, diff_str

new_x, new_y, intercept, slope, r_value, forecast_dt = get_linregress(time,
                                                                 inverse_rate)

diff, diff_str = get_forecast_error(forecast_dt, explosion, time_unit)

# Figure
days = md.DayLocator(interval=15)
dayFmt = md.DateFormatter('%m-%d')

fig = plt.figure(figsize=(6,4))
fig.suptitle('Error del pronóstico: {}'.format(diff_str))
fig.subplots_adjust(left=.12, bottom=.1, right=.88, top=.92, wspace=.06,
                    hspace=.2)

ax = fig.add_subplot(121)
ax.set_ylabel('Número inverso de eventos por {}'.format(time_unit[rule[-1]]))
ax.scatter(time, inverse_rate)
ax.axvline(x=explosion, c='r')
ax.axhline(y=0, c='k', linestyle='--')
ax.plot(new_x, new_y,c='r',linewidth=2)
ax.text(.5, .9, 'y = {}x + {}\n$R^2$ = {}'.format(round(slope, 3),
                           round(intercept, 2), round(r_value**2, 1)),
        transform=ax.transAxes, bbox=dict(facecolor='white'),
        verticalalignment='center', horizontalalignment='center')

ax1 = fig.add_subplot(122)
ax1.set_ylabel('Número de eventos por {}'.format(time_unit[rule[-1]]),
               labelpad=20, rotation=270)
ax1.scatter(time, rate)
ax1.plot(new_x[new_y>0], 1/new_y[new_y>0])
ax1.axvline(x=explosion, c='r')
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
ax1.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])

for ax in fig.get_axes():
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(dayFmt)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xlabel(explosion.year)
plt.show()

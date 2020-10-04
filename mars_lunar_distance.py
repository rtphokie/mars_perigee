import unittest
from skyfield import api, almanac
from datetime import datetime
from contextlib import closing
from skyfield.searchlib import find_maxima, find_minima
import numpy as np
from scipy.signal import argrelextrema
from pprint import pprint
from itertools import combinations
import collections
import math


ts = api.load.timescale(builtin=True)
load = api.Loader('/var/data')
eph = load('de422.bsp') # JPL's long period ephemeris -3000 through 3000
# eph = load('de430t.bsp') # shorter period, better accuracy at higher time resolutions

millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.1f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def venus_elongation_degrees(t):
    global body
    e = eph['earth'].at(t)
    u = e.observe(eph['venus']).apparent()
    return u.distance().km

def angular_separation_two_bodies(t, bodies):
    # todo, make min/max optional to improve performance
    e = eph['earth'].at(t)
    body1 = eph[bodies[0]]
    body2 = eph[bodies[1]]
    sep = e.observe(body1).separation_from(e.observe(body2))
    jkl = argrelextrema(sep.degrees, np.less)
    minima = jkl[0]
    jkl = argrelextrema(sep.degrees, np.greater)
    maxima = jkl[0]
    return minima, maxima, sep

def earth_distance(t, body):
    astrometric = eph['earth'].at(t).observe(eph[body])
    ra, dec, distance = astrometric.radec()
    jkl = argrelextrema(distance.km, np.less)
    minima = jkl[0]
    jkl = argrelextrema(distance.km, np.greater)
    maxima = jkl[0]
    return minima, maxima, distance

def narrowdownmin(bodies, th, extreme='minima',):
    minima, maxima, sep = angular_separation_two_bodies(th, bodies)
    if extreme == 'minima':
        extreme = minima
    else:
        extreme = maxima
    for ih in extreme:
        tm = ts.utc(th[ih].utc.year, th[ih].utc.month, th[ih].utc.day, th[ih].utc.hour, range(-60, 60))
        minima_m, maxima_m, sep_m = angular_separation_two_bodies(tm, bodies)
        if extreme == 'minima':
            extreme_m = minima_m
        else:
            extreme_m = maxima_m
        return tm[extreme_m].utc_iso()[0], sep_m.degrees[extreme_m][0]

class MyTestCase(unittest.TestCase):
    def test_uranus_closest(self):
        t = ts.utc(2020,1,range(1,370))
        venus_elongation_degrees.step_days = 1
        t1 = ts.utc(2020)
        t2 = ts.utc(2021)
        body = eph['mercury']

        t, values = find_minima(t1, t2, venus_elongation_degrees)
        print(len(t), 'maxima found')
        for ti, vi in zip(t, values):
            print(ti.utc_strftime('%Y-%m-%d %H:%M '), '%.2f' % vi,
                  'degrees elongation')

    def test_gen_minima(self):
        td = ts.utc(2020,1,range(1,368))
        observable_bodys=['venus', 'moon', 'mars barycenter', 'jupiter barycenter', 'saturn barycenter', 'mercury']
        observable_bodys=['venus', 'moon']
        jkl = combinations(observable_bodys, 2)
        data = {}

        for bodies in list(combinations(observable_bodys, 2)):
            if bodies[0] not in data.keys():
                data[bodies[0]] = dict((val, {'minima': {}, 'maxima': {}})
                  for val in observable_bodys)
                del data[bodies[0]][bodies[0]]

            minima_d, maxima_d, sep_d = angular_separation_two_bodies(td, bodies)
            print('-'*20)
            print(minima_d)
            print(maxima_d)
            for idi in minima_d:
                th = ts.utc(td[idi].utc.year, td[idi].utc.month, td[idi].utc.day, range(-24, 24))
                jkl = narrowdownmin(bodies, th, extreme='minima')
                data[bodies[0]][bodies[1]]['minima'][jkl[0]] = {'deg': jkl[1]}
            for idx in maxima_d:
                th = ts.utc(td[idx].utc.year, td[idx].utc.month, td[idx].utc.day, range(-24, 24))
                print(th)
                jkl = narrowdownmin(bodies, th, extreme='maxima')
                data[bodies[0]][bodies[1]]['maxima'][jkl[0]] = {'deg': jkl[1]}
        pprint(data)

    def test_lunar_apsis(self):
        td = ts.utc(2020,1,range(1,368))
        minima_d, max, dist = earth_distance(td, 'Moon')

        for idi in minima_d:
            print(f"{td[idi].utc_jpl()} {dist.km[idi]}")


    def test_Mars_apsis(self):
        # closet days 1900-2100
        startingyear=0
        years=3000
        leapdays=round(years/4)
        leapdays-=1 # 2000 is a common year
        days=years*365+leapdays
        td = ts.utc(startingyear,0,range(days))
        minima_d, max, dist = earth_distance(td, 'Mars Barycenter')
        distances={}
        for idi in minima_d:
            distances[td[idi]] = dist.km[idi]
        from operator import itemgetter

        years=[]
        lookup={}
        print(f"across {td[0].utc_jpl()} - {td[-1].utc_jpl()}")
        for k,v in sorted(distances.items(), key=itemgetter(1)):
            year=k.utc_strftime('%Y')
            lookup[year]=v
            year=int(year)
            years.append(year)
            years_past = list(filter(lambda x: (x < year), years))
            years_future = list(filter(lambda x: (x > year), years))
            print(f"{k.utc_jpl()} {millify(v)} km", end='')
            if len(years_past) > 0:
               print(f", since {np.max(years_past)} ", end='')
            if len(years_future) > 0:
               print(f", until {np.min(years_future)}", end='')
            print()
            # print (len(years), len(years_past), len(years_future))

        # closet minute in calendar year 2020
        td = ts.utc(2020,1,1,0,range(525600))
        minima_d, max, dist = earth_distance(td, 'Mars Barycenter')
        for idi in minima_d:
            print(f"{td[idi].utc_jpl()} {millify(dist.km[idi])} km")


if __name__ == '__main__':
    with closing(eph):
        unittest.main()

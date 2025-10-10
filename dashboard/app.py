from flask import Flask, jsonify, render_template, request
from flask_caching import Cache

import polars as pl
import numpy as np

from hkjc.live import live_odds, _fetch_live_races
from hkjc.harville_model import fit_harville_to_odds
from hkjc.historical import get_horse_data
from hkjc.speedpro import speedmap, speedpro_energy
from hkjc.strategy import qpbanker, place_only


def arr_to_dict(arr: np.ndarray, dtype=float):
    """Convert 0-indexed numpy array into 1-indexed nested dictionary

    Args:
        arr (np.ndarray): 0-indexed numpy array
        dtype (type, optional): data type. Defaults to float.

    Returns:
        dict: 1-indexed nested dictionary
    """
    if arr.ndim == 1:
        return {i+1: dtype(v) for i, v in enumerate(arr) if not (np.isnan(v) or np.isinf(v))}

    return {i+1: arr_to_dict(v) for i, v in enumerate(arr)}


app = Flask(__name__)
config = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_REDIS_HOST": "localhost",
    "CACHE_REDIS_PORT": "6379"
}
app.config.from_mapping(config)
cache = Cache(app)


@app.route('/')
def disp_race_info():
    race_info = _fetch_live_races('', '')

    try:
        df_speedpro = speedpro_energy(race_info['Date'])
        for race_num, race in race_info['Races'].items():
            for i, runner in enumerate(race['Runners']):
                df = (df_speedpro
                      .filter(pl.col('RaceNo') == race_num)
                      .filter(pl.col('RunnerNumber') == int(runner['No']))
                      )
                race_info['Races'][race_num]['Runners'][i]['SPEnergy'] = df['SpeedPRO_Energy_Difference'].item(
                    0)
                race_info['Races'][race_num]['Runners'][i]['Fitness'] = df['FitnessRatings'].item(
                    0)
    except:  # fill with dummy value if SpeedPro not available
        for race_num, race in race_info['Races'].items():
            for i, runner in enumerate(race['Runners']):
                race_info['Races'][race_num]['Runners'][i]['SPEnergy'] = 0
                race_info['Races'][race_num]['Runners'][i]['Fitness'] = 0

    return render_template('index.html',
                           race_info=race_info)


turf_going_dict = {'FIRM': 'F',
                   'GOOD TO FIRM': 'GF',
                   'GOOD': 'G',
                   'GOOD TO YIELDING': 'GY',
                   'YIELDING': 'Y',
                   'YIELDING TO SOFT': 'YS',
                   'SOFT': 'S',
                   'HEAVY': 'H'}
aw_going_dict = {'WET FAST': 'WF',
                 'FAST': 'FT',
                 'GOOD': 'GD',
                 'SLOW': 'SL',
                 'WET SLOW': 'WS',
                 'RAIN AFFECTED': 'RA',
                 'NORMAL WATERING': 'NW'}
going_dict = {'TURF': turf_going_dict, 'ALL WEATHER TRACK': aw_going_dict}


@app.route('/horse_info/<horse_no>', methods=['GET'])
# cache horse history for 1 day
@cache.cached(timeout=86400, query_string=True)
def disp_horse_info(horse_no):
    # read optional filters
    dist = request.args.get('dist', type=int)
    track = request.args.get('track')
    going = request.args.get('going')
    venue = request.args.get('venue')

    if track not in going_dict.keys():
        track = None
    if venue not in ['HV', 'ST']:
        venue = None
    if (going is not None) and (track is not None) and (going in going_dict[track].keys()):
        going = going_dict[track][going]  # translate going to code
    else:
        going = None

    df = get_horse_data(horse_no)

    if df.height > 0:
        if dist is not None:
            df = df.filter(pl.col('Dist') == dist)
        if track and track.upper() == 'TURF':
            df = df.filter(pl.col('Track') == 'Turf')
        elif track and track.upper() == 'ALL WEATHER TRACK':
            df = df.filter(pl.col('Track') == 'AWT')
        if going is not None:
            df = df.filter(pl.col('G') == going)
        if venue is not None:
            df = df.filter(pl.col('Venue') == venue)

    return render_template('horse-info.html', df=df.head(5))


@app.route('/live_odds/<int:race_no>')
def disp_live_odds(race_no=1):
    odds_dict = live_odds('', '', race_no)
    fitted_odds = fit_harville_to_odds(odds_dict)['odds_fit']
    odds_json = {'Raw': {k: arr_to_dict(v) for k, v in odds_dict.items()},
                 'Fit': {k: arr_to_dict(v) for k, v in fitted_odds.items()}
                 }

    return jsonify(odds_json)


@app.route('/speedmap/<int:race_no>')
def disp_speedmap(race_no=1):
    return speedmap(race_no)


@app.route('/qp/<int:race_no>/<int:banker>/<cover>', methods=['GET'])
def disp_qp_metrics(race_no=1, banker=1, cover='2'):
    use_filter = request.args.get('filter', type=bool)
    odds = live_odds('', '', race_number=race_no)
    if use_filter:
        res = fit_harville_to_odds(odds)
        if res['success']:
            odds = res['odds_fit']
    covered = [int(v) for v in cover.split(',')]
    ev = qpbanker.expected_value(odds['PLA'], odds['QPL'], banker, covered)
    win = qpbanker.win_probability(odds['PLA'], banker, covered)
    avg_odds = qpbanker.average_odds(odds['QPL'], banker, covered)

    return {'Banker': banker, 'Covered': covered, 'ExpectedValue': ev, 'Win Rate': win, 'AvgOdds': avg_odds}


# TODO: Recommend QP trade given banker ?exclude= , , , &maxC=  => return list of pareto trades, cover, ev, win-prob, avgodds
@app.route('/qprec/<int:race_no>/<int:banker>', methods=['GET'])
def disp_qp_recs(race_no=1, banker=1, exclude=[], maxC=None):
    pass

# TODO: Recommend Place only trade ?exclude= , , , &maxC => return list of pareto trades, cover, ev, win-prob, avgodds

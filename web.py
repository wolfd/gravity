#!/usr/bin/env python2
import pandas as pd
import pygame
import sys
from optparse import OptionParser

from flask import Flask
import json
app = Flask(__name__)

parser = OptionParser()
parser.add_option("-f", "--file", dest="csv_name",
                  help="read from FILE", metavar="FILE",
		  default="output.csv")
parser.add_option("-n", "--num", dest="num_particles",
                  help="particles in sim", type="int")
parser.add_option("-s", "--skip", dest="skip",
                  help="iterations to skip", type="int",
                  default=1)

(options, args) = parser.parse_args()

PARTICLES = options.num_particles


scale = 1e-3

sun_index = 0

focus_planet = 0 


def to_pygame(x, y):
    scale_screen = scale / size[0]
    return ((x * scale_screen) + (size[0] / 2), (y * scale_screen) + (size[1] / 2))
    #return (size[0] / 2 + (x / scale) * size[0], size[1] / 2 + (y / scale) * size[0])

def get_particle(df, index):
    return (df['x'][index],  df['y'][index],  df['z'][index], 
            df['xv'][index], df['yv'][index], df['zv'][index])

def subtract_particles(a, b):
    return [a_i - b_i for a_i, b_i in zip(a, b)]

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

reader = pd.read_csv(options.csv_name, chunksize=PARTICLES, header=None, iterator=True)
chunk = 0
@app.route("/chunk")
@crossdomain(origin='*')
def main():
    global reader, chunk
    chunk += 1
    #import pdb; pdb.set_trace()
    d = reader.get_chunk()
    d.columns = ['x', 'y', 'z', 'xv', 'yv', 'zv']
    
    return json.dumps(d.as_matrix().tolist())

#iteration = 0
#for d in reader:
#    iteration += 1
#    if iteration % options.skip != 0:
#        continue
#    d.columns = ['x', 'y', 'z', 'xv', 'yv', 'zv']
#    ref = get_particle(d, focus_planet)
#    #pygame.display.set_caption(str(iteration))
#
#    #import pdb; pdb.set_trace()
#
#    for i in range(d.shape[0]):
#        focus = get_particle(d, focus_planet)
#        cur = get_particle(d, i)
#
#        rel = subtract_particles(cur, focus)
#        if i == sun_index:
#            draw_particle(rel, sun_color)
#        elif i in range(6):
#            draw_particle(rel, planet_color)
#        else:
#            draw_particle(rel)

if __name__ == "__main__":
    app.debug = True
    app.run(port=5001)

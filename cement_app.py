#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 🚀 cement_app.py: Live Cement Sensor + Forecast Dashboard
#
# This script simulates sensor feeds (OPC-UA + Modbus), stores them
# in-memory (no InfluxDB needed), computes a rolling mean, trains a
# simple regression model every second, and forecasts 7 steps ahead.
# A Flask + Chart.js dashboard displays the raw series, the rolling
# mean, the forecast line, the first-step forecast trail, live footer
# stats (latest raw, next-step pred, R²), and lets you download a CSV.
#
# Usage:
#   pip install flask numpy scikit-learn
#   python3 cement_app.py
#   open http://localhost:5101/ in your browser
################################################################################

import threading
import time
import math
import random
import io
import csv
from collections import deque
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request, make_response

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ───────────────────────────────────────────────────────────────────────────────
#                               CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
PORT             = 5101      # dashboard port
HIST_POINTS      = 200       # how many samples to keep
ROLLING_WINDOW   = 5         # for moving average
FORECAST_HORIZON = 7         # how many steps ahead to predict
UPDATE_INTERVAL  = 1.0       # seconds per new sample

# ───────────────────────────────────────────────────────────────────────────────
#                                GLOBAL STATE
# ───────────────────────────────────────────────────────────────────────────────
data_lock       = threading.Lock()
raw_queue       = deque(maxlen=HIST_POINTS)
roll_queue      = deque(maxlen=HIST_POINTS)
model           = LinearRegression()
model_r2        = 0.0
forecast_vector = [None]*FORECAST_HORIZON

# these three will back the CSV download
time_queue      = deque(maxlen=HIST_POINTS)   # ISO timestamps
pred_queue      = deque(maxlen=HIST_POINTS)   # first-step pred at each sample
r2_queue        = deque(maxlen=HIST_POINTS)   # R² at each sample

# ───────────────────────────────────────────────────────────────────────────────
# NEW: deque to track the first forecast-step trail over time
forecast1_trail = deque(maxlen=HIST_POINTS)
# ───────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────────
#                       SENSOR SIMULATION + MODEL LOOP
# ───────────────────────────────────────────────────────────────────────────────
def sensor_loop():
    t0 = time.time()
    # seed history
    for i in range(HIST_POINTS):
        ts = i * UPDATE_INTERVAL
        val = 25 + 5*math.sin(2*math.pi*(ts/60)) + random.uniform(-0.5,0.5)
        with data_lock:
            raw_queue.append(val)
            if len(raw_queue) >= ROLLING_WINDOW:
                window = list(raw_queue)[-ROLLING_WINDOW:]
                roll_queue.append(sum(window)/ROLLING_WINDOW)
            else:
                roll_queue.append(None)
    retrain_and_record()  # initial train

    # live updates
    while True:
        time.sleep(UPDATE_INTERVAL)
        elapsed = time.time() - t0
        val = 25 + 5*math.sin(2*math.pi*(elapsed/60)) + random.uniform(-0.5,0.5)
        with data_lock:
            raw_queue.append(val)
            window = list(raw_queue)[-ROLLING_WINDOW:]
            roll_queue.append(sum(window)/ROLLING_WINDOW)
        retrain_and_record()


def retrain_and_record():
    """Retrain model, compute R², forecast, record trails & timestamped stats."""
    global model_r2, forecast_vector

    with data_lock:
        Xy = [(roll_queue[i], raw_queue[i+1])
              for i in range(len(roll_queue)-1)
              if roll_queue[i] is not None]
    if len(Xy) < 10:
        return

    X_vals, y_vals = zip(*Xy)
    split = int(len(X_vals)*0.8)
    X_train, X_test = X_vals[:split], X_vals[split:]
    y_train, y_test = y_vals[:split], y_vals[split:]

    model.fit(np.array(X_train).reshape(-1,1), y_train)
    preds = model.predict(np.array(X_test).reshape(-1,1))
    model_r2 = float(r2_score(y_test, preds))

    # recursive forecast
    window = [roll_queue[-1]]
    fc = []
    for _ in range(FORECAST_HORIZON):
        p = float(model.predict([[window[-1]]])[0])
        fc.append(p)
        window.append(p)
    forecast_vector = fc

    # record first-step trail + timestamp + stats
    now_iso = datetime.utcnow().isoformat()
    with data_lock:
        forecast1_trail.append(fc[0])
        time_queue.append(now_iso)
        pred_queue.append(fc[0])
        r2_queue.append(model_r2)


# ───────────────────────────────────────────────────────────────────────────────
#                                   FLASK APP
# ───────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html>
<head>
  <title>Live Cement Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>Live Cement Sensor + Forecast</h1>

  <!-- date/time filters & download button -->
  <div style="margin-bottom:1em;">
    From: <input id="dt-start" type="datetime-local">
    To:   <input id="dt-end"   type="datetime-local">
    <button id="btn-download">Download CSV</button>
  </div>

  <canvas id="chart"></canvas>
  <div style="margin-top:1em;">
    Latest raw: <span id="raw">–</span> |
    Next-step pred: <span id="pred">–</span> |
    Model R²: <span id="r2">–</span>
  </div>

  <script>
    const ctx = document.getElementById('chart').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Raw',            data: [], borderColor:'blue',    fill:false },
          { label: 'Rolling-5',      data: [], borderColor:'orange',  fill:false },
          { label: 'Forecast→',      data: [], borderColor:'red',     borderDash:[5,5], fill:false },
          { label: '1st-step trail', data: [], borderColor:'pink',    pointRadius:2, fill:false }
        ]
      },
      options:{ animation:false }
    });

    async function fetchData(){
      const resp = await fetch('/data');
      const d    = await resp.json();
      chart.data.labels              = d.index;
      chart.data.datasets[0].data    = d.raw;
      chart.data.datasets[1].data    = d.rolling;
      chart.data.datasets[2].data    = d.forecast;
      chart.data.datasets[3].data    = d.trail;
      chart.update('none');
    }

    async function fetchStats(){
      const resp = await fetch('/status');
      const s    = await resp.json();
      document.getElementById('raw').textContent  = s.latest_raw.toFixed(2);
      document.getElementById('pred').textContent = s.next_pred.toFixed(2);
      document.getElementById('r2').textContent   = s.model_r2.toFixed(3);
    }

    document.getElementById('btn-download').onclick = () => {
      const start = document.getElementById('dt-start').value;
      const end   = document.getElementById('dt-end').value;
      let url = '/download';
      const params = new URLSearchParams();
      if(start) params.set('start', start);
      if(end)   params.set('end',   end);
      if([...params].length) url += '?' + params.toString();
      window.location = url;
    };

    setInterval(fetchData, 1000);
    setInterval(fetchStats, 1000);
    fetchData(); fetchStats();
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/data')
def data():
    with data_lock:
        N        = len(raw_queue)
        idx      = list(range(N+FORECAST_HORIZON))
        raw      = list(raw_queue) + [None]*FORECAST_HORIZON
        rolling  = list(roll_queue) + [None]*FORECAST_HORIZON
        forecast = [None]*N + forecast_vector
        trail    = list(forecast1_trail) \
                   + [None]*(N - len(forecast1_trail)) \
                   + [None]*FORECAST_HORIZON
    return jsonify(index=idx, raw=raw, rolling=rolling,
                   forecast=forecast, trail=trail)

@app.route('/status')
def status():
    with data_lock:
        lr  = raw_queue[-1]
        np1 = forecast_vector[0]
        r2v = model_r2
    return jsonify(latest_raw=lr, next_pred=np1, model_r2=r2v)

@app.route('/download')
def download_csv():
    """
    Download the timestamped raw/pred/R² as CSV.
    Query params: start=YYYY-MM-DDTHH:MM, end=...
    """
    start = request.args.get('start')
    end   = request.args.get('end')
    def to_dt(s): return datetime.fromisoformat(s)
    start_dt = to_dt(start) if start else None
    end_dt   = to_dt(end)   if end   else None

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['timestamp','raw','pred','r2'])
    with data_lock:
        for ts, raw, pred, r2 in zip(time_queue, raw_queue, pred_queue, r2_queue):
            dt = datetime.fromisoformat(ts)
            if (not start_dt or dt >= start_dt) and (not end_dt or dt <= end_dt):
                writer.writerow([ts, raw, pred, r2])

    resp = make_response(buf.getvalue())
    resp.headers['Content-Type']        = 'text/csv'
    resp.headers['Content-Disposition'] = 'attachment; filename="cement_data.csv"'
    return resp

if __name__=='__main__':
    threading.Thread(target=sensor_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT)

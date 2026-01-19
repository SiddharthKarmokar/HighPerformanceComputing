import csv
import json
import os

# Read CSV Data
filename = 'results.csv'
if not os.path.exists(filename):
    print("Error: results.csv not found")
    exit(1)

data = []
try:
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['N'] = int(row['N'])
            row['threads'] = int(row['threads'])
            row['sec'] = float(row['sec'])
            data.append(row)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Organize Data
# 1. Single Thread: Time vs N (series by pattern)
single_thread_data = {} # pattern -> {N -> sec}
all_ns = sorted(list(set(d['N'] for d in data)))
patterns = sorted(list(set(d['pattern'] for d in data))) # Changed from pattern_name

for row in data:
    if row['threads'] == 1:
        p = row['pattern'] # Changed from pattern_name
        if p not in single_thread_data:
            single_thread_data[p] = {}
        single_thread_data[p][row['N']] = row['sec']

# 2. Multi Thread: Time vs N (series by pattern, seperate charts per thread count)
thread_counts = sorted(list(set(d['threads'] for d in data if d['threads'] > 1)))
multi_thread_data = {} # thread_count -> pattern -> {N -> sec}

for t in thread_counts:
    multi_thread_data[t] = {}
    for row in data:
        if row['threads'] == t:
            p = row['pattern'] # Changed from pattern_name
            if p not in multi_thread_data[t]:
                multi_thread_data[t][p] = {}
            multi_thread_data[t][p][row['N']] = row['sec']

# 3. Best Time (N=Max)
max_n = max(all_ns)
best_times = {} # pattern -> sec
for row in data:
    if row['N'] == max_n:
        p = row['pattern'] # Changed from pattern_name
        s = row['sec']
        if p not in best_times or s < best_times[p]:
            best_times[p] = s
            
# Helper to generate Chart.js datasets
def get_datasets(source_data):
    datasets = []
    colors = [
        'rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(255, 206, 86)', 
        'rgb(75, 192, 192)', 'rgb(153, 102, 255)', 'rgb(255, 159, 64)'
    ]
    for i, p in enumerate(patterns):
        if p in source_data:
            pts = []
            for n in all_ns:
                val = source_data[p].get(n, None)
                pts.append(val)
            datasets.append({
                'label': p,
                'data': pts,
                'borderColor': colors[i % len(colors)],
                'fill': False,
                'tension': 0.1
            })
    return datasets

single_thread_datasets = get_datasets(single_thread_data)

multi_thread_charts = []
for t in thread_counts:
    multi_thread_charts.append({
        'threads': t,
        'datasets': get_datasets(multi_thread_data[t])
    })

best_time_labels = sorted(best_times.keys())
best_time_values = [best_times[k] for k in best_time_labels]


# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Matrix Addition Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #f4f4f9; }}
        .chart-container {{ 
            width: 45%; 
            display: inline-block; 
            margin: 10px; 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        }}
        h1, h2 {{ color: #333; }}
        .row {{ display: flex; flex-wrap: wrap; }}
    </style>
</head>
<body>
    <h1>Benchmark Results</h1>
    
    <div class="row">
        <div class="chart-container" style="width: 95%;">
            <h2>Time vs Matrix Size (1 Thread)</h2>
            <canvas id="chartSingle"></canvas>
        </div>
    </div>

    <div class="row">
        {"".join([f'<div class="chart-container"><h2>Time vs Size ({c["threads"]} Threads)</h2><canvas id="chartMulti{c["threads"]}"></canvas></div>' for c in multi_thread_charts])}
    </div>

    <div class="row">
        <div class="chart-container" style="width: 95%;">
            <h2>Best Time per Pattern (N={max_n})</h2>
            <canvas id="chartBest"></canvas>
        </div>
    </div>

    <script>
        const ns = {json.dumps(all_ns)};

        // Single Thread Chart
        new Chart(document.getElementById('chartSingle'), {{
            type: 'line',
            data: {{
                labels: ns,
                datasets: {json.dumps(single_thread_datasets)}
            }},
            options: {{ scales: {{ y: {{ type: 'logarithmic', title: {{display: true, text: 'Seconds'}} }}, x: {{ title: {{display: true, text: 'N'}} }} }} }}
        }});

        // Multi Thread Charts
        const multiData = {json.dumps(multi_thread_charts)};
        multiData.forEach(item => {{
            new Chart(document.getElementById('chartMulti' + item.threads), {{
                type: 'line',
                data: {{
                    labels: ns,
                    datasets: item.datasets
                }},
                options: {{ scales: {{ y: {{ title: {{display: true, text: 'Seconds'}} }}, x: {{ title: {{display: true, text: 'N'}} }} }} }}
            }});
        }});

        // Best Time Chart
        new Chart(document.getElementById('chartBest'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(best_time_labels)},
                datasets: [{{
                    label: 'Time (s)',
                    data: {json.dumps(best_time_values)},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                }}]
            }},
            options: {{ scales: {{ y: {{ title: {{display: true, text: 'Seconds'}} }} }} }}
        }});
    </script>
</body>
</html>
"""

with open('benchmark_report.html', 'w') as f:
    f.write(html_content)

print("Generated benchmark_report.html")

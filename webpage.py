#!/usr/bin/env python3
"""
webpage.py – build a searchable, sortable HTML table from a CSV.

Expected CSV header:
    start_date,start_time,end_date,end_time,monitored_time,type

Run:
    python webpage.py
"""

from pathlib import Path
import platform
import sys
import webbrowser

import pandas as pd


def build_html_from_csv(csv_path: str = "data.csv",
                        html_path: str = "table.html") -> Path:
    # 1. Read the CSV
    df = pd.read_csv(csv_path)

    # 2. Normalise date strings → YYYY-MM-DD
    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = (
                pd.to_datetime(df[col], errors="coerce")
                  .dt.strftime("%Y-%m-%d")
            )

    # 3. Normalise time strings → HH:MM:SS
    for col in ("start_time", "end_time", "monitored_time"):
        if col in df.columns:
            df[col] = (
                pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce")
                  .dt.strftime("%H:%M:%S")
            )

    # 4. Rename columns for display
    df.rename(columns={
        "start_date":      "Starting Date",
        "start_time":      "Starting Time",
        "end_date":        "End Date",
        "end_time":        "End Time",
        "monitored_time":  "Productive Time",   # ← display label
        "type":            "Type"
    }, inplace=True)

    # 5. Convert to HTML table and wrap in centred div
    table_html = df.to_html(index=False,
                            classes="display nowrap",
                            table_id="myTable")
    wrapped_table = f'<div class="table-container">{table_html}</div>'

    # 6. Build full HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>History</title>

<!-- DataTables & dependencies -->
<link rel="stylesheet"
      href="https://cdn.datatables.net/2.0.3/css/dataTables.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/2.0.3/js/dataTables.min.js"></script>

<!-- Moment.js + plug-in for proper date/time sorting -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/moment.min.js"></script>
<script src="https://cdn.datatables.net/plug-ins/2.0.3/sorting/datetime-moment.js"></script>

<style>
  .table-container {{
    width: fit-content;
    margin: 2rem auto;
  }}
  table.dataTable tbody tr:hover {{
    background:#f6f6f6;
  }}
</style>
</head>
<body>
  <h2>History</h2>

  {wrapped_table}

  <script>
    $.fn.dataTable.moment('YYYY-MM-DD');
    $.fn.dataTable.moment('HH:mm:ss');

    $(document).ready(function () {{
      $('#myTable').DataTable({{
        scrollX: true,
        order: [[0, 'asc']],          // sort by Starting Date first
        columnDefs: [
          {{ type:'datetime', targets:[0,1,2,3,4] }}
        ]
      }});
    }});
  </script>
</body>
</html>"""

    # 7. Write out the file
    out_path = Path(html_path).resolve()
    out_path.write_text(html, encoding="utf-8")
    return out_path


def open_in_browser(path: Path) -> None:
    """Open the generated HTML in the default browser."""
    try:
        webbrowser.open_new_tab(path.as_uri())
    except Exception:
        if platform.system() == "Linux":
            import subprocess
            subprocess.run(["xdg-open", str(path)])
        else:
            raise



html_file = build_html_from_csv()
print(f"✓ HTML written to {html_file}")
print("🚀 Opening in browser …")
open_in_browser(html_file)
sys.exit(0)

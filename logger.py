import csv
from pathlib import Path

def logger(file_path, start_date, start_time, end_date, end_time,monitored_time, event_type):
    header = ["start_date", "start_time", "end_date", "end_time", "monitored_time", "type"]

    need_header = not Path(file_path).is_file() or Path(file_path).stat().st_size == 0

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow([start_date, start_time, end_date, end_time,monitored_time, event_type])

    print(f"Appended event: {start_date} {start_time} → {end_date} {end_time}")

def convert_seconds(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"



import pandas as pd
import numpy as np
import math

# --- 1) Read summary data from CSV (e.g., tmu_summary_updated.csv) ---
df_summary = pd.read_csv('tmu_summary_updated.csv', index_col=0)

# Retrieve the "Overall Total" row to extract the skilled worker's work time
if 'Overall Total' not in df_summary.index:
    print("Error: 'Overall Total' row not found in summary data.")
    exit()

overall = df_summary.loc['Overall Total']
mean = overall['Total Time']  # Use the skilled worker's work time as the standard (mean)
sigma = 0.1 * mean            # Assume standard deviation is 10% of the mean

# --- 2) Generate normal distribution data ---
num_points = 50  # Number of points for the normal distribution curve
x_vals = np.linspace(mean - 3*sigma, mean + 3*sigma, num_points)
y_vals = (1 / (sigma * np.sqrt(2 * math.pi))) * np.exp(-((x_vals - mean)**2) / (2 * sigma**2))

# Create a DataFrame for chart data (only the normal distribution curve)
chart_data = pd.DataFrame({'x': x_vals, 'y': y_vals})

# --- 3) Write to Excel file and create chart ---
output_file = 'tmu_summary_with_chart.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Write the summary data to the "Summary" sheet
    df_summary.to_excel(writer, sheet_name='Summary')
    
    # Write the chart data to the "ChartData" sheet (including header)
    chart_data.to_excel(writer, sheet_name='ChartData', index=False)
    
    # Get workbook and create a worksheet for the chart
    workbook  = writer.book
    chart_ws = workbook.add_worksheet('Chart')
    
    # Create a line chart
    chart = workbook.add_chart({'type': 'line'})
    
    # Add the normal distribution series:
    # In Excel, row 0 is the header; the data starts at row 1.
    # Thus, rows 1 to 50 correspond to the normal distribution data (A2:A51 for x, B2:B51 for y).
    chart.add_series({
        'name':       'Normal Distribution',
        'categories': ['ChartData', 1, 0, num_points, 0],  # X-axis: A2:A51
        'values':     ['ChartData', 1, 1, num_points, 1],  # Y-axis: B2:B51
        'line':       {'color': 'black'}
    })
    
    # Set the chart title and axis labels
    chart.set_title({'name': 'Normal Distribution of Work Time'})
    chart.set_x_axis({'name': 'Work Time'})
    chart.set_y_axis({'name': 'Probability Density'})
    
    # Insert the chart into the "Chart" sheet
    chart_ws.insert_chart('B2', chart, {'x_scale': 2.0, 'y_scale': 1.5})

print(f"Excel file {output_file} with the chart has been saved.")

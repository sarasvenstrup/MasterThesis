import pandas as pd
import os

# Path to the SwapVol_OIS.xlsx file
EXCEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'SwapVol_OIS.xlsx')

# Read the sheet, get both the code row (option/swap) and the data row
header_rows = [1, 4]  # 2nd row (index 1) for codes, 5th row (index 4) for data
raw = pd.read_excel(EXCEL_PATH, sheet_name=0, header=None)

# Extract the code row (row 1, index 1)
code_row = raw.iloc[0]
# Extract the data starting from row 5 (index 4)
data = pd.read_excel(EXCEL_PATH, sheet_name=0, header=4)

# Debug prints
data_columns = list(data.columns)
print('Code row:', list(code_row))
print('Data columns:', data_columns)
print('First 6 rows of raw Excel:')
print(raw.head(6))

# Build a mapping from column index to (option_maturity, swap_tenor)
col_map = {}
for idx, col in enumerate(data_columns):
    code = code_row[idx] if idx < len(code_row) else None
    code_str = str(code).strip() if code is not None else ''
    if code_str.isdigit():
        if len(code_str) == 2:
            option_maturity = int(code_str[0])
            swap_tenor = int(code_str[1])
        elif len(code_str) == 3:
            option_maturity = int(code_str[0:2])
            swap_tenor = int(code_str[2])
        elif len(code_str) == 4:
            option_maturity = int(code_str[0:2])
            swap_tenor = int(code_str[2:])
        else:
            option_maturity = None
            swap_tenor = None
        col_map[col] = (option_maturity, swap_tenor)
    else:
        col_map[col] = (None, None)

# Identify the date column (second column)
date_col = data.columns[1]

# Remove columns that are all NaN or irrelevant
cols_to_keep = [col for col in data.columns if not data[col].isnull().all()]
data = data[cols_to_keep]

# Melt the DataFrame to long format
melted = data.melt(id_vars=[date_col], var_name='swap_col', value_name='vol')

# Map option_maturity and swap_tenor using col_map
melted['option_maturity'] = melted['swap_col'].map(lambda c: col_map.get(c, (None, None))[0])
melted['swap_tenor'] = melted['swap_col'].map(lambda c: col_map.get(c, (None, None))[1])

# Filter out rows where both option_maturity and swap_tenor are None (i.e., not a swap vol column)
melted = melted[~(melted['option_maturity'].isna() & melted['swap_tenor'].isna())]

# Add currency and rename date column
melted['currency'] = 'EUR'
melted = melted.rename(columns={date_col: 'as_of_date'})

# Ensure as_of_date is datetime
melted['as_of_date'] = pd.to_datetime(melted['as_of_date'], errors='coerce')

# Select and reorder columns
final_cols = ['currency', 'as_of_date', 'option_maturity', 'swap_tenor', 'vol']
melted = melted[final_cols]

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), '..', 'SwapVol_OIS_formatted.csv')
melted.to_csv(output_path, index=False)
print(melted.head())
print(f"Saved formatted data to {output_path}")

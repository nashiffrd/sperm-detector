import numpy as np
import pandas as pd

def compute_velocity(df, fps=30):
    df = df.sort_values(['particle', 'frame'])
    df['dx'] = df.groupby('particle')['x'].diff()
    df['dy'] = df.groupby('particle')['y'].diff()
    df['step_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['velocity'] = df['step_distance'] * fps
    return df.groupby('particle')['velocity'].mean().reset_index(name='mean_velocity')

def compute_linearity(df):
    records = []

    for p in df['particle'].unique():
        sub = df[df['particle'] == p].sort_values('frame')
        dx = np.diff(sub['x'])
        dy = np.diff(sub['y'])
        straight = np.sqrt((sub['x'].iloc[-1]-sub['x'].iloc[0])**2 +
                           (sub['y'].iloc[-1]-sub['y'].iloc[0])**2)
        total = np.sum(np.sqrt(dx**2 + dy**2))
        linearity = straight / total if total > 0 else 0
        records.append({'particle': p, 'linearity': linearity})

    return pd.DataFrame(records)
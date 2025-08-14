from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
import os

# --- Flask App Initialization ---
app = Flask(__name__)

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. Data and Properties Setup ---
ELEMENT_PROPERTIES = {
    'Al': {'radius': 143, 'electronegativity': 1.61, 'VEC': 3, 'Tm': 933.47, 'dHmix_base': {}},
    'Co': {'radius': 125, 'electronegativity': 1.88, 'VEC': 9, 'Tm': 1768, 'dHmix_base': {}},
    'Cr': {'radius': 128, 'electronegativity': 1.66, 'VEC': 6, 'Tm': 2180, 'dHmix_base': {}},
    'Fe': {'radius': 126, 'electronegativity': 1.83, 'VEC': 8, 'Tm': 1811, 'dHmix_base': {}},
    'Ni': {'radius': 124, 'electronegativity': 1.91, 'VEC': 10, 'Tm': 1728, 'dHmix_base': {}},
    'Cu': {'radius': 128, 'electronegativity': 1.90, 'VEC': 11, 'Tm': 1357.77, 'dHmix_base': {}},
    'Mn': {'radius': 127, 'electronegativity': 1.55, 'VEC': 7, 'Tm': 1519, 'dHmix_base': {}},
    'Ti': {'radius': 147, 'electronegativity': 1.54, 'VEC': 4, 'Tm': 1941, 'dHmix_base': {}},
    'V':  {'radius': 134, 'electronegativity': 1.63, 'VEC': 5, 'Tm': 2183, 'dHmix_base': {}},
    'Nb': {'radius': 146, 'electronegativity': 1.6, 'VEC': 5, 'Tm': 2750, 'dHmix_base': {}},
    'Mo': {'radius': 139, 'electronegativity': 2.16, 'VEC': 6, 'Tm': 2896, 'dHmix_base': {}},
    'Zr': {'radius': 160, 'electronegativity': 1.33, 'VEC': 4, 'Tm': 2128, 'dHmix_base': {}},
    'Hf': {'radius': 159, 'electronegativity': 1.3, 'VEC': 4, 'Tm': 2506, 'dHmix_base': {}},
    'Ta': {'radius': 146, 'electronegativity': 1.5, 'VEC': 5, 'Tm': 3290, 'dHmix_base': {}},
    'W':  {'radius': 139, 'electronegativity': 2.36, 'VEC': 6, 'Tm': 3695, 'dHmix_base': {}},
    'C':  {'radius': 75, 'electronegativity': 2.55, 'VEC': 4, 'Tm': 3800, 'dHmix_base': {}},
    'Mg': {'radius': 160, 'electronegativity': 1.31, 'VEC': 2, 'Tm': 923, 'dHmix_base': {}},
    'Zn': {'radius': 134, 'electronegativity': 1.65, 'VEC': 12, 'Tm': 692.68, 'dHmix_base': {}},
    'Si': {'radius': 116, 'electronegativity': 1.90, 'VEC': 4, 'Tm': 1687, 'dHmix_base': {}},
    'Re': {'radius': 137, 'electronegativity': 1.9, 'VEC': 7, 'Tm': 3459, 'dHmix_base': {}},
    'N':  {'radius': 71, 'electronegativity': 3.04, 'VEC': 5, 'Tm': 63.15, 'dHmix_base': {}},
    'Sc': {'radius': 162, 'electronegativity': 1.36, 'VEC': 3, 'Tm': 1814, 'dHmix_base': {}},
    'Li': {'radius': 152, 'electronegativity': 0.98, 'VEC': 1, 'Tm': 453.65, 'dHmix_base': {}},
    'Sn': {'radius': 140, 'electronegativity': 1.96, 'VEC': 4, 'Tm': 505.08, 'dHmix_base': {}},
    'Be': {'radius': 112, 'electronegativity': 1.57, 'VEC': 2, 'Tm': 1560, 'dHmix_base': {}}
}
dHmix_data = {
    ('Al', 'Co'): -19, ('Al', 'Cr'): -10, ('Al', 'Fe'): -11, ('Al', 'Ni'): -22, ('Al', 'Ti'): -30,
    ('Co', 'Cr'): -4, ('Co', 'Fe'): -1, ('Co', 'Ni'): 0, ('Co', 'Ti'): -28, ('Cr', 'Fe'): 1,
    ('Cr', 'Ni'): -7, ('Cr', 'Ti'): -7, ('Fe', 'Ni'): -2, ('Fe', 'Ti'): -17, ('Ni', 'Ti'): -35
}
for (el1, el2), val in dHmix_data.items():
    ELEMENT_PROPERTIES[el1]['dHmix_base'][el2] = val
    ELEMENT_PROPERTIES[el2]['dHmix_base'][el1] = val

# --- Global variables for the trained model objects ---
MODEL = None
SCALER = None
FEATURE_COLS = None
PHASE_MAP = None

def train_model_and_scaler():
    """
    This function trains the model and scaler ONCE and stores them globally.
    """
    global MODEL, SCALER, FEATURE_COLS, PHASE_MAP
    
    # Construct the path to the CSV file relative to the script's location
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'HEA.csv')
    
    df1 = pd.read_csv(csv_path, encoding='Latin1')
    df = df1.copy()
    df.drop_duplicates(subset='Alloy ', inplace=True)
    df.drop(columns=['Alloy ', 'Alloy ID', 'References', 'Unnamed: 51', 'Unnamed: 52'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['Annealing_Time_(min)', 'Homogenization_Time', 'Homogenization_Temp', 'Hot-Cold_Working', 'Quenching', 'Annealing_Temp', 'HPR'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['Microstructure_', 'Microstructure', 'Multiphase', 'IM_Structure'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['omega'] = (df['Tm'] * df['dSmix']) / np.abs(df['dHmix'])
    df['lambda'] = df['dSmix'] / (df['Atom.Size.Diff']**2)
    df['r_avg'], df['r_minimum'], df['r_maximum'], df['gamma'] = [0, 0, 0, 0]
    df['Phases'].replace(['BCC_SS', 'FCC_SS', 'FCC_PLUS_BCC', 'Im'], [0, 1, 2, 3], inplace=True)
    df['Sythesis_Route'].replace(['AC', 'PM'], [0, 1], inplace=True)
    df.drop(columns=['W', 'Re', 'N', 'Sc', 'Be', 'Sn', 'Li'], inplace=True)
    
    x = df.drop(columns=['Phases'])
    y = df['Phases']
    
    smote = SMOTE(random_state=42)
    x_balanced, y_balanced = smote.fit_resample(x, y)
    
    SCALER = StandardScaler()
    x_scaled = SCALER.fit_transform(x_balanced)
    
    MODEL = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=50)
    MODEL.fit(x_scaled, y_balanced)
    
    FEATURE_COLS = list(x.columns)
    PHASE_MAP = {0: 'BCC_SS', 1: 'FCC_SS', 2: 'FCC_PLUS_BCC', 3: 'Im'}
    print("Model trained and ready.")

def calculate_features(composition, processing_temp_k=None, synthesis_route='AC'):
    elements = list(composition.keys())
    fractions = np.array(list(composition.values()))
    radii = np.array([ELEMENT_PROPERTIES[el]['radius'] for el in elements])
    electronegativities = np.array([ELEMENT_PROPERTIES[el]['electronegativity'] for el in elements])
    vecs = np.array([ELEMENT_PROPERTIES[el]['VEC'] for el in elements])
    tms = np.array([ELEMENT_PROPERTIES[el]['Tm'] for el in elements])
    r_avg = np.sum(fractions * radii)
    elect_avg = np.sum(fractions * electronegativities)
    atom_size_diff = np.sqrt(np.sum(fractions * ((1 - radii / r_avg)**2))) * 100
    elect_diff = np.sqrt(np.sum(fractions * ((electronegativities - elect_avg)**2)))
    vec = np.sum(fractions * vecs)
    tm = np.sum(fractions * tms)
    dSmix = -8.314 * np.sum(fractions * np.log(fractions)) if np.all(fractions > 0) else 0
    dHmix = 0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el1, el2 = elements[i], elements[j]
            dH = ELEMENT_PROPERTIES[el1]['dHmix_base'].get(el2, 0)
            dHmix += 4 * dH * fractions[i] * fractions[j]
    
    temp_for_g = processing_temp_k if processing_temp_k is not None else tm
    dGmix = dHmix - temp_for_g * dSmix / 1000

    omega = (tm * dSmix) / (np.abs(dHmix) * 1000) if dHmix != 0 else 0
    lambda_param = dSmix / (atom_size_diff**2) if atom_size_diff != 0 else 0
    r_min, r_max = np.min(radii), np.max(radii)
    term_min = (1 - np.sqrt(((r_min + r_avg)**2 - (r_avg**2)) / (r_min + r_avg)**2))
    term_max = (1 - np.sqrt(((r_max + r_avg)**2 - (r_avg**2)) / (r_max + r_avg)**2))
    gamma = term_min / term_max if term_max != 0 else 0
    
    route_code = 1 if synthesis_route == 'PM' else 0

    features = {
        'Num_of_Elem': len(elements), 'Density_calc': 0, 'dHmix': dHmix, 'dSmix': dSmix,
        'dGmix': dGmix, 'Tm': tm, 'n.Para': 0, 'Atom.Size.Diff': atom_size_diff,
        'Elect.Diff': elect_diff, 'VEC': vec, 'Sythesis_Route': route_code, 'omega': omega,
        'lambda': lambda_param, 'r_avg': r_avg, 'r_minimum': r_min, 'r_maximum': r_max, 'gamma': gamma
    }
    for el in ELEMENT_PROPERTIES:
        features[el] = composition.get(el, 0)
    return features

# --- Train the model on startup ---
train_model_and_scaler()

# --- API Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict_hea_phase_api():
    try:
        data = request.get_json()
        composition_dict = data.get('composition')
        temp_c = data.get('temperature')
        synthesis_route = data.get('route', 'AC')

        total_percentage = sum(composition_dict.values())
        if total_percentage == 0:
            return jsonify({'error': 'Composition cannot be empty.'}), 400
        
        normalized_composition = {el: perc / total_percentage for el, perc in composition_dict.items()}
        
        processing_temp_k = (temp_c + 273.15) if temp_c is not None else None
        
        new_alloy_features = calculate_features(normalized_composition, processing_temp_k, synthesis_route)
        new_alloy_df = pd.DataFrame([new_alloy_features])[FEATURE_COLS]
        new_alloy_scaled = SCALER.transform(new_alloy_df)
        prediction_code = MODEL.predict(new_alloy_scaled)[0]
        predicted_phase = PHASE_MAP[int(prediction_code)]
        
        return jsonify({'predicted_phase': predicted_phase})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- This block runs the local development server ---
if __name__ == '__main__':
    app.run(debug=True)

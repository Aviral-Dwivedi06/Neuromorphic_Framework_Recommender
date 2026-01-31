import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

class NeuromorphicModel:
    def __init__(self):
        # We use random_state for consistent results
        self.model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        self.encoders = {}
        self.output_cols = ['Neuron_Model', 'Architecture', 'Device_Type', 'Est_Performance', 'Rationale']

    def train(self, data_path):
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        # OPTIMIZATION: Convert App Type to lowercase for training
        df['Application_Type'] = df['Application_Type'].str.strip().str.lower()
        
        self.encoders['Application_Type'] = LabelEncoder()
        df['App_Encoded'] = self.encoders['Application_Type'].fit_transform(df['Application_Type'])
        
        Y = df[self.output_cols].copy()
        for col in self.output_cols:
            self.encoders[col] = LabelEncoder()
            Y[col] = self.encoders[col].fit_transform(Y[col])
            
        X = df[['Tech_Node_nm', 'Power_mW', 'App_Encoded']]
        self.model.fit(X, Y)

    def recommend(self, node, power, app_type):
        app_clean = app_type.strip().lower()
        if app_clean not in self.encoders['Application_Type'].classes_:
            raise ValueError(f"Application '{app_type}' not recognized.")

        app_enc = self.encoders['Application_Type'].transform([app_clean])[0]
        input_data = pd.DataFrame([[node, power, app_enc]], 
                                 columns=['Tech_Node_nm', 'Power_mW', 'App_Encoded'])
        
        prediction = self.model.predict(input_data)[0]
        results = {}
        for i, col in enumerate(self.output_cols):
            results[col] = self.encoders[col].inverse_transform([int(prediction[i])])[0]

        # --- RATIONALE INTELLIGENCE LAYER ---
        warnings = []
        
        # Rule 1: Leakage vs. Ultra-low power
        # 10nW is 0.00001 mW. Advanced nodes (28nm) have high leakage.
        if power < 0.0001 and node < 40:
            warnings.append("⚠️ WARNING: Static leakage in <40nm nodes may exceed your power budget.")
            
        # Rule 2: Precision vs. Device Type
        if "Memristive" in results['Device_Type'] and "CNN" in app_type.upper():
            warnings.append("⚠️ NOTE: RRAM/Memristive weights may require error-correction for high-precision CNNs.")

        if warnings:
            results['Rationale'] = results['Rationale'] + " | " + " ".join(warnings)

        return results
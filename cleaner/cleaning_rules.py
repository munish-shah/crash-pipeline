import pandas as pd
import json
    
def drop_cols(df):
    cols_to_drop =  [
        "report_type",
        "photos_taken_i", 
        "statements_taken_i",
        "date_police_notified",
        "veh_vehicle_id_list_json",
        "ppl_person_id_list_json",
        "location_json",
        "street_name",
        "first_crash_type",
        # QC-identified leakage columns (post-crash outcomes)
        "injuries_fatal",
        "injuries_incapacitating",
        "injuries_no_indication",
        "injuries_non_incapacitating",
        "injuries_reported_not_evident",
        "injuries_total",
        "injuries_unknown",
        "most_severe_injury"
    ]

    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    df = df.drop(columns=cols_to_drop)
        
    return df

def standardize_boolean_columns(df):
        cols_to_convert = [c for c in df.columns if '_i' in c[-2:]]
        for col in cols_to_convert:
            df[col] = df[col].map({'Y': 1, 'N': 0})
        return df

def create_time_features(df):
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['year'] = df['crash_date'].dt.year
    df['month'] = df['crash_date'].dt.month
    df['day'] = df['crash_date'].dt.day
    df['hour'] = df['crash_date'].dt.hour
    df['day_of_week'] = df['crash_date'].dt.dayofweek  # QC-identified missing feature (0=Monday, 6=Sunday)
    df['is_weekend'] = (df['crash_date'].dt.dayofweek >= 5).astype(int)
    df['hour_bin'] = pd.cut(df['hour'], bins=[0, 7, 13, 19, 24], labels=['night', 'morning', 'afternoon', 'evening'], right=False)
    
    return df 

def clean_location_features(df):
    df = df[(df['latitude'] > 41.6) & (df['latitude'] < 42.1) & 
            (df['longitude'] > -87.9) & (df['longitude'] < -87.5)].copy()
    
    df['lat_bin'] = df['latitude'].round(2)
    df['lng_bin'] = df['longitude'].round(2)
    df['grid_id'] = df['lat_bin'].astype(str) + '_' + df['lng_bin'].astype(str)
    
    return df

def clean_road_environment(df):
    text_cols = ['roadway_surface_cond', 'lighting_condition', 'weather_condition', 
                 'traffic_control_device']
    
    for col in text_cols:
        df[col] = df[col].str.lower().str.strip().replace(['unknown', 'unk', 'n/a'], None)
    
    return df

def parse_vehicle_features(df):
    if 'veh_unit_type_list_json' in df.columns:
        df['veh_truck_i'] = df['veh_unit_type_list_json'].str.contains('truck', case=False, na=False).astype(int)
        df['veh_mc_i'] = df['veh_unit_type_list_json'].str.contains('motorcycle', case=False, na=False).astype(int)
    
    return df

def parse_people_features(df):
    if 'ppl_age_list_json' in df.columns:
        ages = df['ppl_age_list_json'].apply(lambda x: 
            [float(a) for a in json.loads(x) if a is not None and 0 < float(a) < 110] 
            if pd.notna(x) and x != '[]' else []
        )
        
        df['ppl_age_mean'] = ages.apply(lambda x: sum(x)/len(x) if len(x) > 0 else None)
        df['ppl_age_min'] = ages.apply(lambda x: min(x) if len(x) > 0 else None)
        df['ppl_age_max'] = ages.apply(lambda x: max(x) if len(x) > 0 else None)
    
    return df

def group_contributory_causes(df):
    def categorize(cause):
        if pd.isna(cause):
            return 'Other'
        cause = str(cause).upper()
        
        if 'SPEED' in cause or 'EXCEED' in cause:
            return 'Speeding'
        elif 'ALCOHOL' in cause or 'DRUG' in cause or 'INFLUENCE' in cause:
            return 'DUI/Impairment'
        elif 'DISTRACTION' in cause or 'CELL PHONE' in cause:
            return 'Distraction/Inattention'
        elif 'YIELD' in cause:
            return 'Failure-to-Yield'
        elif 'FOLLOWING TOO CLOSELY' in cause:
            return 'Following Too Closely'
        elif 'WEATHER' in cause:
            return 'Weather-related'
        elif 'VISION OBSCURED' in cause:
            return 'Lighting/Visibility'
        else:
            return 'Other'
    
    df['contributory_cause_group'] = df['prim_contributory_cause'].apply(categorize)
    return df

def handle_missing_values(df):

    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing')
    
    return df


def handle_outliers(df):
    if 'ppl_count' in df.columns:
        df['ppl_count'] = df['ppl_count'].clip(upper=10)
    if 'veh_count' in df.columns:
        df['veh_count'] = df['veh_count'].clip(upper=5)
    # injuries_total removed (data leakage - post-crash outcome)
    
    for col in ['ppl_age_mean', 'ppl_age_min', 'ppl_age_max']:
        if col in df.columns:
            df[col] = df[col].clip(upper=100)
    
    return df

def binarize_crash_type(df):
    df['crash_type'] = df['crash_type'].str.upper().map({
        'INJURY AND / OR TOW DUE TO CRASH': 1,
        'NO INJURY / DRIVE AWAY': 0
    })
    return df

def clean_dataframe(df):
    df = drop_cols(df)
    df = standardize_boolean_columns(df)
    df = create_time_features(df)
    df = clean_location_features(df)
    df = clean_road_environment(df)
    df = parse_vehicle_features(df)
    df = parse_people_features(df)
    df = group_contributory_causes(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = binarize_crash_type(df)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("merged.csv")
    df = clean_dataframe(df)
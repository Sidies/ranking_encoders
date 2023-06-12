import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

categorical_columns = [
    'building_id',
    'geo_level_1_id',
    'geo_level_2_id',
    'geo_level_3_id',
    'land_surface_condition',
    'foundation_type',
    'roof_type',
    'ground_floor_type',
    'other_floor_type',
    'position',
    'plan_configuration',
    'has_superstructure_adobe_mud',
    'has_superstructure_mud_mortar_stone',
    'has_superstructure_stone_flag',
    'has_superstructure_cement_mortar_stone',
    'has_superstructure_mud_mortar_brick',
    'has_superstructure_cement_mortar_brick',
    'has_superstructure_timber',
    'has_superstructure_bamboo',
    'has_superstructure_rc_non_engineered',
    'has_superstructure_rc_engineered',
    'has_superstructure_other',
    'legal_ownership_status',
    'has_secondary_use',
    'has_secondary_use_agriculture',
    'has_secondary_use_hotel',
    'has_secondary_use_rental',
    'has_secondary_use_institution',
    'has_secondary_use_school',
    'has_secondary_use_industry',
    'has_secondary_use_health_post',
    'has_secondary_use_gov_office',
    'has_secondary_use_use_police',
    'has_secondary_use_other'
]

numerical_columns = [
    'count_floors_pre_eq',
    'age',
    'area_percentage',
    'height_percentage',
    'count_families'
]
        
has_secondary_use_columns = [
    'has_secondary_use',
    'has_secondary_use_agriculture', 
    'has_secondary_use_hotel',
    'has_secondary_use_rental', 
    'has_secondary_use_institution',
    'has_secondary_use_school', 
    'has_secondary_use_industry',
    'has_secondary_use_health_post', 
    'has_secondary_use_gov_office',
    'has_secondary_use_use_police', 
    'has_secondary_use_other'
]
    
has_superstructure_columns = [
    'has_superstructure_adobe_mud',
    'has_superstructure_mud_mortar_stone', 
    'has_superstructure_stone_flag',
    'has_superstructure_cement_mortar_stone',
    'has_superstructure_mud_mortar_brick',
    'has_superstructure_cement_mortar_brick', 
    'has_superstructure_timber',
    'has_superstructure_bamboo', 
    'has_superstructure_rc_non_engineered',
    'has_superstructure_rc_engineered', 
    'has_superstructure_other'
]
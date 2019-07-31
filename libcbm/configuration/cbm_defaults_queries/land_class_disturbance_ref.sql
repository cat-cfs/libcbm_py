select
disturbance_type.id as disturbance_type_id,
disturbance_type_tr.name as disturbance_type_name,
land_class.id as land_class_id,
land_class.code as land_class_code,
land_class_tr.description as land_class_description
from disturbance_type
inner join land_class on disturbance_type.transition_land_class_id = land_class.id
inner join land_class_tr on land_class_tr.land_class_id = land_class.id
inner join disturbance_type_tr on disturbance_type_tr.disturbance_type_id == disturbance_type.id
inner join locale dt_loc on disturbance_type_tr.locale_id = dt_loc.id
inner join locale lc_loc on land_class_tr.locale_id = lc_loc.id
where dt_loc.code = ? and lc_loc.code = ?
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
inner join locale on disturbance_type_tr.locale_id = locale.id
where locale.code = ? and land_class_tr.locale_id = disturbance_type_tr.locale_id
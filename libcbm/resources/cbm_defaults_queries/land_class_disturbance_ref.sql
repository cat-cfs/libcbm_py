select
disturbance_type.id as disturbance_type_id,
disturbance_type_tr.name as disturbance_type_name,
land_type.id as land_type_id
from disturbance_type
inner join land_type on disturbance_type.land_type_id = land_type.id
inner join disturbance_type_tr on disturbance_type_tr.disturbance_type_id == disturbance_type.id
inner join locale on disturbance_type_tr.locale_id = locale.id
where locale.code = ?
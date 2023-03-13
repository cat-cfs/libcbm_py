select
disturbance_type.id as disturbance_type_id,
disturbance_type_tr.name as disturbance_type_name,
disturbance_type_tr.description as disturbance_type_description
from disturbance_type
inner join disturbance_type_tr on disturbance_type_tr.disturbance_type_id == disturbance_type.id
inner join locale on disturbance_type_tr.locale_id = locale.id
where locale.code = ?
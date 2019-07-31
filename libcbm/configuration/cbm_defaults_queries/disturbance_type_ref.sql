select  disturbance_type.id, disturbance_type_tr.name
from disturbance_type
inner join disturbance_type_tr on disturbance_type_tr.disturbance_type_id == disturbance_type.id
inner join locale on disturbance_type_tr.locale_id = locale.id
where locale.code = ?
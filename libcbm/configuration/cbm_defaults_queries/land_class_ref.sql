select
land_class.id as land_class_id, land_class.code, land_class_tr.description
from land_class
inner join land_class_tr on land_class_tr.land_class_id = land_class.id
inner join locale on land_class_tr.locale_id = locale.id
where locale.code = ?;
select
land_class.id as land_class_id,
land_class.code,
land_class_tr.description,
land_class.is_forest,
land_class.is_simulated,
land_class.transitional_period,
land_class.transition_id,
land_class.land_type_id_1,
land_class.land_type_id_2
from land_class
inner join land_class_tr on land_class_tr.land_class_id = land_class.id
inner join locale on land_class_tr.locale_id = locale.id
where locale.code = ?;
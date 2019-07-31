select afforestation_pre_type.id, afforestation_pre_type_tr.name
from afforestation_pre_type inner join afforestation_pre_type_tr
on afforestation_pre_type_tr.afforestation_pre_type_id = afforestation_pre_type.id
inner join locale on afforestation_pre_type_tr.locale_id = locale.id
where locale.code = ? and afforestation_pre_type.id>0
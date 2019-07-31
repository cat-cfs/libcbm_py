select species_tr.name, species.id, species.forest_type_id
from species
inner join species_tr on species_tr.species_id = species.id
inner join locale on species_tr.locale_id = locale.id
where locale.code = ?
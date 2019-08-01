select
species.id as species_id,
species_tr.name as species_name,
species.genus_id as genus_id,
genus_tr.name as genus_name,
species.forest_type_id as forest_type_id,
forest_type_tr.name as forest_type_name
from species
inner join species_tr on species_tr.species_id = species.id
inner join genus on species.genus_id = genus.id
inner join genus_tr on genus.id = genus_tr.genus_id
inner join forest_type on species.forest_type_id = forest_type.id
inner join forest_type_tr on forest_type.id = forest_type_tr.forest_type_id
inner join locale on species_tr.locale_id = locale.id
where locale.code = ? 
    and genus_tr.locale_id = species_tr.locale_id
    and forest_type_tr.locale_id = species_tr.locale_id
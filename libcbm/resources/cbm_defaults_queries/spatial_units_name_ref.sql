select spatial_unit.id as spatial_unit_id, admin_boundary_tr.name as admin_boundary_name,
eco_boundary_tr.name as eco_boundary_name from spatial_unit
inner join eco_boundary on eco_boundary.id = spatial_unit.eco_boundary_id
inner join admin_boundary on admin_boundary.id = spatial_unit.admin_boundary_id
inner join eco_boundary_tr on eco_boundary_tr.eco_boundary_id = eco_boundary.id
inner join admin_boundary_tr on admin_boundary_tr.admin_boundary_id = admin_boundary.id
inner join locale on admin_boundary_tr.locale_id = locale.id
where admin_boundary_tr.locale_id = eco_boundary_tr.locale_id and locale.code = ?
order by spatial_unit.id
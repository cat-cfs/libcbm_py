select
pool.id as Pool,
decay_parameter.base_decay_rate as OrganicMatterDecayRate,
decay_parameter.reference_temp as ReferenceTemp,
decay_parameter.q10 as Q10,
decay_parameter.prop_to_atmosphere as PropToAtmosphere,
decay_parameter.max_rate as MaxDecayRate
from decay_parameter
inner join dom_pool on dom_pool.id = decay_parameter.dom_pool_id
inner join pool on pool.id = dom_pool.pool_id;
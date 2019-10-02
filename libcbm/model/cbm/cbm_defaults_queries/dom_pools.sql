select pool.id, pool.code 
from dom_pool 
inner join pool 
on pool.id == dom_pool.pool_id
order by pool.id;
